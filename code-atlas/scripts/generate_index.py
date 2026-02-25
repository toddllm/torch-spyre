#!/usr/bin/env python3
"""Generate code-atlas data artifacts from local repository clones."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_permalinks import build_permalink, normalize_github_origin
from extract_snippets import (
    compute_context_bounds,
    detect_language,
    extract_target_ranges,
    slice_lines,
    with_line_numbers,
)


@dataclass
class RepoInfo:
    name: str
    root: Path
    head_sha: str
    head_short: str
    head_ref: str
    is_dirty: bool
    head_in_origin_refs: bool | None
    origin: str
    github_base: str


class SymbolCollector(ast.NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.stack: list[str] = []
        self.items: list[dict[str, Any]] = []

    def _push_symbol(self, node: ast.AST, kind: str, name: str) -> None:
        qualname = ".".join([*self.stack, name]) if self.stack else name
        start_line = int(getattr(node, "lineno", 1))
        end_line = int(getattr(node, "end_lineno", start_line))
        self.items.append(
            {
                "module": self.module,
                "kind": kind,
                "name": name,
                "qualname": qualname,
                "start_line": start_line,
                "end_line": end_line,
            }
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._push_symbol(node, "class", node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._push_symbol(node, "function", node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._push_symbol(node, "async_function", node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def run(cmd: list[str], cwd: Path) -> str:
    out = subprocess.check_output(cmd, cwd=str(cwd), text=True)
    return out.strip()


def run_result(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def check_head_in_origin_refs(root: Path, head_sha: str) -> bool | None:
    proc = run_result(["git", "branch", "-r", "--contains", head_sha], root)
    if proc.returncode != 0:
        return None

    refs = [line.strip().lstrip("* ").strip() for line in proc.stdout.splitlines() if line.strip()]
    if not refs:
        return False
    return any(ref.startswith("origin/") for ref in refs)


def read_lines_from_git_blob(repo_root: Path, ref: str, rel_path: str) -> list[str] | None:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{rel_path}"],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace").splitlines()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_json(path: Path | None) -> Any | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return load_json(path)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def merge_seam_narratives(
    seams: list[dict[str, Any]], seam_narratives: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not seam_narratives:
        return seams

    by_id = seam_narratives.get("seams", {})
    merged: list[dict[str, Any]] = []
    for seam in seams:
        seam_id = seam.get("id")
        seam_overlay = by_id.get(seam_id, {})
        seam_out = dict(seam)
        seam_out.update(seam_overlay)
        merged.append(seam_out)
    return merged


def build_selection_reason(extractor: str, anchor: str, matched: str) -> str:
    matched_short = " ".join(str(matched).split())
    if len(matched_short) > 140:
        matched_short = matched_short[:137] + "..."

    if extractor == "symbol":
        return f"Matched symbol `{anchor}` (resolved: `{matched_short}`)"
    if extractor == "regex":
        return f"Matched regex `{anchor}` around line text: `{matched_short}`"
    if extractor == "literal":
        return f"Matched literal anchor `{anchor}` around line text: `{matched_short}`"
    if extractor == "literal_any":
        return f"Matched one-of literal anchors `{anchor}` around line text: `{matched_short}`"
    if extractor == "contains_all":
        return f"Matched all required literals `{anchor}` around line text: `{matched_short}`"
    if extractor == "range":
        return f"Used explicit line range `{anchor}`"
    if extractor == "grep":
        return f"Matched grep pattern `{anchor}` around line text: `{matched_short}`"
    return f"Extractor `{extractor}` matched `{matched_short}`"


def evidence_strength(extractor: str) -> dict[str, Any]:
    ex = (extractor or "").strip().lower()
    if ex in {"symbol", "range"}:
        return {"tier": "strong", "score": 1.0}
    if ex in {"literal", "literal_any", "contains_all"}:
        return {"tier": "strong", "score": 0.9}
    if ex in {"regex", "grep"}:
        return {"tier": "legacy", "score": 0.5}
    return {"tier": "unknown", "score": 0.3}


def infer_anchor_type(target: dict[str, Any]) -> str:
    if "symbol" in target:
        return "symbol"
    if "literal" in target:
        return "literal"
    if "literal_any" in target:
        return "literal_any"
    if "contains_all" in target:
        return "contains_all"
    if "start" in target and "end" in target:
        return "range"
    if "regex" in target or "grep" in target:
        return "regex"
    return "unknown"


def infer_evidence_kind(repo_name: str, file_path: str) -> str:
    if repo_name == "vllm":
        if file_path.startswith("docs/"):
            return "design_reference"
        if file_path.startswith("examples/"):
            return "runtime_example"
        return "upstream_contract"
    if repo_name == "vllm-spyre":
        if file_path.startswith("vllm_spyre_next/"):
            return "target_state"
        return "current_state"
    if repo_name == "torch-spyre":
        return "runtime_constraints"
    return "supplemental"


def validate_target_schema(seam_id: str, target: dict[str, Any]) -> None:
    required_keys = ["repo", "file"]
    missing = [k for k in required_keys if k not in target]
    if missing:
        raise ValueError(f"[{seam_id}] target missing required keys: {missing} target={target}")

    anchor_type = infer_anchor_type(target)
    if anchor_type == "unknown":
        raise ValueError(
            f"[{seam_id}] target has no recognized anchor type "
            f"(symbol/literal/literal_any/contains_all/range): {target}"
        )

    anchor_keys = [
        "symbol",
        "literal",
        "literal_any",
        "contains_all",
        "start",
        "regex",
        "grep",
    ]
    present = [k for k in anchor_keys if k in target]
    # range uses start+end and counts as one anchor mode.
    if "start" in present and "end" in target:
        present = [k for k in present if k not in {"start"}]
        present.append("range")
    if len(set(present)) > 1:
        # allow start+end as one mode only.
        if not (set(present) == {"range"}):
            raise ValueError(
                f"[{seam_id}] target mixes multiple anchor styles {present}; "
                f"use one anchor style per target. target={target}"
            )


def default_target_role(repo_name: str, file_path: str, seam_id: str) -> str:
    if repo_name == "vllm":
        return f"Upstream reference anchor for seam `{seam_id}`"
    if repo_name == "vllm-spyre" and file_path.startswith("vllm_spyre_next/"):
        return f"vllm-spyre-next scaffold/target-state anchor for seam `{seam_id}`"
    if repo_name == "vllm-spyre":
        return f"Current vllm-spyre behavior anchor for seam `{seam_id}`"
    if repo_name == "torch-spyre":
        return f"torch-spyre compiler/runtime constraint anchor for seam `{seam_id}`"
    return f"{repo_name} evidence anchor for seam `{seam_id}`"


def default_target_note(
    evidence_kind: str,
    seam_question: str,
    file_path: str,
    anchor_type: str,
    anchor_value: str,
) -> str:
    prefix = {
        "upstream_contract": "Treat this as the upstream contract/reference point",
        "design_reference": "Treat this as design-document reference evidence",
        "runtime_example": "Treat this as runtime example evidence",
        "current_state": "Treat this as current vllm-spyre behavior evidence",
        "target_state": "Treat this as vllm-spyre-next target-state/scaffold evidence",
        "runtime_constraints": "Treat this as torch-spyre runtime constraint evidence",
    }.get(evidence_kind, "Treat this as supporting seam evidence")
    q = f" Evaluate: {seam_question}" if seam_question else ""
    anchor = f" Anchor({anchor_type}): {anchor_value}" if anchor_value else ""
    return f"{prefix} in `{file_path}`.{q}{anchor}"


def default_target_takeaway(evidence_kind: str, seam_why_it_matters: str) -> str:
    kind_takeaway = {
        "upstream_contract": "Use this as the baseline behavior/contract all other implementations are compared against.",
        "design_reference": "Use this to align implementation assumptions with documented intent.",
        "runtime_example": "Use this to verify expected runtime wiring in practical usage.",
        "current_state": "Use this to characterize present behavior and identify divergence from upstream seams.",
        "target_state": "Use this to scope what is already scaffolded vs still unimplemented in the next path.",
        "runtime_constraints": "Use this to identify device/runtime limits that can block or shape feasible integration.",
    }.get(evidence_kind, "")
    if seam_why_it_matters:
        return f"{kind_takeaway} {seam_why_it_matters}".strip()
    return kind_takeaway or "Use this as concrete supporting evidence for seam analysis."


def bucket_from_repo_and_path(repo_name: str, file_path: str) -> str:
    if repo_name == "vllm":
        return "upstream-vllm"
    if repo_name == "vllm-spyre":
        if file_path.startswith("vllm_spyre_next/"):
            return "vllm-spyre-next"
        return "vllm-spyre"
    if repo_name == "torch-spyre":
        return "torch-spyre"
    return "other"


def target_anchor_value(target: dict[str, Any]) -> str:
    if target.get("symbol"):
        return str(target["symbol"])
    if target.get("literal"):
        return str(target["literal"])
    if target.get("literal_any"):
        vals = [str(x) for x in target.get("literal_any", []) if str(x)]
        return " | ".join(vals[:3])
    if target.get("contains_all"):
        vals = [str(x) for x in target.get("contains_all", []) if str(x)]
        return " && ".join(vals[:4])
    if "start" in target and "end" in target:
        return f"{target.get('start')}:{target.get('end')}"
    return ""


def infer_story_kind(target: dict[str, Any], evidence_kind: str, file_path: str) -> str:
    explicit = str(target.get("story_kind", "")).strip().lower()
    if explicit:
        return explicit

    anchor_text = " ".join(
        [
            str(target.get("symbol", "")),
            str(target.get("literal", "")),
            " ".join(str(x) for x in target.get("contains_all", [])),
            file_path,
        ]
    ).lower()

    lifecycle_tokens = [
        "start_",
        "wait_",
        "save_",
        "load_",
        "schedule",
        "alloc",
        "commit",
        "new_step",
    ]
    if any(tok in anchor_text for tok in lifecycle_tokens):
        return "lifecycle"

    data_tokens = [
        "slot_mapping",
        "block_table",
        "connector_meta",
        "metadata",
        "kv_cache_shape",
        "kv_cache_stride",
        "stride",
        "dtype",
        "head_size",
        "block_size",
    ]
    if any(tok in anchor_text for tok in data_tokens):
        return "data_structure"

    callsite_tokens = [
        "get_",
        "update_",
        "build_",
        "register",
        "resolve_",
        "check_and_update",
        "run",
        "forward",
    ]
    if any(tok in anchor_text for tok in callsite_tokens):
        return "call_site"

    if evidence_kind == "runtime_example":
        return "example"
    if evidence_kind in {"current_state", "target_state", "runtime_constraints"}:
        return "divergence"
    return "contract"


def story_kind_label(kind: str) -> str:
    return {
        "contract": "Contract",
        "call_site": "Call site",
        "data_structure": "Data structure",
        "lifecycle": "Lifecycle",
        "divergence": "Divergence",
        "example": "Example",
    }.get(kind, "Evidence")


def default_story_proves(
    story_kind: str,
    evidence_kind: str,
    seam_question: str,
    anchor_value: str,
) -> str:
    by_kind = {
        "contract": "This snippet defines the interface contract that downstream implementations must honor.",
        "call_site": "This snippet shows where the seam is invoked in runtime control flow.",
        "data_structure": "This snippet exposes the concrete metadata/data layout that flows across seam boundaries.",
        "lifecycle": "This snippet establishes ordering semantics across lifecycle methods.",
        "divergence": "This snippet captures where current Spyre/runtime behavior diverges from upstream expectations.",
        "example": "This snippet gives a concrete reference path that can be mirrored in a minimal POC.",
    }
    base = by_kind.get(story_kind, "This snippet is direct evidence for the seam analysis.")
    if seam_question:
        base = f"{base} It directly informs: {seam_question}"
    if anchor_value:
        return f"{base} Anchor: `{anchor_value}`."
    return base


def default_story_boundary(story_kind: str, evidence_kind: str) -> str:
    if story_kind == "contract":
        return "Ownership boundary: upstream interface defines obligations; backend/plugin code is responsible for conformance."
    if story_kind == "call_site":
        return "Ownership boundary: caller controls timing/context; callee should consume provided metadata without re-deriving scheduler state."
    if story_kind == "data_structure":
        return "Ownership boundary: schema producers (scheduler/manager) and consumers (runner/backend/connector) must agree on exact layout semantics."
    if story_kind == "lifecycle":
        return "Ownership boundary: lifecycle ordering is enforced by orchestrator paths; implementation must preserve method sequencing."
    if story_kind == "divergence":
        if evidence_kind == "target_state":
            return "Ownership boundary: target-state scaffold indicates intent, but implementation ownership is still open."
        if evidence_kind == "runtime_constraints":
            return "Ownership boundary: runtime stack capabilities constrain what higher layers can safely assume."
        return "Ownership boundary: current stack uses compatibility behavior that may not match upstream ownership assumptions."
    return "Ownership boundary: treat this as supporting context and validate against contract anchors."


def default_story_implications(
    story_kind: str, seam_why_it_matters: str, evidence_kind: str
) -> list[str]:
    by_kind = {
        "contract": "Integration should start by conforming to this contract before optimization work.",
        "call_site": "Do not duplicate orchestration logic in plugins; align with existing call-site ordering.",
        "data_structure": "Lock the metadata/layout contract early, then optimize physical implementation.",
        "lifecycle": "Add tests that assert method ordering and state transitions across scheduling/execution.",
        "divergence": "Use this as migration scope definition; avoid carrying divergence into vllm-spyre-next by default.",
        "example": "Use this as implementation template for a minimal correctness-first path.",
    }
    out = [by_kind.get(story_kind, "Use this as actionable seam evidence.")]
    if seam_why_it_matters:
        out.append(seam_why_it_matters)
    if evidence_kind == "runtime_constraints":
        out.append("Hardware/runtime constraints may force phased rollout even when interface work is complete.")
    return out


def default_story_inference(evidence_kind: str) -> str:
    if evidence_kind == "design_reference":
        return "Inference: design docs capture intended behavior; validate with call-site/runtime snippets before final decisions."
    if evidence_kind == "target_state":
        return "Inference: scaffold code indicates intended integration direction but not full behavior."
    if evidence_kind == "runtime_constraints":
        return "Inference: runtime gaps are likely blockers for performance/production hardening, not for correctness-only seams."
    return ""


def default_story_sequence(layer: str) -> list[str]:
    layer_norm = (layer or "").strip().lower()
    if layer_norm == "disaggregation":
        return ["contract", "call_site", "data_structure", "lifecycle", "divergence", "example"]
    if layer_norm == "kv_memory":
        return ["contract", "data_structure", "call_site", "divergence", "example", "lifecycle"]
    if layer_norm == "platform":
        return ["contract", "call_site", "divergence", "example", "lifecycle", "data_structure"]
    if layer_norm == "compiler_runtime":
        return ["contract", "call_site", "divergence", "example", "data_structure", "lifecycle"]
    return ["contract", "call_site", "data_structure", "lifecycle", "divergence", "example"]


def infer_decision_target(layer: str) -> str:
    layer_norm = (layer or "").strip().lower()
    if layer_norm in {"kv_memory", "disaggregation"}:
        return "Interface decision"
    if layer_norm in {"platform", "compiler_runtime"}:
        return "Feasibility decision"
    return "Design choice decision"


def default_decision_answer(
    seam: dict[str, Any], coverage: dict[str, Any], selected_story_kinds: list[str]
) -> str:
    if seam.get("decision_answer"):
        return str(seam.get("decision_answer"))
    repos_present = set(coverage.get("repos_present", []))
    if "divergence" in selected_story_kinds and "vllm-spyre" in repos_present:
        return "Partially: correctness path is feasible now, but migration/divergence work is required to align with upstream seams."
    if "runtime_constraints" in [str(x) for x in seam.get("tags", [])]:
        return "Partially: contract-level work is feasible now; runtime maturity defines production readiness."
    return "Yes for correctness-first implementation, provided the seam contract shown here is followed exactly."


def default_claims(seam: dict[str, Any]) -> list[str]:
    if seam.get("claims"):
        return [str(x) for x in seam.get("claims", []) if str(x).strip()]
    claims: list[str] = []
    summary = str(seam.get("summary", "")).strip()
    why = str(seam.get("why_it_matters", "")).strip()
    if summary:
        claims.append(summary)
    if why:
        claims.append(why)
    if not claims and seam.get("question"):
        claims.append(f"This seam answers: {seam.get('question')}")
    return claims[:3]


def default_recommendation(seam: dict[str, Any]) -> dict[str, Any]:
    rec = seam.get("recommendation")
    if isinstance(rec, dict):
        return rec
    poc_refs = [str(x) for x in seam.get("poc_links", []) if str(x).strip()]
    follows = ", ".join(f"`{x}`" for x in poc_refs) if poc_refs else "seam-specific POC tasks"
    return {
        "now": "Implement/validate against the contract and call sites shown in story snippets before optimization.",
        "depends_on": "This depends on consistent metadata ownership across scheduler, runner, and backend/connector seams.",
        "success_test": f"Add a focused integration check tied to {follows}.",
    }


def build_story_selection(
    seam: dict[str, Any], seam_target_plan: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    candidates = [
        t
        for t in seam_target_plan
        if t.get("status") == "matched" and t.get("snippet_ids") and t.get("include_in_story", True)
    ]
    candidates.sort(key=lambda t: (int(t.get("story_priority", t.get("target_index", 9999))), int(t.get("target_index", 9999))))

    if not candidates:
        return [], []

    sequence = [str(x) for x in seam.get("story_sequence", []) if str(x).strip()]
    if not sequence:
        sequence = default_story_sequence(str(seam.get("layer", "")))

    budget_min = int(seam.get("story_budget_min", 3))
    budget_max = int(seam.get("story_budget_max", 5))
    if budget_max < budget_min:
        budget_max = budget_min

    selected_target_ids: list[str] = []
    used: set[str] = set()

    for desired_kind in sequence:
        for target in candidates:
            tid = str(target.get("target_id", ""))
            if tid in used:
                continue
            if str(target.get("story_kind", "")) == desired_kind:
                selected_target_ids.append(tid)
                used.add(tid)
                break
        if len(selected_target_ids) >= budget_max:
            break

    for target in candidates:
        if len(selected_target_ids) >= max(budget_min, 1):
            break
        tid = str(target.get("target_id", ""))
        if tid in used:
            continue
        selected_target_ids.append(tid)
        used.add(tid)

    for target in candidates:
        if len(selected_target_ids) >= budget_max:
            break
        tid = str(target.get("target_id", ""))
        if tid in used:
            continue
        selected_target_ids.append(tid)
        used.add(tid)

    story_snippet_ids: list[str] = []
    for tid in selected_target_ids:
        target = next((x for x in candidates if str(x.get("target_id")) == tid), None)
        if not target:
            continue
        story_snippet_ids.append(str(target["snippet_ids"][0]))

    return story_snippet_ids, selected_target_ids


def build_seam_completeness(
    seam: dict[str, Any],
    selected_story_kinds: list[str],
    coverage: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    repo_count = len(set(coverage.get("repos_present", [])))
    compare_ok = True
    if repo_count >= 2:
        compare_ok = bool(seam.get("repo_lens"))
    items = [
        {"id": "contract", "label": "Contract shown", "ok": "contract" in selected_story_kinds},
        {"id": "call_site", "label": "Call site shown", "ok": "call_site" in selected_story_kinds},
        {
            "id": "data_structure",
            "label": "Data structure shown",
            "ok": "data_structure" in selected_story_kinds,
        },
        {"id": "lifecycle", "label": "Lifecycle described", "ok": "lifecycle" in selected_story_kinds},
        {
            "id": "ownership",
            "label": "Ownership described",
            "ok": bool(seam.get("repo_lens")),
        },
        {
            "id": "failure_mode",
            "label": "Failure mode noted",
            "ok": bool(seam.get("common_pitfalls")),
        },
        {
            "id": "test_suggestion",
            "label": "Test suggestion included",
            "ok": bool(recommendation.get("success_test")) or bool(seam.get("poc_links")),
        },
        {
            "id": "compare",
            "label": "Compare included where meaningful",
            "ok": compare_ok,
        },
    ]
    passed = sum(1 for x in items if x["ok"])
    return {
        "items": items,
        "passed": passed,
        "total": len(items),
        "missing_labels": [x["label"] for x in items if not x["ok"]],
    }


def build_compare_rows(
    seam: dict[str, Any],
    story_snippet_ids: list[str],
    snippet_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    repo_lens = seam.get("repo_lens", {}) or {}
    bucket_order = ["upstream-vllm", "vllm-spyre", "vllm-spyre-next", "torch-spyre"]

    rows: list[dict[str, Any]] = []
    for bucket in bucket_order:
        snippet_id = ""
        for sid in story_snippet_ids:
            sn = snippet_lookup.get(sid)
            if not sn:
                continue
            if bucket_from_repo_and_path(sn["repo"], sn["file_path"]) == bucket:
                snippet_id = sid
                break
        rows.append(
            {
                "bucket": bucket,
                "note": str(repo_lens.get(bucket, "")).strip(),
                "story_snippet_id": snippet_id,
            }
        )
    return rows


def resolve_repo_infos(config: dict[str, Any], config_path: Path) -> dict[str, RepoInfo]:
    repo_roots = config["repo_roots"]
    infos: dict[str, RepoInfo] = {}

    for name, root_raw in repo_roots.items():
        root = (config_path.parent / root_raw).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Repo root for {name} does not exist: {root}")

        head_sha = run(["git", "rev-parse", "HEAD"], root)
        head_short = head_sha[:10]
        head_ref = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)
        status = run_result(["git", "status", "--porcelain"], root)
        is_dirty = bool((status.stdout or "").strip())
        head_in_origin_refs = check_head_in_origin_refs(root, head_sha)
        origin = run(["git", "remote", "get-url", "origin"], root)
        github_base = normalize_github_origin(origin).github_base

        infos[name] = RepoInfo(
            name=name,
            root=root,
            head_sha=head_sha,
            head_short=head_short,
            head_ref=head_ref,
            is_dirty=is_dirty,
            head_in_origin_refs=head_in_origin_refs,
            origin=origin,
            github_base=github_base,
        )

    return infos


def collect_target_files(seams: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for seam in seams:
        for target in seam.get("targets", []):
            repo = target["repo"]
            out.setdefault(repo, set()).add(target["file"])
    return out


def load_file_snapshots(
    repo_infos: dict[str, RepoInfo], target_files: dict[str, set[str]]
) -> dict[tuple[str, str], dict[str, Any]]:
    snapshots: dict[tuple[str, str], dict[str, Any]] = {}
    for repo_name, files in target_files.items():
        repo_info = repo_infos[repo_name]
        for rel_path in sorted(files):
            lines = read_lines_from_git_blob(repo_info.root, repo_info.head_sha, rel_path)
            if lines is None:
                continue
            snapshots[(repo_name, rel_path)] = {
                "repo": repo_name,
                "file_path": rel_path,
                "lines": lines,
                "source_mode": "head_commit",
                "source_ref": repo_info.head_sha,
            }
    return snapshots


def collect_symbols(file_snapshots: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []

    for (repo_name, rel_path), snap in sorted(file_snapshots.items()):
        if not rel_path.endswith(".py"):
            continue

        try:
            src = "\n".join(snap["lines"])
            tree = ast.parse(src)
        except Exception:
            continue

        module = rel_path[:-3].replace("/", ".")
        collector = SymbolCollector(module=module)
        collector.visit(tree)

        for item in collector.items:
            symbols.append(
                {
                    "repo": repo_name,
                    "file_path": rel_path,
                    **item,
                }
            )

    return symbols


def collect_grep_hits(
    pattern_tags: list[str],
    file_snapshots: dict[tuple[str, str], dict[str, Any]],
    grep_context: int,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []

    for pattern in pattern_tags:
        rgx = re.compile(pattern)
        for (repo_name, rel_path), snap in sorted(file_snapshots.items()):
            lines = snap["lines"]
            for i, line in enumerate(lines, start=1):
                if rgx.search(line):
                    start = max(1, i - grep_context)
                    end = min(len(lines), i + grep_context)
                    excerpt = "\n".join(lines[start - 1 : end])
                    hits.append(
                        {
                            "pattern": pattern,
                            "repo": repo_name,
                            "file_path": rel_path,
                            "line": i,
                            "start_line": start,
                            "end_line": end,
                            "excerpt": excerpt,
                        }
                    )
                    if len(hits) > 20000:
                        return hits

    return hits


def build_snippets(
    seams: list[dict[str, Any]],
    repo_infos: dict[str, RepoInfo],
    file_snapshots: dict[tuple[str, str], dict[str, Any]],
    symbols: list[dict[str, Any]],
    default_context: int,
    allow_regex_targets: bool,
    strict_target_matches: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    symbols_by_file: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in symbols:
        symbols_by_file.setdefault((s["repo"], s["file_path"]), []).append(s)

    snippets: list[dict[str, Any]] = []
    snippet_lookup: dict[str, dict[str, Any]] = {}
    seams_out: list[dict[str, Any]] = []

    # File cache used by UI for "full file" and context toggles.
    files_map: dict[str, dict[str, Any]] = {}

    for seam in seams:
        seam_id = seam["id"]
        seam_snippet_ids: list[str] = []
        seam_extractors: list[str] = []
        seam_quality_scores: list[float] = []
        seam_target_plan: list[dict[str, Any]] = []
        seam_buckets_seen: set[str] = set()

        for target_idx, target in enumerate(seam.get("targets", []), start=1):
            validate_target_schema(seam_id, target)
            repo_name = target["repo"]
            file_path = target["file"]
            target_id = str(target.get("target_id", f"{seam_id}__t{target_idx:02d}"))
            evidence_kind = str(target.get("evidence_kind", infer_evidence_kind(repo_name, file_path)))
            anchor_type = infer_anchor_type(target)
            anchor_value = target_anchor_value(target)
            story_kind = infer_story_kind(target, evidence_kind, file_path)
            story_priority = int(target.get("story_priority", target_idx))
            include_in_story = bool(target.get("include_in_story", True))
            target_required = bool(target.get("required", True))
            repo_info = repo_infos[repo_name]
            snapshot = file_snapshots.get((repo_name, file_path))
            target_snippet_ids: list[str] = []
            if snapshot is None:
                seam_target_plan.append(
                    {
                        "target_id": target_id,
                        "target_index": target_idx,
                        "repo": repo_name,
                        "file": file_path,
                        "anchor_type": anchor_type,
                        "anchor_value": anchor_value,
                        "evidence_kind": evidence_kind,
                        "story_kind": story_kind,
                        "story_label": story_kind_label(story_kind),
                        "story_priority": story_priority,
                        "include_in_story": include_in_story,
                        "required": target_required,
                        "status": "missing_file",
                        "match_count": 0,
                        "snippet_ids": [],
                    }
                )
                if target_required and strict_target_matches:
                    raise ValueError(
                        f"[{seam_id}] target file not found in pinned commit blob: "
                        f"repo={repo_name} file={file_path}"
                    )
                continue

            lines = snapshot["lines"]
            n_lines = len(lines)
            try:
                ranges = extract_target_ranges(
                    target=target,
                    lines=lines,
                    symbols_by_file=symbols_by_file,
                    default_context=default_context,
                    allow_regex_targets=allow_regex_targets,
                )
            except Exception as exc:
                raise ValueError(
                    f"[{seam_id}] failed to extract target repo={repo_name} file={file_path}: {exc}"
                ) from exc

            if not ranges and target_required and strict_target_matches:
                raise ValueError(
                    f"[{seam_id}] no matches for required target repo={repo_name} "
                    f"file={file_path} target={target}"
                )

            seam_target_plan.append(
                {
                    "target_id": target_id,
                    "target_index": target_idx,
                    "repo": repo_name,
                    "file": file_path,
                    "anchor_type": anchor_type,
                    "anchor_value": anchor_value,
                    "evidence_kind": evidence_kind,
                    "story_kind": story_kind,
                    "story_label": story_kind_label(story_kind),
                    "story_priority": story_priority,
                    "include_in_story": include_in_story,
                    "required": target_required,
                    "status": "matched" if ranges else "no_match",
                    "match_count": len(ranges),
                    "snippet_ids": target_snippet_ids,
                }
            )

            for hit_idx, r in enumerate(ranges, start=1):
                start = int(r["start"])
                end = int(r["end"])
                code_lines = slice_lines(lines, start, end)
                context_start, context_end = compute_context_bounds(
                    start, end, n_lines, default_context
                )
                context_lines = slice_lines(lines, context_start, context_end)

                sid = (
                    f"{seam_id}__{repo_name}__"
                    f"{file_path.replace('/', '__').replace('.', '_')}__"
                    f"{start}_{end}_{target_idx}_{hit_idx}"
                )

                permalink_sha = build_permalink(
                    repo_info.github_base, repo_info.head_sha, file_path, start, end
                )
                permalink_branch = ""
                if repo_info.head_ref and repo_info.head_ref != "HEAD":
                    permalink_branch = build_permalink(
                        repo_info.github_base, repo_info.head_ref, file_path, start, end
                    )

                permalink_mode = "sha"
                permalink = permalink_sha
                if repo_info.head_in_origin_refs is False:
                    permalink_mode = "sha_not_in_origin_refs"

                selection_reason = build_selection_reason(
                    extractor=str(r.get("extractor", "")),
                    anchor=str(r.get("anchor", "")),
                    matched=str(r.get("matched", "")),
                )
                strength = evidence_strength(str(r.get("extractor", "")))

                target_role = str(target.get("role", "")).strip()
                if not target_role:
                    target_role = default_target_role(repo_name, file_path, seam_id)

                target_note = str(target.get("note", "")).strip()
                if not target_note:
                    seam_question = str(seam.get("question", "")).strip()
                    anchor_value = str(
                        target.get("symbol")
                        or target.get("literal")
                        or " && ".join(target.get("contains_all", []))
                        or ""
                    )
                    target_note = default_target_note(
                        evidence_kind=evidence_kind,
                        seam_question=seam_question,
                        file_path=file_path,
                        anchor_type=anchor_type,
                        anchor_value=anchor_value,
                    )

                target_takeaway = str(target.get("takeaway", "")).strip()
                if not target_takeaway:
                    target_takeaway = default_target_takeaway(
                        evidence_kind=evidence_kind,
                        seam_why_it_matters=str(seam.get("why_it_matters", "")).strip(),
                    )

                target_compare = str(target.get("compare", "")).strip()
                if not target_compare:
                    target_compare = ""

                target_checklist = target.get("checklist", [])
                if not target_checklist:
                    target_checklist = list(seam.get("reading_checklist", []))[:3]

                story_proves = str(target.get("story_proves", "")).strip()
                if not story_proves:
                    story_proves = default_story_proves(
                        story_kind=story_kind,
                        evidence_kind=evidence_kind,
                        seam_question=str(seam.get("question", "")).strip(),
                        anchor_value=anchor_value,
                    )

                story_boundary = str(target.get("story_boundary", "")).strip()
                if not story_boundary:
                    story_boundary = default_story_boundary(
                        story_kind=story_kind,
                        evidence_kind=evidence_kind,
                    )

                story_implications = target.get("story_implications", [])
                if not story_implications:
                    story_implications = default_story_implications(
                        story_kind=story_kind,
                        seam_why_it_matters=str(seam.get("why_it_matters", "")).strip(),
                        evidence_kind=evidence_kind,
                    )

                story_risks = target.get("story_risks", [])
                if not story_risks:
                    story_risks = list(seam.get("common_pitfalls", []))[:2]

                story_inference = str(target.get("story_inference", "")).strip()
                if not story_inference:
                    story_inference = default_story_inference(evidence_kind)

                snippet = {
                    "id": sid,
                    "seam_id": seam_id,
                    "seam_title": seam["title"],
                    "repo": repo_name,
                    "file_path": file_path,
                    "commit": repo_info.head_sha,
                    "commit_short": repo_info.head_short,
                    "head_ref": repo_info.head_ref,
                    "repo_dirty": repo_info.is_dirty,
                    "head_in_origin_refs": repo_info.head_in_origin_refs,
                    "origin": repo_info.origin,
                    "github_base": repo_info.github_base,
                    "permalink_sha": permalink_sha,
                    "permalink_branch": permalink_branch,
                    "permalink_mode": permalink_mode,
                    "permalink": permalink,
                    "start_line": start,
                    "end_line": end,
                    "language": detect_language(file_path),
                    "extractor": r["extractor"],
                    "anchor": r["anchor"],
                    "matched": r["matched"],
                    "selection_reason": selection_reason,
                    "target_index": target_idx,
                    "target_id": target_id,
                    "anchor_type": anchor_type,
                    "anchor_value": anchor_value,
                    "evidence_kind": evidence_kind,
                    "target_role": target_role,
                    "target_note": target_note,
                    "target_takeaway": target_takeaway,
                    "target_compare": target_compare,
                    "target_checklist": target_checklist,
                    "story_kind": story_kind,
                    "story_label": story_kind_label(story_kind),
                    "story_priority": story_priority,
                    "story_proves": story_proves,
                    "story_boundary": story_boundary,
                    "story_implications": story_implications,
                    "story_risks": story_risks,
                    "story_inference": story_inference,
                    "evidence_tier": strength["tier"],
                    "evidence_score": strength["score"],
                    "source_mode": snapshot["source_mode"],
                    "source_ref": snapshot["source_ref"],
                    "tags": sorted(set([*seam.get("tags", []), *target.get("tags", [])])),
                    "code": "\n".join(code_lines),
                    "lines": with_line_numbers(code_lines, start),
                    "context_start_line": context_start,
                    "context_end_line": context_end,
                    "context_code": "\n".join(context_lines),
                    "context_lines": with_line_numbers(context_lines, context_start),
                }
                snippets.append(snippet)
                snippet_lookup[sid] = snippet
                seam_snippet_ids.append(sid)
                target_snippet_ids.append(sid)
                seam_buckets_seen.add(repo_name)
                seam_extractors.append(str(r.get("extractor", "")))
                seam_quality_scores.append(float(strength["score"]))

                file_key = f"{repo_name}::{file_path}"
                if file_key not in files_map:
                    files_map[file_key] = {
                        "file_key": file_key,
                        "repo": repo_name,
                        "file_path": file_path,
                        "language": detect_language(file_path),
                        "source_mode": snapshot["source_mode"],
                        "source_ref": snapshot["source_ref"],
                        "total_lines": n_lines,
                        "content": "\n".join(lines),
                        "lines": with_line_numbers(lines, 1),
                    }

        seam_out = dict(seam)
        seam_out["snippet_ids"] = seam_snippet_ids
        extractor_counts: dict[str, int] = {}
        for ex in seam_extractors:
            extractor_counts[ex] = extractor_counts.get(ex, 0) + 1
        seam_out["evidence_stats"] = {
            "extractors": extractor_counts,
            "mean_score": (sum(seam_quality_scores) / len(seam_quality_scores))
            if seam_quality_scores
            else 0.0,
        }
        seam_out["target_plan"] = seam_target_plan
        seam_out["coverage"] = {
            "repos_present": sorted(seam_buckets_seen),
            "has_upstream_vllm": "vllm" in seam_buckets_seen,
            "has_vllm_spyre": "vllm-spyre" in seam_buckets_seen,
            "has_torch_spyre": "torch-spyre" in seam_buckets_seen,
        }

        story_snippet_ids, selected_target_ids = build_story_selection(seam, seam_target_plan)
        selected_kinds: list[str] = []
        for tid in selected_target_ids:
            plan = next((x for x in seam_target_plan if str(x.get("target_id")) == tid), None)
            if plan:
                selected_kinds.append(str(plan.get("story_kind", "")))

        story_target_plan = [
            x for x in seam_target_plan if str(x.get("target_id", "")) in set(selected_target_ids)
        ]
        story_target_plan.sort(
            key=lambda t: (
                int(t.get("story_priority", t.get("target_index", 9999))),
                int(t.get("target_index", 9999)),
            )
        )

        story_set = set(story_snippet_ids)
        evidence_drawer_snippet_ids = [sid for sid in seam_snippet_ids if sid not in story_set]

        seam_out["story_snippet_ids"] = story_snippet_ids
        seam_out["evidence_drawer_snippet_ids"] = evidence_drawer_snippet_ids
        seam_out["story_target_ids"] = selected_target_ids
        seam_out["story_target_plan"] = story_target_plan
        seam_out["story_kinds"] = selected_kinds
        seam_out["story_budget"] = {
            "min": int(seam.get("story_budget_min", 3)),
            "max": int(seam.get("story_budget_max", 5)),
            "selected": len(story_snippet_ids),
        }

        decision_target = str(seam.get("decision_target", "")).strip()
        if not decision_target:
            decision_target = infer_decision_target(str(seam.get("layer", "")))

        decision_question = str(seam.get("decision_question", "")).strip()
        if not decision_question:
            decision_question = str(seam.get("question", "")).strip()

        decision_answer = default_decision_answer(
            seam=seam,
            coverage=seam_out["coverage"],
            selected_story_kinds=selected_kinds,
        )
        claims = default_claims(seam)
        recommendation = default_recommendation(seam)
        completeness = build_seam_completeness(
            seam=seam,
            selected_story_kinds=selected_kinds,
            coverage=seam_out["coverage"],
            recommendation=recommendation,
        )

        seam_out["decision_target"] = decision_target
        seam_out["decision_question"] = decision_question
        seam_out["decision_answer"] = decision_answer
        seam_out["claims"] = claims
        seam_out["recommendation"] = recommendation
        seam_out["completeness"] = completeness
        seam_out["compare_rows"] = build_compare_rows(
            seam=seam,
            story_snippet_ids=story_snippet_ids,
            snippet_lookup=snippet_lookup,
        )
        seams_out.append(seam_out)

    files_list = sorted(files_map.values(), key=lambda f: (f["repo"], f["file_path"]))
    return snippets, files_map, seams_out


def build_report_markdown(
    seams: list[dict[str, Any]], snippet_count: int, story_data: dict[str, Any] | None
) -> str:
    story_data = story_data or {}

    lines: list[str] = []
    lines.append("# vLLM-Spyre Code Atlas Report")
    lines.append("")
    lines.append(
        story_data.get(
            "atlas_purpose",
            "Local-first architecture-to-code atlas that ties claims to pinned snippets and permalinks.",
        )
    )
    lines.append("")
    lines.append(f"Generated snippets: **{snippet_count}**")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("- Feasibility: high for correctness-first POCs on `vllm-spyre-next` seams.")
    lines.append("- Primary risk: KV layout/metadata mismatches at attention + connector boundaries.")
    lines.append("- Validation strategy: prove contracts with pinned snippets before optimizing kernels.")
    lines.append("")
    quality = build_quality_report(seams)
    qagg = quality["aggregate"]
    lines.append("## Evidence Quality Snapshot")
    lines.append("")
    lines.append(f"- Seams: `{qagg['seam_count']}`")
    lines.append(f"- Mean evidence score: `{qagg['mean_of_means']:.2f}`")
    lines.append(
        f"- Seams with full required-target matches: `{qagg['seams_with_full_required_matches']}/{qagg['seam_count']}`"
    )
    lines.append(
        f"- Seams with upstream anchors: `{qagg['seams_with_upstream_anchor']}/{qagg['seam_count']}`"
    )
    lines.append(
        f"- Seams with 3-5 story snippets: `{qagg['seams_with_story_budget_3_to_5']}/{qagg['seam_count']}`"
    )
    lines.append(f"- Mean seam completeness: `{qagg['mean_completeness']:.2f}`")
    lines.append(f"- Legacy anchor count: `{qagg['total_legacy_anchor_count']}`")
    if quality["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for w in quality["warnings"][:12]:
            lines.append(f"- {w}")
    lines.append("")
    lines.append("## Evidence Method")
    lines.append("")
    lines.append("- Anchor types: `symbol`, `literal`, `contains_all`, `range`.")
    lines.append("- Regex anchors are disabled by default and treated as legacy when explicitly enabled.")
    lines.append("- Required targets are strict: generation fails when a required anchor is missing.")
    lines.append("- Snippets are extracted from `git show <HEAD>:<path>` blobs and linked by pinned SHA.")
    lines.append("- Each seam is editorialized into 3-5 story snippets (when available) plus an evidence drawer.")
    lines.append("")
    lines.append("## How To Read This Atlas")
    lines.append("")
    for step in story_data.get("how_to_read", []):
        lines.append(f"### {step.get('title', 'Step')}")
        lines.append(step.get("details", ""))
        seam_refs = [f"`{sid}`" for sid in step.get("seams", [])]
        if seam_refs:
            lines.append("")
            lines.append(f"Seams: {', '.join(seam_refs)}")
        lines.append("")

    lines.append("## Architecture Layers")
    lines.append("")
    for layer in story_data.get("architecture_layers", []):
        lines.append(f"### {layer.get('title', 'Layer')}")
        lines.append(layer.get("summary", ""))
        verify = layer.get("what_to_verify", [])
        if verify:
            lines.append("")
            lines.append("Verification checklist:")
            for item in verify:
                lines.append(f"- {item}")
        seam_refs = [f"`{sid}`" for sid in layer.get("seams", [])]
        if seam_refs:
            lines.append("")
            lines.append(f"Seams: {', '.join(seam_refs)}")
        lines.append("")

    lines.append("## Seam Details")
    lines.append("")
    for seam in seams:
        lines.append(f"### {seam['title']}")
        lines.append(f"`{seam['id']}`")
        lines.append("")
        if seam.get("question"):
            lines.append(f"Question: {seam['question']}")
            lines.append("")
        if seam.get("decision_target"):
            lines.append(f"Decision target: {seam['decision_target']}")
            lines.append("")
        if seam.get("decision_answer"):
            lines.append(f"Decision answer: {seam['decision_answer']}")
            lines.append("")
        claims = seam.get("claims", [])
        if claims:
            lines.append("Claims:")
            for claim in claims:
                lines.append(f"- {claim}")
            lines.append("")
        if seam.get("summary"):
            lines.append(f"Summary: {seam['summary']}")
            lines.append("")
        if seam.get("why_it_matters"):
            lines.append(f"Why it matters: {seam['why_it_matters']}")
            lines.append("")
        lines.append(f"Snippet count: {len(seam.get('snippet_ids', []))}")
        lines.append(f"Story snippets: {len(seam.get('story_snippet_ids', []))}")
        lines.append(f"Evidence drawer snippets: {len(seam.get('evidence_drawer_snippet_ids', []))}")
        lines.append("")

        story_plan = seam.get("story_target_plan", [])
        if story_plan:
            lines.append("Story path:")
            for item in story_plan:
                lines.append(
                    f"- [{item.get('story_label', 'Evidence')}] `{item.get('target_id', '')}` "
                    f"({item.get('repo', '')}:{item.get('file', '')})"
                )
            lines.append("")

        completeness = seam.get("completeness", {})
        if completeness:
            lines.append(
                f"Completeness: {completeness.get('passed', 0)}/{completeness.get('total', 0)}"
            )
            for item in completeness.get("items", []):
                marker = "x" if item.get("ok") else " "
                lines.append(f"- [{marker}] {item.get('label', '')}")
            lines.append("")

        checklist = seam.get("reading_checklist", [])
        if checklist:
            lines.append("Reading checklist:")
            for item in checklist:
                lines.append(f"- {item}")
            lines.append("")

        lens = seam.get("repo_lens", {})
        if lens:
            lines.append("Repo lens:")
            for key in ["upstream-vllm", "vllm-spyre", "vllm-spyre-next", "torch-spyre"]:
                if key in lens:
                    lines.append(f"- {key}: {lens[key]}")
            lines.append("")

        pitfalls = seam.get("common_pitfalls", [])
        if pitfalls:
            lines.append("Common pitfalls:")
            for item in pitfalls:
                lines.append(f"- {item}")
            lines.append("")

        related = seam.get("related_seams", [])
        if related:
            refs = ", ".join(f"`{x}`" for x in related)
            lines.append(f"Related seams: {refs}")
            lines.append("")

        rec = seam.get("recommendation", {})
        if rec:
            lines.append("Recommendation:")
            if rec.get("now"):
                lines.append(f"- Now: {rec.get('now')}")
            if rec.get("depends_on"):
                lines.append(f"- Depends on: {rec.get('depends_on')}")
            if rec.get("success_test"):
                lines.append(f"- Success test: {rec.get('success_test')}")
            lines.append("")

    lines.append("## POC References")
    lines.append("")
    for poc in story_data.get("poc_plan", []):
        lines.append(f"### {poc.get('title', poc.get('id', 'POC'))}")
        for step in poc.get("steps", []):
            lines.append(f"- {step.get('text', '')}")
            seam_refs = [f"`{sid}`" for sid in step.get("seams", [])]
            if seam_refs:
                lines.append(f"Seams: {', '.join(seam_refs)}")
        lines.append("")

    lines.append("## Global Caveats")
    lines.append("")
    for item in story_data.get("global_caveats", []):
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def ensure_output_dirs(output_dir: Path, web_data_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    web_data_dir.mkdir(parents=True, exist_ok=True)


def build_quality_report(seams: list[dict[str, Any]]) -> dict[str, Any]:
    seam_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for seam in seams:
        seam_id = seam.get("id", "")
        target_plan = seam.get("target_plan", [])
        required_total = 0
        required_matched = 0
        legacy_anchors = 0
        for t in target_plan:
            if t.get("required", True):
                required_total += 1
                if t.get("status") == "matched":
                    required_matched += 1
            if t.get("anchor_type") in {"regex", "grep"}:
                legacy_anchors += 1

        has_upstream = bool(seam.get("coverage", {}).get("has_upstream_vllm", False))
        mean_score = float(seam.get("evidence_stats", {}).get("mean_score", 0.0))
        story_count = len(seam.get("story_snippet_ids", []))
        matched_targets = sum(1 for t in target_plan if t.get("status") == "matched")
        min_story = 3 if matched_targets >= 3 else matched_targets
        story_budget_ok = min_story <= story_count <= 5 if matched_targets else story_count == 0
        completeness = seam.get("completeness", {}) or {}
        completeness_rate = 0.0
        if completeness.get("total"):
            completeness_rate = float(completeness.get("passed", 0)) / float(
                completeness.get("total", 1)
            )
        row = {
            "seam_id": seam_id,
            "title": seam.get("title", seam_id),
            "required_targets": required_total,
            "required_matched": required_matched,
            "required_match_rate": (required_matched / required_total) if required_total else 1.0,
            "legacy_anchor_count": legacy_anchors,
            "has_upstream_vllm": has_upstream,
            "mean_evidence_score": mean_score,
            "story_snippet_count": story_count,
            "matched_targets": matched_targets,
            "story_budget_ok": story_budget_ok,
            "completeness_rate": completeness_rate,
        }
        seam_rows.append(row)

        if required_total and required_matched < required_total:
            warnings.append(
                f"[{seam_id}] required targets matched {required_matched}/{required_total}."
            )
        if legacy_anchors > 0:
            warnings.append(
                f"[{seam_id}] uses legacy regex/grep anchors ({legacy_anchors})."
            )
        if not has_upstream:
            warnings.append(
                f"[{seam_id}] has no upstream vLLM anchor; comparison depth may be limited."
            )
        if story_count < 3 and len(target_plan) >= 3:
            warnings.append(
                f"[{seam_id}] story snippet count is {story_count}; target editorial budget is 3-5."
            )

    aggregate = {
        "seam_count": len(seam_rows),
        "mean_of_means": (sum(r["mean_evidence_score"] for r in seam_rows) / len(seam_rows))
        if seam_rows
        else 0.0,
        "mean_completeness": (sum(r["completeness_rate"] for r in seam_rows) / len(seam_rows))
        if seam_rows
        else 0.0,
        "seams_with_full_required_matches": sum(
            1 for r in seam_rows if r["required_match_rate"] == 1.0
        ),
        "seams_with_upstream_anchor": sum(1 for r in seam_rows if r["has_upstream_vllm"]),
        "seams_with_story_budget_3_to_5": sum(
            1 for r in seam_rows if bool(r.get("story_budget_ok", False))
        ),
        "total_legacy_anchor_count": sum(r["legacy_anchor_count"] for r in seam_rows),
        "warning_count": len(warnings),
    }
    return {"aggregate": aggregate, "seams": seam_rows, "warnings": warnings}


def write_outputs(
    output_dir: Path,
    web_data_dir: Path,
    commits: dict[str, Any],
    seams: list[dict[str, Any]],
    snippets: list[dict[str, Any]],
    symbols: list[dict[str, Any]],
    grep_hits: list[dict[str, Any]],
    files_map: dict[str, dict[str, Any]],
    report_md: str,
    story_data: dict[str, Any] | None,
    extraction_policy: dict[str, Any],
) -> None:
    story_data = story_data or {}
    quality_report = build_quality_report(seams)

    first_principles = story_data.get("first_principles", [
        {
            "id": "prefill_decode",
            "title": "Prefill vs decode",
            "summary": "Unified scheduling model with token catch-up instead of strict phase split.",
            "drill_seams": ["first_principles_prefill_decode", "scheduler_connector_hooks"],
        },
        {
            "id": "paged_kv",
            "title": "Paged KV and block mapping",
            "summary": "Block tables and slot mapping drive physical KV access and updates.",
            "drill_seams": ["attention_backend_contract", "block_table_slot_mapping"],
        },
        {
            "id": "disaggregation",
            "title": "KV transfer/disaggregated prefill-decode",
            "summary": "Connector lifecycle bridges scheduler decisions and runtime load/save operations.",
            "drill_seams": ["kv_connector_base_v1", "decode_bench_reference", "scheduler_connector_hooks"],
        },
        {
            "id": "plugin_stack",
            "title": "Plugin integration seams",
            "summary": "Platform, worker, model runner, attention backend, connector, and compile stack.",
            "drill_seams": [
                "platform_plugin_registration",
                "platform_resolution_interface",
                "spyre_platform_wiring",
                "spyre_next_platform_scaffold",
                "torch_spyre_inductor_hooks",
            ],
        },
    ])

    poc_plan = story_data.get("poc_plan", [
        {
            "id": "poc1",
            "title": "POC 1: correctness-first AttentionBackend",
            "steps": [
                {
                    "text": "Implement gather -> reference attention -> scatter path.",
                    "seams": ["attention_backend_contract", "block_table_slot_mapping", "spyre_next_platform_scaffold"],
                },
                {
                    "text": "Compare logits against upstream CPU path on short decode chains.",
                    "seams": ["attention_backend_contract", "first_principles_prefill_decode"],
                },
            ],
        },
        {
            "id": "poc2",
            "title": "POC 2: minimal KVConnector",
            "steps": [
                {
                    "text": "Implement scheduler and worker lifecycle methods in a toy connector.",
                    "seams": ["kv_connector_base_v1", "decode_bench_reference", "scheduler_connector_hooks"],
                },
                {
                    "text": "Validate with disaggregated prefill harness.",
                    "seams": ["first_principles_prefill_decode", "kv_connector_base_v1"],
                },
            ],
        },
        {
            "id": "poc3",
            "title": "POC 3: layer-wise split execution",
            "steps": [
                {
                    "text": "Return host-visible KV outputs from prefill graph and consume on decode side.",
                    "seams": ["spyre_model_runner_kv_spec", "spyre_next_stubs", "torch_spyre_runtime_gaps"],
                }
            ],
        },
    ])

    index_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seam_count": len(seams),
        "snippet_count": len(snippets),
        "seams": seams,
        "snippet_ids": [s["id"] for s in snippets],
        "has_files_index": True,
        "first_principles": first_principles,
        "poc_plan": poc_plan,
        "atlas_purpose": story_data.get("atlas_purpose", ""),
        "how_to_read": story_data.get("how_to_read", []),
        "architecture_layers": story_data.get("architecture_layers", []),
        "badge_legend": story_data.get("badge_legend", []),
        "global_caveats": story_data.get("global_caveats", []),
        "glossary": story_data.get("glossary", []),
        "story_kind_legend": [
            {"kind": "contract", "label": "Contract", "description": "Interface/shape/lifecycle obligations."},
            {"kind": "call_site", "label": "Call site", "description": "Where runtime invokes the seam."},
            {"kind": "data_structure", "label": "Data structure", "description": "Metadata and layout semantics."},
            {"kind": "lifecycle", "label": "Lifecycle", "description": "Ordering/timing across steps."},
            {"kind": "divergence", "label": "Divergence", "description": "Where current/target paths differ from upstream."},
            {"kind": "example", "label": "Example", "description": "Concrete reference implementation path."},
        ],
        "extraction_policy": extraction_policy,
        "quality_report": quality_report,
    }

    files = {
        "commits.json": commits,
        "snippets.json": snippets,
        "files.json": sorted(files_map.values(), key=lambda f: (f["repo"], f["file_path"])),
        "symbols.json": symbols,
        "grep_hits.json": grep_hits,
        "index.json": index_payload,
    }

    for name, payload in files.items():
        p = output_dir / name
        dump_json(p, payload)
        dump_json(web_data_dir / name, payload)

    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    (web_data_dir / "report.md").write_text(report_md, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate code-atlas data artifacts")
    parser.add_argument("--config", default=os.environ.get("CODE_ATLAS_CONFIG", "config.json"))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_json(config_path)

    output_dir = (config_path.parent / config.get("output_dir", "data")).resolve()
    web_data_dir = (config_path.parent / config.get("web_data_dir", "web/data")).resolve()
    ensure_output_dirs(output_dir, web_data_dir)

    repo_infos = resolve_repo_infos(config, config_path)
    seam_spec_path = (config_path.parent / config["seam_spec"]).resolve()
    seams = load_json(seam_spec_path)
    seam_narratives_path = config.get("seam_narratives")
    seam_narratives = load_optional_json(
        (config_path.parent / seam_narratives_path).resolve() if seam_narratives_path else None
    )
    seams = merge_seam_narratives(seams, seam_narratives)

    story_path = config.get("story_spec")
    story_data = load_optional_json(
        (config_path.parent / story_path).resolve() if story_path else None
    )

    target_files = collect_target_files(seams)
    file_snapshots = load_file_snapshots(repo_infos, target_files)
    symbols = collect_symbols(file_snapshots)
    snippets, files_map, seams_out = build_snippets(
        seams=seams,
        repo_infos=repo_infos,
        file_snapshots=file_snapshots,
        symbols=symbols,
        default_context=int(config.get("default_context", 50)),
        allow_regex_targets=bool(config.get("allow_regex_targets", False)),
        strict_target_matches=bool(config.get("strict_target_matches", True)),
    )

    grep_hits = collect_grep_hits(
        pattern_tags=list(config.get("pattern_tags", [])),
        file_snapshots=file_snapshots,
        grep_context=int(config.get("grep_context", 4)),
    )

    commits = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repos": {
            name: {
                "root": str(info.root),
                "head_sha": info.head_sha,
                "head_short": info.head_short,
                "head_ref": info.head_ref,
                "is_dirty": info.is_dirty,
                "head_in_origin_refs": info.head_in_origin_refs,
                "origin": info.origin,
                "github_base": info.github_base,
            }
            for name, info in sorted(repo_infos.items())
        },
    }

    report_md = build_report_markdown(seams_out, len(snippets), story_data=story_data)

    write_outputs(
        output_dir=output_dir,
        web_data_dir=web_data_dir,
        commits=commits,
        seams=seams_out,
        snippets=snippets,
        symbols=symbols,
        grep_hits=grep_hits,
        files_map=files_map,
        report_md=report_md,
        story_data=story_data,
        extraction_policy={
            "allow_regex_targets": bool(config.get("allow_regex_targets", False)),
            "strict_target_matches": bool(config.get("strict_target_matches", True)),
            "anchor_types": [
                "symbol",
                "literal",
                "literal_any",
                "contains_all",
                "range",
            ],
        },
    )

    min_snippets = int(config.get("min_snippets", 30))
    if len(snippets) < min_snippets:
        print(
            f"WARNING: generated snippet count {len(snippets)} is below required minimum {min_snippets}"
        )

    print(f"Generated seams: {len(seams_out)}")
    print(f"Generated snippets: {len(snippets)}")
    print(f"Output dir: {output_dir}")
    print(f"Web data dir: {web_data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
