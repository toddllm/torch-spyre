#!/usr/bin/env python3
"""Snippet extraction helpers for seam-based code atlas generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def detect_language(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".sh": "bash",
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cuh": "cpp",
        ".cu": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".txt": "text",
    }.get(suffix, "text")


def read_file_lines(repo_root: Path, rel_path: str) -> list[str]:
    p = repo_root / rel_path
    text = p.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def clamp_range(start: int, end: int, n_lines: int) -> tuple[int, int]:
    start = max(1, start)
    end = min(n_lines, end)
    if end < start:
        end = start
    return start, end


def _symbol_name_match(needle: str, candidate: str) -> bool:
    if needle == candidate:
        return True
    if candidate.endswith("." + needle):
        return True
    # Allow Class.method target matching with nested qualname
    if needle.count(".") >= 1 and candidate.endswith(needle):
        return True
    return False


def extract_by_symbol(
    target: dict[str, Any],
    symbols_by_file: dict[tuple[str, str], list[dict[str, Any]]],
    n_lines: int,
    default_context: int,
) -> list[dict[str, Any]]:
    repo = target["repo"]
    file_path = target["file"]
    symbol = target["symbol"]

    matches: list[dict[str, Any]] = []
    for sym in symbols_by_file.get((repo, file_path), []):
        qual = sym.get("qualname", "")
        name = sym.get("name", "")
        if _symbol_name_match(symbol, qual) or _symbol_name_match(symbol, name):
            start = int(sym["start_line"])
            end = int(sym["end_line"])
            start, end = clamp_range(start, end, n_lines)
            matches.append(
                {
                    "start": start,
                    "end": end,
                    "extractor": "symbol",
                    "anchor": symbol,
                    "matched": qual or name,
                }
            )

    if not matches:
        return []

    # If many matches, keep shortest first then earliest; this keeps targeted methods.
    matches.sort(key=lambda m: (m["end"] - m["start"], m["start"]))
    if target.get("max_hits"):
        return matches[: int(target["max_hits"])]

    # Default to a single best match for symbol extraction.
    return matches[:1]


def extract_by_regex(
    target: dict[str, Any], lines: list[str], n_lines: int, default_context: int
) -> list[dict[str, Any]]:
    pattern = target["regex"]
    before = int(target.get("before", default_context // 5))
    after = int(target.get("after", default_context // 5))
    max_hits = int(target.get("max_hits", 3))

    rgx = re.compile(pattern)
    out: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        if rgx.search(line):
            s, e = clamp_range(idx - before, idx + after, n_lines)
            out.append(
                {
                    "start": s,
                    "end": e,
                    "extractor": "regex",
                    "anchor": pattern,
                    "matched": line.strip()[:200],
                }
            )
            if len(out) >= max_hits:
                break
    return out


def extract_by_literal(
    target: dict[str, Any], lines: list[str], n_lines: int, default_context: int
) -> list[dict[str, Any]]:
    anchor = target["literal"]
    before = int(target.get("before", default_context // 5))
    after = int(target.get("after", default_context // 5))
    max_hits = int(target.get("max_hits", 1))
    match_mode = str(target.get("match_mode", "contains")).strip().lower()
    case_sensitive = bool(target.get("case_sensitive", True))

    needle = anchor if case_sensitive else anchor.lower()
    out: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        hay = line if case_sensitive else line.lower()
        matched = False
        if match_mode == "contains":
            matched = needle in hay
        elif match_mode == "equals":
            matched = hay.strip() == needle.strip()
        elif match_mode == "startswith":
            matched = hay.lstrip().startswith(needle)
        else:
            raise ValueError(f"Unsupported literal match_mode: {match_mode}")

        if matched:
            s, e = clamp_range(idx - before, idx + after, n_lines)
            out.append(
                {
                    "start": s,
                    "end": e,
                    "extractor": "literal",
                    "anchor": anchor,
                    "matched": line.strip()[:200],
                }
            )
            if len(out) >= max_hits:
                break
    return out


def extract_by_literal_any(
    target: dict[str, Any], lines: list[str], n_lines: int, default_context: int
) -> list[dict[str, Any]]:
    candidates = list(target.get("literal_any", []))
    if not candidates:
        return []

    for literal in candidates:
        t = dict(target)
        t["literal"] = literal
        out = extract_by_literal(t, lines, n_lines, default_context)
        if out:
            for item in out:
                item["extractor"] = "literal_any"
                item["anchor"] = f"any_of:{literal}"
            return out
    return []


def extract_by_contains_all(
    target: dict[str, Any], lines: list[str], n_lines: int, default_context: int
) -> list[dict[str, Any]]:
    needles = [str(x) for x in target.get("contains_all", []) if str(x)]
    if not needles:
        return []

    before = int(target.get("before", default_context // 5))
    after = int(target.get("after", default_context // 5))
    max_hits = int(target.get("max_hits", 1))
    case_sensitive = bool(target.get("case_sensitive", True))

    norm_needles = needles if case_sensitive else [n.lower() for n in needles]
    out: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        hay = line if case_sensitive else line.lower()
        if all(n in hay for n in norm_needles):
            s, e = clamp_range(idx - before, idx + after, n_lines)
            out.append(
                {
                    "start": s,
                    "end": e,
                    "extractor": "contains_all",
                    "anchor": " && ".join(needles),
                    "matched": line.strip()[:200],
                }
            )
            if len(out) >= max_hits:
                break
    return out


def extract_by_range(target: dict[str, Any], n_lines: int) -> list[dict[str, Any]]:
    s = int(target["start"])
    e = int(target["end"])
    s, e = clamp_range(s, e, n_lines)
    return [
        {
            "start": s,
            "end": e,
            "extractor": "range",
            "anchor": f"{s}:{e}",
            "matched": "explicit_range",
        }
    ]


def extract_target_ranges(
    target: dict[str, Any],
    lines: list[str],
    symbols_by_file: dict[tuple[str, str], list[dict[str, Any]]],
    default_context: int,
    allow_regex_targets: bool = False,
) -> list[dict[str, Any]]:
    n_lines = len(lines)

    ranges: list[dict[str, Any]] = []
    extractor = str(target.get("extractor", "")).strip().lower()

    if not extractor:
        if "start" in target and "end" in target:
            extractor = "range"
        elif "symbol" in target:
            extractor = "symbol"
        elif "literal" in target:
            extractor = "literal"
        elif "literal_any" in target:
            extractor = "literal_any"
        elif "contains_all" in target:
            extractor = "contains_all"
        elif "regex" in target or "grep" in target:
            extractor = "regex"

    if extractor == "range":
        ranges.extend(extract_by_range(target, n_lines))
    elif extractor == "symbol":
        ranges.extend(extract_by_symbol(target, symbols_by_file, n_lines, default_context))
    elif extractor == "literal":
        ranges.extend(extract_by_literal(target, lines, n_lines, default_context))
    elif extractor == "literal_any":
        ranges.extend(extract_by_literal_any(target, lines, n_lines, default_context))
    elif extractor == "contains_all":
        ranges.extend(extract_by_contains_all(target, lines, n_lines, default_context))
    elif extractor == "regex":
        if not allow_regex_targets:
            raise ValueError(
                "Regex extractor disabled by policy. Use symbol/literal/range typed anchors."
            )
        regex_target = dict(target)
        if "grep" in regex_target and "regex" not in regex_target:
            regex_target["regex"] = regex_target["grep"]
        ranges.extend(extract_by_regex(regex_target, lines, n_lines, default_context))
    else:
        raise ValueError(
            f"Unknown or unsupported extractor for target file={target.get('file')} "
            f"repo={target.get('repo')} extractor={extractor or '<none>'}"
        )

    # de-duplicate ranges
    seen: set[tuple[int, int]] = set()
    deduped: list[dict[str, Any]] = []
    for r in ranges:
        key = (int(r["start"]), int(r["end"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def slice_lines(lines: list[str], start: int, end: int) -> list[str]:
    return lines[start - 1 : end]


def with_line_numbers(lines: list[str], start: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, line in enumerate(lines, start=start):
        out.append({"line": i, "text": line})
    return out


def compute_context_bounds(start: int, end: int, n_lines: int, context: int) -> tuple[int, int]:
    return clamp_range(start - context, end + context, n_lines)
