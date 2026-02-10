#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
from pathlib import Path

PATCH_ID = "010_add_dma_pybind"

CANONICAL_SET_BINDING = '  m.def("set_dma_address", &spyre::spyre_tensor_set_dmpa);\n'
CANONICAL_GET_BINDING = '  m.def("get_dma_address", &spyre::spyre_tensor_get_dmpa);\n'

SET_HELPER = """void spyre_tensor_set_dmpa(at::Tensor tensor, uint64_t dmpa) {
  auto *ctx =
      static_cast<SharedOwnerCtx *>(tensor.storage().data_ptr().get_context());
  DmpaSetBytes(ctx->owner, dmpa);
}

"""

GET_HELPER = """uint64_t spyre_tensor_get_dmpa(at::Tensor tensor) {
  auto *ctx =
      static_cast<SharedOwnerCtx *>(tensor.storage().data_ptr().get_context());
  return DmpaAsBytes(ctx->owner);
}

"""


def detect(repo_root: Path) -> dict:
    module_path = find_module_cpp(repo_root)
    text = module_path.read_text()
    ns_open, ns_close = find_namespace_spyre_block(text)
    mod_open, mod_close = find_pybind_module_block(text)

    ns_body = text[ns_open + 1 : ns_close]
    mod_body = text[mod_open + 1 : mod_close]

    helper_set_exists = bool(re.search(r"\bspyre_tensor_set_dmpa\s*\(", ns_body))
    helper_get_exists = bool(re.search(r"\bspyre_tensor_get_dmpa\s*\(", ns_body))

    literal_set_count = count_matches(
        mod_body, r'^[ \t]*m\.def\(\s*"set_dma_address"\s*,[^\n]*\);\s*$'
    )
    literal_get_count = count_matches(
        mod_body, r'^[ \t]*m\.def\(\s*"get_dma_address"\s*,[^\n]*\);\s*$'
    )
    canonical_set_count = count_matches(
        mod_body,
        r'^[ \t]*m\.def\(\s*"set_dma_address"\s*,\s*&spyre::spyre_tensor_set_dmpa\s*\);\s*$',
    )
    canonical_get_count = count_matches(
        mod_body,
        r'^[ \t]*m\.def\(\s*"get_dma_address"\s*,\s*&spyre::spyre_tensor_get_dmpa\s*\);\s*$',
    )

    includes_cstdint = bool(
        re.search(r'^\s*#\s*include\s*<cstdint>\s*$', text, flags=re.MULTILINE)
    )

    applied = (
        helper_set_exists
        and helper_get_exists
        and literal_set_count == 1
        and literal_get_count == 1
        and canonical_set_count == 1
        and canonical_get_count == 1
    )

    return {
        "patch_id": PATCH_ID,
        "module_path": str(module_path),
        "helpers": {
            "spyre_tensor_set_dmpa": helper_set_exists,
            "spyre_tensor_get_dmpa": helper_get_exists,
        },
        "bindings": {
            "set_dma_address_literal_count": literal_set_count,
            "get_dma_address_literal_count": literal_get_count,
            "set_dma_address_canonical_count": canonical_set_count,
            "get_dma_address_canonical_count": canonical_get_count,
        },
        "includes": {"cstdint": includes_cstdint},
        "applied": applied,
    }


def apply(repo_root: Path, dry_run: bool = False) -> dict:
    module_path = find_module_cpp(repo_root)
    original = module_path.read_text()
    updated = original
    changes: list[str] = []

    updated, helper_changes = ensure_helpers(updated)
    changes.extend(helper_changes)

    updated, include_changed = ensure_cstdint_include(updated)
    if include_changed:
        changes.append("added #include <cstdint>")

    updated, binding_changed = ensure_bindings(updated)
    if binding_changed:
        changes.append("updated PYBIND11 DMA exports")

    changed = updated != original
    if changed and not dry_run:
        module_path.write_text(updated)

    return {
        "patch_id": PATCH_ID,
        "module_path": str(module_path),
        "changed": changed,
        "changes": changes,
    }


def verify(repo_root: Path) -> tuple[bool, str]:
    verify_script = Path(__file__).resolve().parent / "verify.sh"
    completed = subprocess.run(["bash", str(verify_script)], cwd=repo_root)
    if completed.returncode != 0:
        return False, f"{verify_script} exited {completed.returncode}"
    return True, f"{verify_script} passed"


def find_module_cpp(repo_root: Path) -> Path:
    candidate = repo_root / "torch_spyre/csrc/module.cpp"
    if candidate.exists():
        return candidate

    completed = subprocess.run(
        ["git", "grep", "-n", "PYBIND11_MODULE(_C, m)"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError("Unable to locate module.cpp containing PYBIND11_MODULE(_C, m)")

    first_line = completed.stdout.strip().splitlines()[0]
    module_relpath = first_line.split(":", 1)[0]
    module_path = repo_root / module_relpath
    if not module_path.exists():
        raise RuntimeError(f"Resolved module path does not exist: {module_path}")
    return module_path


def ensure_helpers(text: str) -> tuple[str, list[str]]:
    ns_open, ns_close = find_namespace_spyre_block(text)
    ns_body = text[ns_open + 1 : ns_close]
    added: list[str] = []
    snippets: list[str] = []

    if not re.search(r"\bspyre_tensor_set_dmpa\s*\(", ns_body):
        snippets.append(SET_HELPER)
        added.append("added spyre_tensor_set_dmpa helper")
    if not re.search(r"\bspyre_tensor_get_dmpa\s*\(", ns_body):
        snippets.append(GET_HELPER)
        added.append("added spyre_tensor_get_dmpa helper")

    if not snippets:
        return text, added

    insert_idx = find_namespace_insert_idx(text, ns_open, ns_close)
    helper_block = "".join(snippets)
    updated = text[:insert_idx] + helper_block + text[insert_idx:]
    return updated, added


def ensure_cstdint_include(text: str) -> tuple[str, bool]:
    if "uint64_t" not in text:
        return text, False
    if re.search(r'^\s*#\s*include\s*<cstdint>\s*$', text, flags=re.MULTILINE):
        return text, False

    cstdlib_match = re.search(r"^\s*#\s*include\s*<cstdlib>.*$", text, flags=re.MULTILINE)
    if cstdlib_match:
        insert_idx = line_end_idx(text, cstdlib_match.end())
        return text[:insert_idx] + "#include <cstdint>\n" + text[insert_idx:], True

    include_matches = list(re.finditer(r"^\s*#\s*include\b.*$", text, flags=re.MULTILINE))
    if include_matches:
        insert_idx = line_end_idx(text, include_matches[-1].end())
        return text[:insert_idx] + "#include <cstdint>\n" + text[insert_idx:], True

    return "#include <cstdint>\n" + text, True


def ensure_bindings(text: str) -> tuple[str, bool]:
    mod_open, mod_close = find_pybind_module_block(text)
    body_start = mod_open + 1
    body_end = mod_close
    body = text[body_start:body_end]

    literal_set_count = count_matches(
        body, r'^[ \t]*m\.def\(\s*"set_dma_address"\s*,[^\n]*\);\s*$'
    )
    literal_get_count = count_matches(
        body, r'^[ \t]*m\.def\(\s*"get_dma_address"\s*,[^\n]*\);\s*$'
    )
    canonical_set_count = count_matches(
        body,
        r'^[ \t]*m\.def\(\s*"set_dma_address"\s*,\s*&spyre::spyre_tensor_set_dmpa\s*\);\s*$',
    )
    canonical_get_count = count_matches(
        body,
        r'^[ \t]*m\.def\(\s*"get_dma_address"\s*,\s*&spyre::spyre_tensor_get_dmpa\s*\);\s*$',
    )

    set_ok = literal_set_count == 1 and canonical_set_count == 1
    get_ok = literal_get_count == 1 and canonical_get_count == 1
    if set_ok and get_ok:
        return text, False

    remove_re = re.compile(
        r'^[ \t]*m\.def\(\s*"(?:set_dma_address|get_dma_address)"\s*,[^\n]*\);\s*\n?',
        flags=re.MULTILINE,
    )
    cleaned_body = remove_re.sub("", body)
    insert_idx = find_pybind_insert_idx(cleaned_body)
    insertion = CANONICAL_SET_BINDING + CANONICAL_GET_BINDING
    rebuilt_body = cleaned_body[:insert_idx] + insertion + cleaned_body[insert_idx:]
    updated = text[:body_start] + rebuilt_body + text[body_end:]
    return updated, True


def find_namespace_spyre_block(text: str) -> tuple[int, int]:
    match = re.search(r"\bnamespace\s+spyre\s*\{", text)
    if not match:
        raise RuntimeError("Could not find `namespace spyre {`")
    open_idx = text.find("{", match.start(), match.end())
    if open_idx < 0:
        raise RuntimeError("Could not locate opening brace for namespace spyre")
    close_idx = find_matching_brace(text, open_idx)
    return open_idx, close_idx


def find_namespace_insert_idx(text: str, ns_open: int, ns_close: int) -> int:
    for anchor in re.finditer(r"^[ \t]*\}\s*//\s*namespace\s+spyre\s*$", text, flags=re.MULTILINE):
        if ns_open < anchor.start() <= ns_close:
            return line_start_idx(text, anchor.start())
    return line_start_idx(text, ns_close)


def find_pybind_module_block(text: str) -> tuple[int, int]:
    match = re.search(r"PYBIND11_MODULE\s*\(\s*_C\s*,\s*m\s*\)\s*\{", text)
    if not match:
        raise RuntimeError("Could not find `PYBIND11_MODULE(_C, m)` block")
    open_idx = text.find("{", match.start(), match.end())
    if open_idx < 0:
        raise RuntimeError("Could not locate opening brace for PYBIND11_MODULE")
    close_idx = find_matching_brace(text, open_idx)
    return open_idx, close_idx


def find_pybind_insert_idx(body: str) -> int:
    empty_layout = re.search(
        r'^[ \t]*m\.def\(\s*"empty_with_layout"\s*,\s*&spyre::empty_with_layout\s*\);\s*$',
        body,
        flags=re.MULTILINE,
    )
    if empty_layout:
        return line_end_idx(body, empty_layout.end())

    def_matches = list(re.finditer(r"^[ \t]*m\.def\(.*\);\s*$", body, flags=re.MULTILINE))
    if def_matches:
        last = def_matches[0]
        for match in def_matches[1:]:
            gap = body[last.end() : match.start()]
            if re.fullmatch(r"[ \t\r\n]*", gap):
                last = match
            else:
                break
        return line_end_idx(body, last.end())

    doc_match = re.search(r"^[ \t]*m\.doc\(\).*$", body, flags=re.MULTILINE)
    if doc_match:
        return line_end_idx(body, doc_match.end())

    return 0


def count_matches(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def line_start_idx(text: str, idx: int) -> int:
    start = text.rfind("\n", 0, idx)
    return 0 if start < 0 else start + 1


def line_end_idx(text: str, idx: int) -> int:
    end = text.find("\n", idx)
    return len(text) if end < 0 else end + 1


def find_matching_brace(text: str, open_idx: int) -> int:
    depth = 1
    i = open_idx + 1
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False
    raw_end: str | None = None

    while i < len(text):
        if in_line_comment:
            if text[i] == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if text[i : i + 2] == "*/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if raw_end is not None:
            end_idx = text.find(raw_end, i)
            if end_idx < 0:
                raise RuntimeError("Unterminated raw string while scanning braces")
            i = end_idx + len(raw_end)
            raw_end = None
            continue

        if in_string:
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == '"':
                in_string = False
            i += 1
            continue

        if in_char:
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == "'":
                in_char = False
            i += 1
            continue

        two = text[i : i + 2]
        if two == "//":
            in_line_comment = True
            i += 2
            continue
        if two == "/*":
            in_block_comment = True
            i += 2
            continue

        if text[i] == "R" and i + 1 < len(text) and text[i + 1] == '"':
            delim_start = i + 2
            paren_idx = text.find("(", delim_start)
            if paren_idx > -1:
                delim = text[delim_start:paren_idx]
                raw_end = ")" + delim + '"'
                i = paren_idx + 1
                continue

        if text[i] == '"':
            in_string = True
            i += 1
            continue
        if text[i] == "'":
            in_char = True
            i += 1
            continue

        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1

    raise RuntimeError("No matching closing brace found")
