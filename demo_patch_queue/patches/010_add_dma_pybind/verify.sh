#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

python - <<'PY'
from pathlib import Path
import re

module_cpp = Path("torch_spyre/csrc/module.cpp").read_text()

assert "void spyre_tensor_set_dmpa(at::Tensor tensor, uint64_t dmpa)" in module_cpp
assert "uint64_t spyre_tensor_get_dmpa(at::Tensor tensor)" in module_cpp

set_lines = re.findall(
    r'^[ \t]*m\.def\(\s*"set_dma_address"\s*,\s*&spyre::spyre_tensor_set_dmpa\s*\);\s*$',
    module_cpp,
    flags=re.MULTILINE,
)
get_lines = re.findall(
    r'^[ \t]*m\.def\(\s*"get_dma_address"\s*,\s*&spyre::spyre_tensor_get_dmpa\s*\);\s*$',
    module_cpp,
    flags=re.MULTILINE,
)
assert len(set_lines) == 1, f"expected exactly one set_dma_address binding, found {len(set_lines)}"
assert len(get_lines) == 1, f"expected exactly one get_dma_address binding, found {len(get_lines)}"

print("OK: source-level DMA invariants present")
PY
