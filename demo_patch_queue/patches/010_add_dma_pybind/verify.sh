#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

if ! python - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("sendnn") else 1)
PY
then
  echo "WARN: sendnn is not installed; skipping build/import verification."
  python - <<'PY'
from pathlib import Path

module_cpp = Path("torch_spyre/csrc/module.cpp").read_text()
assert "void spyre_tensor_set_dmpa(at::Tensor tensor, uint64_t dmpa)" in module_cpp
assert "uint64_t spyre_tensor_get_dmpa(at::Tensor tensor)" in module_cpp
assert 'm.def("set_dma_address", &spyre::spyre_tensor_set_dmpa);' in module_cpp
assert 'm.def("get_dma_address", &spyre::spyre_tensor_get_dmpa);' in module_cpp
print("OK: source-level DMA invariants present (build skipped)")
PY
  exit 0
fi

python -m pip install -v -e .
python - <<'PY'
import torch_spyre._C as C

assert hasattr(C, "set_dma_address"), "missing set_dma_address"
assert hasattr(C, "get_dma_address"), "missing get_dma_address"
print("OK: DMA symbols present")
PY
