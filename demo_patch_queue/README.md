# Demo Patch Queue Prototype

This directory contains a small "patch queue as code" prototype that keeps a
demo-only C++/pybind patch applied with structure-aware edits.

## Layout

```text
demo_patch_queue/
  README.md
  apply_all.py
  patches/
    010_add_dma_pybind/
      patch.json
      patch.py
      verify.sh
```

## What It Patches

Patch `010_add_dma_pybind` enforces these invariants in
`torch_spyre/csrc/module.cpp`:

- `spyre::spyre_tensor_set_dmpa(...)` helper exists inside `namespace spyre`.
- `spyre::spyre_tensor_get_dmpa(...)` helper exists inside `namespace spyre`.
- `PYBIND11_MODULE(_C, m)` exports:
  - `set_dma_address`
  - `get_dma_address`

The patcher is idempotent. Running it repeatedly is safe.

## Usage

Apply + verify all patches:

```bash
python demo_patch_queue/apply_all.py
```

Dry-run only:

```bash
python demo_patch_queue/apply_all.py --dry-run
```

Skip verification:

```bash
python demo_patch_queue/apply_all.py --skip-verify
```

## Verification Model

Local verification is intentionally source-level only:

- helper function signatures must exist exactly as expected
- DMA pybind exports must exist exactly once with canonical targets

No local build/import verification is attempted because `sendnn` is not
available in local/open environments.
