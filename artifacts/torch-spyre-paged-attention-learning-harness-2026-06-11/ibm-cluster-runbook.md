# IBM Spyre Cluster Runbook — Paged-Attention Learning Harness

Generic runbook for executing the harness on a Spyre-capable host. It assumes a
host with the Spyre runtime, a built `torch_spyre` (compiled `torch_spyre._C`)
against the repo-pinned `torch`, and a working C/C++ toolchain. No cluster names,
endpoints, credentials, or absolute paths are included on purpose — fill those in
from your own environment.

## 0. Prerequisites (verify before running anything)

```bash
# From the repo root, on the harness branch.
python examples/indirect_access/paged_attention_learning_harness.py --mode probe-env
```

A healthy host shows `"torch_spyre_importable": true` and a `torch_version`
matching the repo pin. If `torch_spyre_importable` is `false`, stop and fix the
build first — every kernel will otherwise classify as `skipped_backend_missing`.

## 1. Required environment

```bash
# Gate the Spyre index->address rewrite (the harness sets this for --device spyre,
# but export it explicitly so it is unambiguous in logs).
export SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1

# Indirect-access address computation is single-core for now
# (examples/indirect_access/gather.py: "Address computation for multicore : WIP").
# Pin one core for the first runs.
export SENCORES=1
```

## 2. Smallest first run (start here)

Run the **simplest** indirect form alone — `indirect_gather_shape` is a single
FX graph (op_count ~7), uses the `torch.gather`-over-dim-0 shape the rewrite pass
keys on, and avoids `torch.maximum` (which currently lacks a Spyre lowering):

```bash
export SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1
export SENCORES=1
python examples/indirect_access/paged_attention_learning_harness.py \
  --device spyre \
  --dtype float16 \
  --kernel indirect_gather_shape \
  --mode single \
  --compile \
  --json-out /tmp/paged-attn-spyre-indirect.json
```

If that is `compile_ok` (or at least runs eager-clean without `--compile`), widen
to all three forms:

```bash
python examples/indirect_access/paged_attention_learning_harness.py \
  --device spyre \
  --dtype float16 \
  --kernel all \
  --mode single \
  --compile \
  --json-out /tmp/paged-attn-spyre-single.json
```

## 3. Broader sweep

```bash
export SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1
export SENCORES=1
python examples/indirect_access/paged_attention_learning_harness.py \
  --device spyre \
  --dtype float16 \
  --kernel all \
  --mode sweep \
  --compile \
  --json-out /tmp/paged-attn-spyre-sweep.json \
  --markdown-out /tmp/paged-attn-spyre-sweep.md
```

The sweep walks both dtypes, GQA group sizes, plain MHA, variable context
lengths, a fully-masked block, and a decode-shaped `seq_len=1` case. Every case is
scored against an on-host CPU fp32 oracle the harness computes itself, so numeric
errors are caught automatically (`max_abs_diff`, `allclose`).

To bound numeric drift in CI later, add `--fail-on-mismatch` (exit non-zero if any
case is `numeric_mismatch`).

## 4. Reading the JSON `status` field

Each result row carries a `status`. Triage in this order:

| status | meaning | what to do |
|---|---|---|
| `skipped_backend_missing` | `torch_spyre` / device not available | Fix the build/host; nothing below ran. |
| `compile_failed` | Inductor/Spyre backend could not build the kernel | Read `error_type`/`error_message` and the `explain` summary. **Most informative first failure.** |
| `graph_break_or_explain_only` | compiled but Dynamo split the graph | Inspect `explain.break_reasons`; a break upstream of the gather can hide the indirect access from the rewrite pass. |
| `runtime_failed` | kernel raised at execution | Device/addressing error; compare against the CPU oracle. |
| `numeric_mismatch` | ran, but output diverged from the fp32 oracle | Likely addressing (`SENCORES`/`indices_to_address`) or fp16 tolerance. |
| `compile_ok` / `eager_ok` | success (compiled / eager) | Tighten tolerances; this is the target. |

## 5. Which first failure is most informative

Expect failures roughly in this order; the **first `compile_failed`** is the most
informative because its `error_message` + `explain` localise the gap:

1. **`online_page_loop` → `compile_failed`** is expected and *not* the blocker to
   chase first: it uses `torch.maximum`, which has no Spyre Inductor lowering
   (`paged_attention.py` works around it with a `stack(...).max(dim=0)` helper).
   Treat this as known; focus on the gather forms.
2. **`indirect_gather_shape` → `compile_failed`** is the **most informative**
   signal. This is the bare `torch.gather`-over-dim-0 indirect access with no
   `torch.maximum`. If it fails to compile, the gap is in the indirect-access
   lowering itself (`indices_to_address` / layout assignment), and the
   `error_message` points straight at it.
3. **`indirect_gather_shape` → `numeric_mismatch`** (it compiled and ran, but
   diverged from the CPU oracle) points at **address computation** — re-confirm
   `SENCORES=1`, since multicore addressing is WIP.

Recommended escalation: get `indirect_gather_shape` to `compile_ok` + `allclose`
first (single core, fp16, single config), then re-enable the broader sweep, then
swap `torch.maximum` in `online_page_loop` for the `stack/max` workaround and add
it back.

## 6. Capture for hand-off

Copy the `*.json` (and `*.md`) outputs off the host. The JSON includes an
`environment` block (torch version, backend name, env flags) plus every classified
row, which is enough to reproduce and compare runs without re-reading console logs.
