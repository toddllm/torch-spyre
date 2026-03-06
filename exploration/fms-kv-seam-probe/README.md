# FMS KV Seam Probe Handoff

## Goal
Prove a narrow seam claim on the old stack:

- the compiled FMS attention path can carry safe per-layer readiness metadata
- without connector calls, waits, retry logic, transport orchestration, or scheduler policy inside the attention hook

This is a seam probe, not a feature implementation.

## Artifacts
- `vllm-spyre-seam-probe.patch`
- `fms-seam-probe.patch`
- `cpu_smoke_fms_probe.py`
- `spyre_probe_helper_smoke.py`
- `appendix-fms-seam-research.md`

## Local source revisions used to derive the patch
- `vllm-spyre`: `ea62c71fa237ac7eb90d79eed0eb598c1883c4fe`
- `foundation-model-stack`: `de548620726d8c182e6c94da104b68d08c694e00`

Important: the `vllm-spyre` patch was developed locally on a branch created from a docs-oriented working branch. The actual seam change is small and only touches code files, but on the pod it should be transplanted onto the target code branch, not applied blindly because the branch base may differ.

## Files touched by the seam probe
### vllm-spyre
- `vllm_spyre/envs.py`
- `vllm_spyre/model_executor/model_loader/spyre.py`

### foundation-model-stack
- `fms/modules/attention.py`
- `fms/utils/spyre/paged.py`

## Probe shape
### Env gate
- `VLLM_SPYRE_ENABLE_FMS_LAYER_PROBE=1`

### What the probe adds
- stamps each FMS `MultiHeadAttention` module with a stable `layer_idx` and `layer_name`
- injects `layer_idx` into the FMS attention dispatch kwargs
- passes three fixed-shape probe tensors through the existing kwargs path:
  - `kv_probe_ready`
  - `kv_probe_coverage`
  - `kv_probe_phase`
- after the existing Spyre paged KV store op returns, the seam writes:
  - `ready[layer_idx] = 1`
  - `coverage[layer_idx] = keys.shape[1]`
  - `phase[layer_idx] = 0` for prefill, `1` for decode

### What the probe does not add
- no `KVConnector` calls
- no waits
- no retries
- no transport coordination
- no scheduler-visible policy input
- no dynamic Python step metadata in the attention kwargs

## Correlation discipline
Use reset-and-read, not dynamic Python per-step metadata in the seam.

Before each outer forward:
- `ready[:] = 0`
- `coverage[:] = 0`
- `phase[:] = -1`

After each outer forward:
- read the sink immediately
- correlate it with the outer step externally

## What "ready" means here
`ready[layer_idx] = 1` means the layer-local KV write point relevant to export-readiness has completed for this invocation, and materialized coverage for this invocation is known.

It does not mean:
- full prefix ready
- transfer started
- transfer complete
- decode can start

## Recommended pod flow
1. Check out the target `vllm-spyre` code branch on the internal pod.
2. Check out the matching or nearest-possible FMS revision used by that environment.
3. Use the patch files as the shape.
4. If they do not apply cleanly, transplant only the exact logic described above.
5. Enable the env gate only for probe runs.

## Validation gates on the pod
1. Compile gate
- compile succeeds under the real Spyre/sendnn path
- no obvious graph-break or fallback symptoms

2. Correctness gate
- probe off vs on produces identical outputs

3. Signal gate
- per-layer `ready/coverage/phase` values are correct

4. Stale-signal gate
- no leakage across steps
- reset before each forward actually clears the sink

5. Toxicity gate
- no obvious pathological recompilation or extreme slowdown

## Useful interpretation of results
### If it passes on sendnn
You can say:
- old stack has a validated secondary FMS seam for layer-local save/export-readiness metadata
- outer bridge remains primary
- async load overlap is still unproven

### If it fails on sendnn
You can still say:
- CPU eager and CPU compiled paths validated the seam shape
- the old Spyre compiled path does not currently tolerate this seam as implemented
- outer bridge remains the practical old-stack seam

## Suggested pod commands
### CPU smoke against local FMS clone
```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
PYTHONPATH=/path/to/foundation-model-stack \
python cpu_smoke_fms_probe.py
```

### Spyre helper smoke in editable venv
```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
PYTHONPATH=/path/to/vllm:/path/to/foundation-model-stack \
/path/to/vllm-spyre/.venv/bin/python spyre_probe_helper_smoke.py
```

## Next-step boundary
Even if the seam probe passes on sendnn, do not add connector or transport logic into the seam yet. The next justified step would still be a save/export-first experiment, not a load-overlap claim.
