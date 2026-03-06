# Appendix: FMS Seam Research Findings

## Status

This appendix captures the research conclusions behind the FMS seam probe.

Current status:
- local CPU eager: validated
- local CPU compiled (`torch.compile(..., backend="inductor")`): validated
- real Spyre/sendnn path: not yet validated

This appendix is intentionally research-oriented. It is not a claim that the old stack now supports async layer-level KV transfer.

## Why this appendix exists

The main architecture note should remain conservative and durable. This appendix records:
- what questions were asked about the old-stack FMS seam
- what was actually verified in code
- what was inferred from that code
- what changed in the experimental outlook
- what is still unproven

## Code paths inspected

### FMS attention seam
- `foundation-model-stack/fms/modules/attention.py`
- `foundation-model-stack/fms/utils/spyre/paged.py`

### Old-stack Spyre ownership of FMS model + KV
- `vllm-spyre/vllm_spyre/model_executor/model_loader/spyre.py`

### Upstream connector selection / composition
- `vllm/vllm/distributed/kv_transfer/kv_connector/factory.py`
- `vllm/vllm/config/kv_transfer.py`

## Questions asked and current answers

### 1. Is there a real FMS attention seam, or only an outer `forward()` wrapper?

Answer: there is a real seam.

FMS attention dispatch is registered through:
- `store`
- `compute_prefill`
- `compute_decode`
- `is_prefill`
- `update_attn_kwargs`
- `validate_attn_kwargs`

Spyre paged attention already plugs into this path. That means the old stack is not a total black box.

Implication:
- the outer bridge is not the only experimental seam
- a deeper seam exists inside attention registration

### 2. Does the old stack still have FMS owning the data plane?

Answer: yes.

`SpyreCausalLM` still owns:
- `past_key_value_states`
- KV tensor allocation
- passing those tensors into the FMS model
- receiving updated KV back from the FMS forward path

Implication:
- the old stack remains FMS-shaped in the data plane
- the architecture note does not need to change its high-level conclusion

### 3. Can per-layer identity be recovered without broad FMS churn?

Answer: yes, with a low-churn approach.

The seam probe shows that a practical path exists:
- stamp `MultiHeadAttention` modules after FMS model construction
- store `layer_idx` / `layer_name` on each module
- inject `layer_idx` into the attention dispatch kwargs inside `MultiHeadAttention.forward`

This avoided editing every model loop.

Implication:
- per-layer identity recovery is practical on the old stack
- broad edits across all FMS model families are not the first move

### 4. Is the seam strong enough for true async load overlap?

Answer: not yet.

The current seam is much stronger for save/export-side signaling than for preload/prefetch overlap.

Reason:
- the strongest current hook is still inside the attention path itself
- that is late relative to an ideal prefetch seam
- there is no first-class pre-attention load/wait hook in the current contract

Implication:
- this seam is promising for save/export readiness experiments
- it is not yet enough evidence for load overlap claims

### 5. Can the seam carry safe metadata without embedding connector logic?

Answer: yes.

The probe was implemented as:
- constant per-layer metadata (`layer_idx`)
- fixed-shape tensor sinks
- no connector calls
- no transport logic
- no scheduler policy input
- no waits or retries in the hook

Implication:
- the seam can remain metadata-shaped
- this is the right experimental boundary

### 6. Is dynamic Python per-step metadata required in the hook?

Answer: no.

The probe uses reset-and-read discipline instead:
- reset sink tensors before each outer forward
- run the compiled/eager forward
- read the sink immediately after the outer forward

This avoided passing dynamic Python step metadata through `attn_kwargs`.

Implication:
- step correlation can remain outside the compiled attention path
- this reduces risk of compile churn and stale signal bugs

### 7. What should “extent tracking” mean in the old-stack paged model?

Answer: it should be narrowed.

The broad phrase `extent tracking` is too loose for this context. The more accurate concept is:
- `materialized coverage`
- partial tail state
- frozen/exportable tail state

For this probe, the most practical signal was:
- `materialized_coverage_this_call`

Implementation currently uses:
- `keys.shape[1]`

Implication:
- the architecture language should stay close to paged/block semantics
- avoid making it sound like a separate prefix-tree ownership model

### 8. Who should control policy?

Answer: split policy into two layers.

`orchestrator/client`
- chooses routing / topology / engine pool
- may choose high-level deployment shape

`runtime/scheduler`
- owns block/page-level policy
- retry vs recompute
- invalidate vs reuse
- batching / timing / placement details

Implication:
- low-level offload/retry/recompute behavior should not be modeled as arbitrary client choice

### 9. Should the runtime flow assume push or pull?

Answer: no.

The durable model should be transport-neutral.

Producer side:
- marks source pages exportable

Consumer side:
- prepares destination placement

Transport:
- may use push, pull, or hybrid behavior depending on backend/deployment

Implication:
- the architecture should not assume a single fixed `export -> transport -> import` ordering

### 10. Should retry policy live in the connector?

Answer: only partially.

Connector / transport should:
- report granular success/failure
- remain idempotent
- surface invalid blocks/regions

Scheduler / runtime should:
- choose retry vs recompute vs fail
- own semantic fallback decisions

Implication:
- connector is not the full retry policy owner

### 11. Should scheduler own raw region handles?

Answer: scheduler should own descriptors, runtime should own live handles.

Useful split:
- scheduler-visible descriptor / export plan
- runtime-local live handle / registration / lifetime object

Implication:
- avoid coupling the scheduler to ephemeral runtime registration state

### 12. Does upstream vLLM already support multiple connector implementations?

Answer: partially.

Upstream today is primarily:
- one configured connector per engine instance
- plus `MultiConnector`

It is not currently best modeled as:
- arbitrary per-request client-chosen connector type

Implication:
- transport backend choice should be described as deployment/runtime-configured first

## Seam probe implementation summary

The local probe touched only four files.

### `vllm-spyre`
- `vllm_spyre/envs.py`
- `vllm_spyre/model_executor/model_loader/spyre.py`

### `foundation-model-stack`
- `fms/modules/attention.py`
- `fms/utils/spyre/paged.py`

Probe shape:
- env gate: `VLLM_SPYRE_ENABLE_FMS_LAYER_PROBE=1`
- stamp `MultiHeadAttention` modules with `layer_idx`
- inject `layer_idx` into dispatch kwargs
- pass tiny fixed-shape probe tensors:
  - `kv_probe_ready`
  - `kv_probe_coverage`
  - `kv_probe_phase`
- in the Spyre paged `store` op, write the sink only after the existing paged store op returns

This keeps the probe:
- metadata-shaped
- local to the save/export-side write point
- free of connector, transport, and policy logic

## Local validation that already passed

### CPU seam smoke
Verified locally with:
- eager
- `torch.compile(..., backend="inductor")`

Results:
- per-layer `ready/coverage/phase` values were correct
- probe survived compiled CPU inference on the real Spyre paged attention registration path

### `vllm-spyre` helper validation
Verified locally that:
- attention module stamping works
- `layer_idx` and `layer_name` are stable
- sink buffers reset correctly before reuse
- snapshot/readback works

## What this result proves

1. The old-stack FMS seam is real.
2. Per-layer identity can be recovered with low churn.
3. A tiny tensor sink is viable.
4. The seam can remain metadata-shaped.
5. Reset/read discipline works without dynamic Python step metadata in the seam.

## What this result does not prove

1. `sendnn` compile survivability
2. async layer-level load overlap
3. connector calls inside the seam
4. transport coordination inside the seam
5. scheduler-visible policy input from the seam
6. block-subset export logic from the seam

## Best current interpretation

The strongest current statement is:

- outer bridge remains the primary old-stack lifecycle mechanism
- there is now a validated secondary FMS seam for layer-local save/export-readiness metadata on local CPU paths
- this improves the old-stack experimental story
- it does not change the long-term architectural conclusion

## Recommended next gate

The next real gate is the internal Spyre/sendnn path.

Validation on the pod should check:
1. compile succeeds
2. probe off vs on produces identical outputs
3. per-layer `ready/coverage/phase` values are correct
4. no stale-signal leakage across steps
5. no obvious graph-break / bad fallback behavior
6. no obvious pathological recompilation or extreme slowdown

## How this should affect the docs

Before `sendnn` validation:
- keep this material in an appendix / local handoff context
- do not upgrade the top-level architecture note materially

After `sendnn` validation, if it passes:
- the old-stack appendix can say that there is a validated secondary FMS seam for layer-local save/export-readiness metadata
- the core note should still remain conservative about async load overlap and connector-in-hook logic

## Candidate follow-on questions

If the pod validation passes, the next questions worth exploring are:
1. can this seam support a save/export-first experiment with outer-bridge-owned policy?
2. can the seam produce slightly richer coverage information without becoming compile-hostile?
3. is there any clean pre-attention preload seam, or is the current seam fundamentally save-side-biased?
4. what is the smallest useful scheduler/runtime descriptor that could consume this metadata later without changing the current architecture boundaries?
