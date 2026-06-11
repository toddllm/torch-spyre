# Torch-Spyre Indirect / Paged Attention POC — Report

**Date:** 2026-06-11
**Author:** local POC run (Claude Code)
**Worktree:** `/Users/tdeshane/torch-spyre-open-work/torch-spyre-indirect-paged-attention-poc-wt`
**Branch:** `tdeshane/indirect-paged-attention-poc`
**Base ref:** `origin/pr/2178` @ `7bcf3a4b6f324ffb1c3de0e7318e06310a91ffa8`
**Local commit added:** `2c0324754960bab483797ea593847be127bb5245` (DCO signed, not pushed)

The driving question: can the current public PR stack express the vLLM-style
paged KV attention shape

```text
query + k_pages + v_pages + page_indices (+ optional mask) -> attention output
```

Short answer: **yes, structurally** — PR 2178 already ships a worked
paged-attention example in exactly that shape, plus the full indirect-access
lowering path it depends on. A CPU numerics proof was built and **runs locally**
and passes; the on-device Spyre compile/run path is **structure-checked only**
(no Spyre hardware, no built `torch_spyre._C`, and the pinned `torch~=2.11`
is not installed on this machine).

---

## Refs and exact commits inspected

Mirror used as the source of refs:
`/Users/tdeshane/ibm-git-temp/torch-spyre` (remote `origin` =
`https://github.com/torch-spyre/torch-spyre.git`, with
`+refs/pull/*/head:refs/remotes/origin/pr/*` already configured).

| Ref | Commit | Date | Subject / role |
|---|---|---|---|
| `origin/main` | `6ae38152` | 2026-06-10 | mainline |
| `origin/pr/2178` | `7bcf3a4b` | 2026-06-? | **POC base** — full indirect-access + paged-attention branch |
| `origin/pr/2608` | `a528a468` | 2026-06-10 | slice: detection pass (`detect_indirect_access`) |
| `origin/pr/2620` | `87eb16a3` | 2026-06-11 | slice: indirect store in `spyre_kernel` |
| `origin/pr/2623` | `7ba3f2c3` | 2026-06-11 | slice: full `indirect_access.py` + codegen (`compute_ops`, `superdsc`) |
| `origin/pr/2634` | `1d05d23b` | 2026-06-10 | slice: index→address custom op + C++ address computation |
| `origin/pr/2438` | `0a91f901` | 2026-06-08 | unrelated: async API + distributed (`spyreccl`) |

Merge-base of `main` and `pr/2178`: `f2dfd86c` (2026-06-08).
None of the follow-up PRs contain `pr/2178`'s head; they are branched off recent
`main` (`8711b92c` / `306dbbd0`). They are a **decomposition of the 2178
mega-branch into reviewable slices**, not extensions on top of it.

---

## What PR 2178 adds (the reusable surface)

`git diff --stat f2dfd86c..origin/pr/2178` = **23 files, +3752 / −36**. The
indirect-access lowering path it introduces:

1. **Detection** — `torch_spyre/_inductor/detect_indirect_access.py`. Scans
   Inductor `Pointwise` ops for the indirect pattern (a memory load whose index
   expression contains a `tmp*` symbol = a previously-loaded value) and marks the
   op with `op_info = {index_args, index_value_pairs, tensor_names}`.

2. **Index → address rewrite** — `torch_spyre/_inductor/index_to_address_pass.py`
   (`PatternMatcherPass` `add_index_to_address_pass`). Rewrites `torch.gather`,
   `embedding`, and `aten.index.Tensor` so the index tensor is replaced by an
   *address* tensor produced by the custom op
   `torch.ops.spyre.indices_to_address.default(index, input, dim, segment)`.
   Gated by env `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1`.

3. **Address computation** — `torch_spyre/csrc/spyre_address_computation.{cpp,h}`
   + `module.cpp` binding; the C++ side of `indices_to_address`.

4. **Layout + codegen** — `torch_spyre/_inductor/indirect_access.py` (681 lines)
   provides KERNEL_IDX / INPUT / OUTPUT layout assignment, value↔index pairing,
   and `maxDimSize` rules; consumed by `codegen/superdsc.py`,
   `codegen/compute_ops.py`, `spyre_kernel.py`, `propagate_layouts.py`,
   `op_spec.py`, `passes.py`, `customops.py`.

5. **Worked examples** — `examples/indirect_access/{gather,index_select,embedding,paged_attention}.py`
   and unit tests `tests/indirect_access/test_{indirect_access,detect_indirect_access}.py`.

The key artifact for this POC is
[`examples/indirect_access/paged_attention.py`](../../examples/indirect_access/paged_attention.py):
`create_specialized_paged_attn_kernel(num_blocks)` already takes
`(q, k_pages, v_pages, page_indices, mask_tiles, scale)` and implements
online-softmax attention by gathering one page per loop iteration — i.e. the
vLLM shape is already a first-class example. (Its on-device numeric assertion is
currently commented out with a `TODO: Enable the assertion once the max diff is
reduced.`)

---

## Which follow-up PRs matter

| PR | Matters for paged attention? | Why |
|---|---|---|
| 2608 | Yes (detection slice) | `detect_indirect_access` + `propagate_layouts` hookup — the front of the path. |
| 2620 | Yes (kernel-store slice) | indirect store in `spyre_kernel.py` + `op_spec` — needed to write gathered values. |
| 2623 | Yes (codegen slice) | full `indirect_access.py` + `compute_ops`/`superdsc` codegen — the engine. |
| 2634 | Yes (address slice) | `indices_to_address` custom op + C++ — turns indices into device addresses. |
| 2438 | No | async API / `spyreccl` distributed; unrelated to indirect access. |

**Cherry-pick decision: none cherry-picked.** The POC branch starts from
`pr/2178`, which is the **superset** that already contains the content of 2608 /
2620 / 2623 / 2634 (same files: `detect_indirect_access.py`,
`indirect_access.py` @ 681 lines, the `spyre_kernel`/`compute_ops`/`superdsc`
edits, and the address-computation C++; 2178 carries the pass as
`index_to_address_pass.py` where 2634 renames it `indices_to_address_pass.py`).
Cherry-picking would be redundant and conflict-prone. They are referenced here
for the from-`main` reviewer who wants the path as small reviewable units.
2438 is referenced only and is out of scope.

---

## Files changed (this POC)

| File | Status | Purpose |
|---|---|---|
| `examples/indirect_access/paged_attention_vllm_shape_poc.py` | added | Device-parametrized, CPU-runnable vLLM-shape proof: block-table gather + online-softmax kernel validated vs dense reference. |
| `tests/indirect_access/test_paged_attention_vllm_shape_poc.py` | added | Standalone CPU tests (torch-only): gather correctness, fp32/fp16 numerics, mask wiring. |
| `artifacts/torch-spyre-indirect-paged-attention-poc-2026-06-11/report.md` | added | This report. |
| `artifacts/torch-spyre-indirect-paged-attention-poc-2026-06-11/command-log.md` | added | Command log. |

No existing repo files were modified. The protected worktree
`spyre-inference-impl-agent-wt` was not touched (verified: it is on branch
`tdeshane/spyre-inference-pd-disagg-config-cleanup`, unrelated).

---

## Commands run and results (summary)

Full transcript in `command-log.md`. Highlights:

- `git fetch origin --prune` in the mirror; all seven refs resolve.
- `git worktree list` — POC worktree already present at the target path on
  `tdeshane/indirect-paged-attention-poc` @ `7bcf3a4b` (= `origin/pr/2178`),
  clean.
- PR mapping via `git merge-base` / `git diff --stat` / `git log` (tables above).
- Environment probe: system `python3` 3.9.6 has no torch/sympy; repo pins
  `torch~=2.11.0`. Several unrelated venvs carry torch 2.7/2.9. Used
  `/Users/tdeshane/cleanroom-3/.venv` (torch 2.9.1, sympy 1.14) to run the
  torch-only POC, and `/Users/tdeshane/cleanroom-testing/.venv` (ruff 0.14.13)
  to lint.
- **POC run (CPU):**
  - fp32 → `block-table gather exact_match=True`; `online-softmax vs dense
    max_diff=2.980e-07 allclose=True`; `RESULT: PASS`.
  - fp16 → `max_diff=4.883e-04 allclose(atol=0.02)=True`; `RESULT: PASS`.
- **POC test:** `4/4 passed` run directly with `python`; `4 passed` under
  `pytest … -c /dev/null` (repo conftest bypassed).
- **Lint:** `ruff check` → "All checks passed!"; `ruff format` applied (1 file
  reformatted), re-run still PASS.
- **Existing unit tests blocked:** importing
  `tests/indirect_access/test_indirect_access.py` fails at
  `from torch_spyre._C import DataFormats` (`ModuleNotFoundError: torch_spyre._C`)
  — they need the compiled backend, which is not built here.

---

## Answers to the questions

1. **Reusable surface from 2178?** A complete indirect-access lowering path:
   detection pass → index→address FX pass + `spyre::indices_to_address` custom op
   + C++ address computation → KERNEL_IDX/INPUT/OUTPUT layout + SDSC kernel
   codegen, plus four examples. Paged attention is already a worked example
   (`create_specialized_paged_attn_kernel`).

2. **Can it consume vLLM page lists / block tables / page indices?** Yes,
   structurally. The page list / block table is an int64 index tensor consumed by
   `torch.gather` / `index_select` / `aten.index` over the KV page pool (dim 0),
   which `detect_indirect_access` recognises and `index_to_address_pass` rewrites
   to addresses. The POC's `gather_kv_by_block_table` consumes both a 1D
   `page_indices` and a 2D `[num_seqs, blocks_per_seq]` block table and gathers
   the right pages (exact match), then feeds correct attention. Caveat: the
   shipped example feeds **one scalar page index per loop iteration** (expanded
   into a gather index via `expand_address_tensor`); a true batched 2D block
   table is the natural generalization but is not yet exercised on device.

3. **Which of 2608/2620/2623/2634 matter?** All four — they are the detection,
   kernel-store, codegen, and address-computation slices of 2178. None were
   cherry-picked because 2178 already contains them. 2438 is unrelated.

4. **Runs / compiles / structure-checks locally?** The POC **runs locally on
   CPU** (eager, torch-only) with validated numerics. It does **not** compile
   through the Spyre Inductor backend locally and does **not** execute on the
   Spyre device locally. The existing indirect-access unit tests and the
   `device="spyre"` examples are **structure-checked only** (blocked by missing
   `torch_spyre._C` / torch 2.11 / hardware).

5. **Exact gap before Spyre Inference can use this for real paged KV attention
   (instead of connector-level KV copies)?**
   - **Build/hardware:** a `torch_spyre._C` built against `torch~=2.11` on a
     Spyre-equipped host. Not available locally.
   - **On-device numerics:** the paged-attention example's correctness assertion
     is still disabled (`TODO: Enable the assertion once the max diff is
     reduced`) — the online-softmax-over-gathered-pages path must reach
     acceptable fp16 error on device.
   - **Batched block tables + multicore addressing:** the examples consume a
     single scalar page index per step; `gather.py` notes
     "Address computation for multicore : WIP" and advises `SENCORES=1`.
     Consuming a real 2D vLLM block table across sequences/pages, and the
     multi-core `indices_to_address` address computation, remain to be proven.
   - **Integration:** Spyre Inference would need to hand its block table /
     page-indices tensors directly into a compiled gather-based attention graph
     (with `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1`) instead of copying KV
     through the connector — that wiring does not exist yet.

---

## POC status

- **Built:** `examples/indirect_access/paged_attention_vllm_shape_poc.py` and
  `tests/indirect_access/test_paged_attention_vllm_shape_poc.py`.
- **Runs locally (CPU):** PASS in fp32 (max_diff 2.98e-07) and fp16 (max_diff
  4.88e-04). Block-table gather is an exact match.
- **On device (Spyre):** not run — structure-checked only.
- **Lint:** ruff check + format clean at line-length 88.

## Smallest next runtime test

On a host with `torch~=2.11` and a built `torch_spyre._C` (no Spyre device
needed for this one): run the two pure-Python indirect-access unit tests —

```bash
python -m pytest tests/indirect_access/test_indirect_access.py \
                 tests/indirect_access/test_detect_indirect_access.py
```

Then, on a Spyre-equipped host, the smallest device test:

```bash
SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1 SENCORES=1 \
  python examples/indirect_access/paged_attention_vllm_shape_poc.py --device spyre --dtype float16
```

and compare the device output against the CPU reference this POC already
computes (`paged_attention_dense_reference`), tightening the tolerance until the
paged-attention assertion that is currently a `TODO` in `paged_attention.py` can
be enabled.

## Conclusion

PR 2178 already expresses the vLLM paged KV attention shape and ships the full
indirect-access lowering path it needs; PRs 2608/2620/2623/2634 are its
reviewable slices and require no cherry-pick onto the 2178-based POC branch
(2438 is unrelated). The new POC proves the shape and the online-softmax paged
kernel are numerically correct on CPU. What is not yet provable locally — and is
the real remaining gap — is the on-device path: a torch-2.11 build of
`torch_spyre._C`, batched 2D block-table consumption with multicore address
computation, and acceptable on-device fp16 numerics for the gather-based
attention.

---

## Terminal outcome

```text
poc_runs_locally
```
