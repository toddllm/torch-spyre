# Command Log — Indirect / Paged Attention POC (2026-06-11)

Machine: macOS (darwin 25.5.0). Mirror:
`/Users/tdeshane/ibm-git-temp/torch-spyre`. POC worktree:
`/Users/tdeshane/torch-spyre-open-work/torch-spyre-indirect-paged-attention-poc-wt`.

Venvs used (unrelated projects; borrowed only for a compatible torch / ruff):

- `/Users/tdeshane/cleanroom-3/.venv` — Python 3.12, torch 2.9.1, sympy 1.14.0 (POC run)
- `/Users/tdeshane/cleanroom-testing/.venv` — ruff 0.14.13 (lint)

---

## 1. Refresh refs in the mirror

```bash
cd /Users/tdeshane/ibm-git-temp/torch-spyre
git status -sb                       # main...origin/main [behind 30]
git remote -v
git config --get-all remote.origin.fetch
#   +refs/heads/*:refs/remotes/origin/*
#   +refs/pull/*/head:refs/remotes/origin/pr/*   (PR refspec already present)
git fetch origin --prune             # (no new output)
```

Resolve all required refs:

```bash
for r in main pr/2178 pr/2608 pr/2620 pr/2623 pr/2634 pr/2438; do
  printf "%-12s " "$r:"; git rev-parse --short "origin/$r"; done
# main:     6ae38152
# pr/2178:  7bcf3a4b
# pr/2608:  a528a468
# pr/2620:  87eb16a3
# pr/2623:  7ba3f2c3
# pr/2634:  1d05d23b
# pr/2438:  0a91f901
```

## 2. Worktree (already created at target path from pr/2178)

```bash
git worktree list
#   .../torch-spyre-indirect-paged-attention-poc-wt  7bcf3a4b [tdeshane/indirect-paged-attention-poc]
cd /Users/tdeshane/torch-spyre-open-work/torch-spyre-indirect-paged-attention-poc-wt
git status -sb        # clean, on tdeshane/indirect-paged-attention-poc
git rev-parse HEAD    # 7bcf3a4b6f324ffb1c3de0e7318e06310a91ffa8
git log --oneline -5  # 7bcf3a4b Fix index expression ...
```

## 3. Read indirect-access examples + tests

```bash
ls examples/indirect_access/
#   embedding.py  gather.py  index_select.py  paged_attention.py
git ls-files | grep -iE 'indirect|paged'
#   examples/indirect_access/*.py
#   tests/indirect_access/test_{detect_,}indirect_access.py
#   torch_spyre/_inductor/{detect_indirect_access,indirect_access}.py
#   tests/configs/.../test_{detect_,}indirect_access_config.yaml
```

Read: `examples/indirect_access/paged_attention.py` (has
`create_specialized_paged_attn_kernel(q, k_pages, v_pages, page_indices,
mask_tiles, scale)`), `torch_spyre/_inductor/indirect_access.py`,
`detect_indirect_access.py`, both unit-test files,
`index_to_address_pass.py`, `customops.py`, `torch_spyre/__init__.py`,
`gather.py`.

## 4. Map PR 2178 + follow-ups

```bash
cd /Users/tdeshane/ibm-git-temp/torch-spyre
MB=$(git merge-base origin/main origin/pr/2178)   # f2dfd86c (2026-06-08)
git diff --stat $MB origin/pr/2178
#   23 files changed, 3752 insertions(+), 36 deletions(-)
#   new: examples/indirect_access/{embedding,gather,index_select,paged_attention}.py
#        torch_spyre/_inductor/{detect_indirect_access,indirect_access,index_to_address_pass}.py
#        torch_spyre/csrc/spyre_address_computation.{cpp,h}
#        + compute_ops.py, superdsc.py, spyre_kernel.py, customops.py, op_spec.py,
#          passes.py, propagate_layouts.py edits

for pr in 2608 2620 2623 2634 2438; do
  MB_MAIN=$(git merge-base origin/main origin/pr/$pr)
  git merge-base --is-ancestor origin/pr/2178 origin/pr/$pr && echo "$pr contains 2178" || echo "$pr does NOT contain 2178"
  git diff --stat $MB_MAIN origin/pr/$pr | tail -1
done
# none contain pr/2178; each is a slice off recent main:
#   2608 -> detect_indirect_access.py + propagate_layouts.py (+447)
#   2620 -> spyre_kernel.py (+176) + op_spec.py + indirect_access.py(subset)
#   2623 -> indirect_access.py(681) + compute_ops.py + superdsc.py (+1458/-26)
#   2634 -> indices_to_address_pass.py + customops.py + spyre_address_computation.{cpp,h}
#   2438 -> async api + spyre_distributed.cpp (UNRELATED)
```

## 5. Environment probe

```bash
/usr/bin/python3 --version                 # Python 3.9.6
python3 -c "import torch"                   # ModuleNotFoundError: No module named 'torch'
python3 -c "import sympy"                   # ModuleNotFoundError
grep -n 'torch~=' pyproject.toml            # torch~=2.11.0  (pinned)
pyenv versions                             # system / 3.7 / 3.11 / 3.12 ; no torch envs
# Found torch in unrelated venvs:
#   /Users/tdeshane/cleanroom-3/.venv      -> torch 2.9.1, sympy 1.14.0, py3.12
#   /Users/tdeshane/cleanroom-testing/.venv-> torch 2.9.1, ruff 0.14.13

VENV=/Users/tdeshane/cleanroom-3/.venv
PYTHONPATH=$(pwd) "$VENV/bin/python" -c "import torch_spyre; ..."
# torch_spyre package imports, but:
#   from torch_spyre._inductor.op_spec import OpSpec
#   -> from torch_spyre._C import DataFormats
#   -> ModuleNotFoundError: No module named 'torch_spyre._C'   (extension not built)
```

Conclusion: existing indirect-access unit tests and `device="spyre"` examples
cannot run here (need built `_C` + torch 2.11 + Spyre HW). A torch-only CPU
numerics POC is the runnable check.

## 6. POC author + run (CPU)

```bash
# new files:
#   examples/indirect_access/paged_attention_vllm_shape_poc.py
#   tests/indirect_access/test_paged_attention_vllm_shape_poc.py

VENV=/Users/tdeshane/cleanroom-3/.venv
"$VENV/bin/python" examples/indirect_access/paged_attention_vllm_shape_poc.py --device cpu --dtype float32
# [1] block-table gather: pages(12,2,32,64) x block_table(2,4) -> (2,4,2,32,64) exact_match=True
# [2] online-softmax vs dense: out(1,8,16,64) max_diff=2.980e-07 allclose(atol=0.0001)=True
# RESULT: PASS

"$VENV/bin/python" examples/indirect_access/paged_attention_vllm_shape_poc.py --device cpu --dtype float16
# [2] ... max_diff=4.883e-04 allclose(atol=0.02)=True
# RESULT: PASS
```

## 7. Focused test (CPU), two invocation paths

```bash
# direct (no repo conftest, no torch_spyre):
"$VENV/bin/python" tests/indirect_access/test_paged_attention_vllm_shape_poc.py
# PASS test_block_table_gather_matches_indexing
# PASS test_online_softmax_matches_dense_fp32
# PASS test_online_softmax_matches_dense_fp16
# PASS test_nontrivial_mask_changes_output
# 4/4 passed   (exit 0)

# pytest, repo conftest bypassed (-c /dev/null) so torch_spyre._C is not required:
"$VENV/bin/python" -m pytest tests/indirect_access/test_paged_attention_vllm_shape_poc.py \
    -p no:cacheprovider -o addopts="" -c /dev/null -q
# 4 passed in 0.15s
```

Confirm the existing unit test is blocked (for the record):

```bash
PYTHONPATH=$(pwd) "$VENV/bin/python" -c "import tests.indirect_access.test_indirect_access"
# ... from torch_spyre._C import DataFormats
# ModuleNotFoundError: No module named 'torch_spyre._C'
```

## 8. Lint

```bash
RUFF=/Users/tdeshane/cleanroom-testing/.venv/bin/ruff
grep -nA1 '\[tool.ruff' pyproject.toml          # line-length = 88
"$RUFF" check  examples/.../paged_attention_vllm_shape_poc.py tests/.../test_paged_attention_vllm_shape_poc.py
# All checks passed!
"$RUFF" format examples/.../paged_attention_vllm_shape_poc.py tests/.../test_paged_attention_vllm_shape_poc.py
# 1 file reformatted, 1 file left unchanged
"$RUFF" format --check ...                       # 2 files already formatted
# re-ran POC + test after format -> still PASS / 4/4 passed
```

## 9. Local commit (DCO signed, not pushed)

```bash
git add examples/indirect_access/paged_attention_vllm_shape_poc.py \
        tests/indirect_access/test_paged_attention_vllm_shape_poc.py
git commit -s -m "poc(indirect-access): vLLM-shape paged attention CPU proof"
git log --oneline -1     # 2c032475 poc(indirect-access): vLLM-shape paged attention CPU proof
git rev-parse HEAD       # 2c0324754960bab483797ea593847be127bb5245
# Signed-off-by: toddllm <todd.deshane@gmail.com>
```

Boundary check — protected worktree untouched:

```bash
git -C /Users/tdeshane/torch-spyre-open-work/spyre-inference-impl-agent-wt status -sb
# ## tdeshane/spyre-inference-pd-disagg-config-cleanup...   (unrelated branch, not modified)
```

No push, no PR (per work order).
