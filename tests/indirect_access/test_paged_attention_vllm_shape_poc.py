# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU correctness tests for the vLLM-shape paged-attention POC.

These tests need only ``torch`` -- no ``torch_spyre._C`` and no Spyre
hardware -- so they can validate the *numerics* of the paged / online-softmax
attention kernel on a developer laptop. They are written to run two ways:

* directly (no Spyre runtime / repo conftest required)::

      python tests/indirect_access/test_paged_attention_vllm_shape_poc.py

* under pytest, where a CPU-capable torch is importable.
"""

import os
import sys

import torch

# Make the POC module importable without installing the package.
_EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples",
    "indirect_access",
)
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

import paged_attention_vllm_shape_poc as poc  # noqa: E402


def test_block_table_gather_matches_indexing():
    """The page list / block table gather equals plain advanced indexing."""
    gen = torch.Generator().manual_seed(1)
    pages = torch.randn(10, 2, 16, 32, generator=gen)
    block_table = torch.tensor([[0, 4, 9], [7, 1, 2]], dtype=torch.int64)

    gathered = poc.gather_kv_by_block_table(pages, block_table)

    assert tuple(gathered.shape) == (2, 3, 2, 16, 32)
    assert torch.equal(gathered, pages[block_table])


def test_online_softmax_matches_dense_fp32():
    """The online-softmax paged kernel equals dense attention in fp32."""
    data = poc._build_inputs(device="cpu", dtype=torch.float32)
    ref = poc.paged_attention_dense_reference(
        data["query"],
        data["k_pages"],
        data["v_pages"],
        data["page_indices"],
        data["full_mask"],
        data["scale"],
    )
    paged = poc.paged_attention_online_softmax(
        data["query"],
        data["k_pages"],
        data["v_pages"],
        data["page_indices"],
        data["mask_tiles"],
        data["scale"],
    )
    assert ref.shape == paged.shape
    assert torch.allclose(ref, paged, rtol=1e-4, atol=1e-4)


def test_online_softmax_matches_dense_fp16():
    """The kernel stays close to dense attention in fp16 (Spyre default dtype)."""
    data = poc._build_inputs(device="cpu", dtype=torch.float16)
    ref = poc.paged_attention_dense_reference(
        data["query"],
        data["k_pages"],
        data["v_pages"],
        data["page_indices"],
        data["full_mask"],
        data["scale"],
    )
    paged = poc.paged_attention_online_softmax(
        data["query"],
        data["k_pages"],
        data["v_pages"],
        data["page_indices"],
        data["mask_tiles"],
        data["scale"],
    )
    assert torch.allclose(ref.float(), paged.float(), rtol=1e-2, atol=2e-2)


def test_nontrivial_mask_changes_output():
    """A masked-out page must change the attention output (mask is wired in)."""
    data = poc._build_inputs(device="cpu", dtype=torch.float32)
    unmasked = poc.paged_attention_online_softmax(
        data["query"],
        data["k_pages"],
        data["v_pages"],
        data["page_indices"],
        data["mask_tiles"],
        data["scale"],
    )
    masked_tiles = [m.clone() for m in data["mask_tiles"]]
    masked_tiles[-1] = masked_tiles[-1] + float("-inf")  # drop the last page
    masked = poc.paged_attention_online_softmax(
        data["query"],
        data["k_pages"],
        data["v_pages"],
        data["page_indices"],
        masked_tiles,
        data["scale"],
    )
    assert torch.isfinite(masked).all()
    assert not torch.allclose(unmasked, masked)


def _run_all():
    tests = [
        test_block_table_gather_matches_indexing,
        test_online_softmax_matches_dense_fp32,
        test_online_softmax_matches_dense_fp16,
        test_nontrivial_mask_changes_output,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
