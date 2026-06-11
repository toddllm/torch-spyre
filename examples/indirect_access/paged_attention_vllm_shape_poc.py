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

"""vLLM-shape paged-attention proof of concept (device-parametrized).

This POC answers a single question for the torch-spyre indirect-access stack:
can the public PR 2178 surface express the vLLM-style paged KV attention shape

    query + k_pages + v_pages + page_indices (+ optional mask) -> attention out

It is intentionally CPU-runnable so the *numerics* of a paged / online-softmax
attention kernel can be validated on a laptop, with no Spyre hardware and no
compiled ``torch_spyre._C`` extension. The same kernel structure mirrors
``create_specialized_paged_attn_kernel`` in ``paged_attention.py`` (online
softmax over a page loop), which is the form that the Spyre indirect-access
lowering recognises and turns into gather + matmul on device.

Two things are demonstrated:

1. ``gather_kv_by_block_table`` -- the vLLM "block table" / "page list" is
   consumed directly with ``torch.index_select`` (an indirect access on dim 0
   of the KV page pool). This is the exact primitive the Spyre
   ``detect_indirect_access`` pass keys on (a load whose index is itself a
   loaded value).
2. ``paged_attention_online_softmax`` equals a dense reference
   ``paged_attention_dense_reference`` -- i.e. the block-wise online-softmax
   kernel is numerically correct.

Run on CPU (default, validated locally)::

    python examples/indirect_access/paged_attention_vllm_shape_poc.py

Run on Spyre when hardware + a built backend are available::

    python examples/indirect_access/paged_attention_vllm_shape_poc.py --device spyre
"""

from __future__ import annotations

import argparse

import torch


# ---------------------------------------------------------------------------
# Grouped-query-attention helper
# ---------------------------------------------------------------------------
def expand_kv_heads(t: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Broadcast KV heads up to query heads for grouped-query attention.

    Args:
        t: tensor shaped ``[B, num_kv_heads, S, D]``.
        num_q_heads: number of query heads (multiple of ``num_kv_heads``).

    Returns:
        Tensor shaped ``[B, num_q_heads, S, D]``.
    """
    num_kv_heads = t.shape[1]
    if num_q_heads == num_kv_heads:
        return t
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be a multiple of "
            f"num_kv_heads ({num_kv_heads})"
        )
    group = num_q_heads // num_kv_heads
    return t.repeat_interleave(group, dim=1)


# ---------------------------------------------------------------------------
# (1) Block-table / page-list consumption via indirect access
# ---------------------------------------------------------------------------
def gather_kv_by_block_table(
    pages: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """Select physical KV pages for each sequence using a vLLM block table.

    This is the core "indirect access" primitive: the page-pool tensor is
    indexed by a *tensor of indices* rather than by static offsets. On Spyre
    this lowers through the ``detect_indirect_access`` / index-to-address path
    added by PR 2178; on CPU it is a plain ``index_select`` gather.

    Args:
        pages: page pool shaped ``[num_pages, num_kv_heads, block_size, head_dim]``.
        block_table: int64 tensor shaped ``[num_seqs, blocks_per_seq]`` whose
            entries are page ids into ``pages`` (exactly a vLLM block table).

    Returns:
        Gathered pages shaped
        ``[num_seqs, blocks_per_seq, num_kv_heads, block_size, head_dim]``.
    """
    num_seqs, blocks_per_seq = block_table.shape
    flat = block_table.reshape(-1)
    gathered = pages.index_select(0, flat)
    return gathered.reshape(num_seqs, blocks_per_seq, *pages.shape[1:])


# ---------------------------------------------------------------------------
# (2a) Dense reference attention over the selected pages
# ---------------------------------------------------------------------------
def paged_attention_dense_reference(
    query: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    page_indices: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """One-shot dense attention over the pages named by ``page_indices``.

    Args:
        query: ``[B, num_q_heads, seq_len, head_dim]``.
        k_pages / v_pages: ``[num_pages, num_kv_heads, block_size, head_dim]``.
        page_indices: int64 ``[num_blocks]`` -- the page list for this sequence.
        attn_mask: additive mask ``[B, num_q_heads, seq_len, num_blocks*block_size]``.
        scale: softmax scale (typically ``1/sqrt(head_dim)``).

    Returns:
        Attention output ``[B, num_q_heads, seq_len, head_dim]``.
    """
    num_q_heads = query.shape[1]
    # Gather pages -> [num_blocks, num_kv_heads, block_size, head_dim]
    k_sel = k_pages.index_select(0, page_indices)
    v_sel = v_pages.index_select(0, page_indices)

    num_blocks, num_kv_heads, block_size, head_dim = k_sel.shape
    # -> [1, num_kv_heads, num_blocks*block_size, head_dim]
    k_flat = (
        k_sel.permute(1, 0, 2, 3)
        .reshape(num_kv_heads, num_blocks * block_size, head_dim)
        .unsqueeze(0)
    )
    v_flat = (
        v_sel.permute(1, 0, 2, 3)
        .reshape(num_kv_heads, num_blocks * block_size, head_dim)
        .unsqueeze(0)
    )

    k_flat = expand_kv_heads(k_flat, num_q_heads)
    v_flat = expand_kv_heads(v_flat, num_q_heads)

    scores = torch.matmul(query, k_flat.transpose(-2, -1)) * scale + attn_mask
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_flat)


# ---------------------------------------------------------------------------
# (2b) Paged / online-softmax kernel (the Spyre-shaped form)
# ---------------------------------------------------------------------------
def paged_attention_online_softmax(
    query: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    page_indices: torch.Tensor,
    mask_tiles: list[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    """Block-wise paged attention with online (streaming) softmax.

    This mirrors ``create_specialized_paged_attn_kernel`` in
    ``paged_attention.py``: one page is gathered per loop iteration and the
    softmax statistics are carried across pages. It never materialises the full
    [seq_len, num_blocks*block_size] score matrix, which is what makes it usable
    for long KV caches on a tiled accelerator.

    Args:
        query: ``[B, num_q_heads, seq_len, head_dim]``.
        k_pages / v_pages: ``[num_pages, num_kv_heads, block_size, head_dim]``.
        page_indices: int64 ``[num_blocks]`` page list for this sequence.
        mask_tiles: list of additive masks, one per block, each
            ``[B, num_q_heads, seq_len, block_size]``.
        scale: softmax scale.

    Returns:
        Attention output ``[B, num_q_heads, seq_len, head_dim]``.
    """
    num_q_heads = query.shape[1]
    num_blocks = page_indices.shape[0]

    running_max = None
    running_sum = None
    running_out = None

    for i in range(num_blocks):
        page_id = page_indices[i]
        # Indirect access: select a single page by a *loaded* index.
        k_page = k_pages.index_select(0, page_id.reshape(1))  # [1, Hkv, blk, D]
        v_page = v_pages.index_select(0, page_id.reshape(1))
        k_page = expand_kv_heads(k_page, num_q_heads)
        v_page = expand_kv_heads(v_page, num_q_heads)

        scores = torch.matmul(query, k_page.transpose(-2, -1)) * scale
        scores = scores + mask_tiles[i]
        block_max = scores.max(dim=-1, keepdim=True)[0]

        if running_max is None:
            running_max = block_max
            probs = torch.exp(scores - running_max)
            running_out = torch.matmul(probs, v_page)
            running_sum = probs.sum(dim=-1, keepdim=True)
        else:
            new_max = torch.maximum(running_max, block_max)
            rescale = torch.exp(running_max - new_max)
            running_out = running_out * rescale
            running_sum = running_sum * rescale
            probs = torch.exp(scores - new_max)
            running_out = running_out + torch.matmul(probs, v_page)
            running_sum = running_sum + probs.sum(dim=-1, keepdim=True)
            running_max = new_max

    return running_out / running_sum


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _build_inputs(device: str, dtype: torch.dtype, seed: int = 0):
    """Build a representative vLLM decode-shaped problem."""
    gen = torch.Generator().manual_seed(seed)

    batch_size = 1
    num_q_heads = 8
    num_kv_heads = 2  # grouped-query attention, group size 4
    seq_len = 16
    head_dim = 64
    block_size = 32
    num_pages = 12
    num_blocks = 4  # pages attended to by this sequence

    query = torch.randn(
        batch_size, num_q_heads, seq_len, head_dim, generator=gen, dtype=torch.float32
    )
    k_pages = torch.randn(
        num_pages,
        num_kv_heads,
        block_size,
        head_dim,
        generator=gen,
        dtype=torch.float32,
    )
    v_pages = torch.randn(
        num_pages,
        num_kv_heads,
        block_size,
        head_dim,
        generator=gen,
        dtype=torch.float32,
    )
    # A non-trivial, non-contiguous page list (the block-table row).
    page_indices = torch.tensor([0, 5, 2, 9], dtype=torch.int64)
    assert page_indices.shape[0] == num_blocks

    # Per-block additive mask (here zero, but the plumbing supports causal/pad
    # masks). Use the *same* values for the dense reference and the paged path.
    full_mask = torch.zeros(
        batch_size, num_q_heads, seq_len, num_blocks * block_size, dtype=torch.float32
    )
    mask_tiles = [
        full_mask[..., i * block_size : (i + 1) * block_size] for i in range(num_blocks)
    ]

    scale = 1.0 / (head_dim**0.5)

    def to_dev(t):
        return (
            t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device)
        )

    return {
        "query": to_dev(query),
        "k_pages": to_dev(k_pages),
        "v_pages": to_dev(v_pages),
        "page_indices": page_indices.to(device),
        "full_mask": to_dev(full_mask),
        "mask_tiles": [to_dev(m) for m in mask_tiles],
        "scale": scale,
        "num_pages": num_pages,
        "num_kv_heads": num_kv_heads,
        "block_size": block_size,
        "head_dim": head_dim,
    }


def run_poc(device: str = "cpu", dtype: torch.dtype = torch.float32) -> bool:
    """Run the POC end to end and return ``True`` on success."""
    print("=" * 72)
    print(f"vLLM-shape paged attention POC  |  device={device}  dtype={dtype}")
    print("=" * 72)

    data = _build_inputs(device, dtype)

    # --- (1) block-table / page-list consumption -------------------------
    block_table = torch.tensor(
        [[0, 5, 2, 9], [1, 3, 8, 4]], dtype=torch.int64, device=device
    )
    gathered = gather_kv_by_block_table(data["k_pages"], block_table)
    expected_shape = (
        block_table.shape[0],
        block_table.shape[1],
        data["num_kv_heads"],
        data["block_size"],
        data["head_dim"],
    )
    assert tuple(gathered.shape) == expected_shape, gathered.shape
    # Validate the gather against plain advanced indexing.
    ref_gather = data["k_pages"][block_table]
    gather_ok = torch.equal(gathered, ref_gather)
    print(
        f"[1] block-table gather: pages{tuple(data['k_pages'].shape)} x "
        f"block_table{tuple(block_table.shape)} -> {tuple(gathered.shape)}  "
        f"exact_match={gather_ok}"
    )
    assert gather_ok

    # --- (2) paged online-softmax == dense reference ---------------------
    ref = paged_attention_dense_reference(
        data["query"],
        data["k_pages"],
        data["v_pages"],
        data["page_indices"],
        data["full_mask"],
        data["scale"],
    )
    paged = paged_attention_online_softmax(
        data["query"],
        data["k_pages"],
        data["v_pages"],
        data["page_indices"],
        data["mask_tiles"],
        data["scale"],
    )

    ref_c = ref.float().cpu()
    paged_c = paged.float().cpu()
    assert ref_c.shape == paged_c.shape, (ref_c.shape, paged_c.shape)
    max_diff = (ref_c - paged_c).abs().max().item()

    # fp32 should match tightly; fp16 needs a relaxed bound.
    atol = 1e-4 if dtype == torch.float32 else 2e-2
    rtol = 1e-4 if dtype == torch.float32 else 1e-2
    close = torch.allclose(ref_c, paged_c, rtol=rtol, atol=atol)
    print(
        f"[2] online-softmax vs dense: out{tuple(paged_c.shape)}  "
        f"max_diff={max_diff:.3e}  allclose(atol={atol})={close}"
    )
    assert close, f"paged kernel diverged from dense reference (max_diff={max_diff})"

    print("RESULT: PASS")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device to run on (cpu, or spyre when a built backend exists)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="compute dtype",
    )
    args = parser.parse_args()

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    if args.device != "cpu":
        # Importing torch_spyre registers the 'spyre' PrivateUse1 device.
        import torch_spyre  # noqa: F401

    run_poc(device=args.device, dtype=dtype)


if __name__ == "__main__":
    main()
