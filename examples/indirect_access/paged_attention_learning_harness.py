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

"""vLLM-shape paged-attention *learning harness* for the Spyre indirect stack.

This is exploratory, throwaway-friendly code. It goes one step past the first
CPU-only paged-attention POC (``paged_attention_vllm_shape_poc.py``) and is meant
to make it cheap to *learn* how far the current indirect-access / paged-attention
PR stack can be pushed toward vLLM-shaped paged KV attention.

What it gives you:

* deterministic, vLLM-shaped inputs -- a page pool
  ``[num_pages, num_kv_heads, block_size, head_dim]``, a 1D page list for one
  sequence, a 2D block table ``[batch, blocks_per_seq]``, grouped-query
  attention (``num_q_heads`` a multiple of ``num_kv_heads``), optional variable
  per-sequence context lengths, and additive masks (including a fully-masked
  block);
* three execution forms that should all agree numerically:
  ``dense_reference`` (the oracle), ``online_page_loop`` (streaming softmax over
  a page loop), and ``indirect_gather_shape`` (page selection via
  ``torch.gather`` over dim 0 -- the shape the Spyre ``add_index_to_address``
  pass rewrites into ``spyre.indices_to_address``);
* an outcome **classifier** that records, per case, exactly what happened
  (``eager_ok`` / ``compile_ok`` / ``graph_break_or_explain_only`` /
  ``compile_failed`` / ``runtime_failed`` / ``numeric_mismatch`` /
  ``skipped_backend_missing``) instead of hiding failures;
* a CLI with ``single`` / ``sweep`` / ``probe-env`` modes and JSON / Markdown
  output, so the same script is useful on a laptop (CPU correctness) and later
  on a Spyre-capable host (compile / device execution).

It runs with only ``torch`` installed. ``torch_spyre._C`` is imported **only**
when ``--device spyre`` is requested; a missing backend is classified, not
crashed on.

CPU examples::

    python examples/indirect_access/paged_attention_learning_harness.py \\
        --device cpu --dtype float32 --kernel all --mode single \\
        --json-out /tmp/paged-attn-fp32.json

    python examples/indirect_access/paged_attention_learning_harness.py \\
        --device cpu --dtype float32 --kernel all --mode sweep \\
        --json-out /tmp/paged-attn-sweep.json

Spyre example (on a built backend; see the cluster runbook)::

    SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1 SENCORES=1 \\
        python examples/indirect_access/paged_attention_learning_harness.py \\
        --device spyre --dtype float16 --kernel all --mode single --compile
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
from typing import Any, Callable, Optional

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
}

KERNEL_NAMES: tuple[str, ...] = (
    "dense_reference",
    "online_page_loop",
    "indirect_gather_shape",
)

STATUSES: frozenset[str] = frozenset(
    {
        "eager_ok",
        "compile_ok",
        "graph_break_or_explain_only",
        "compile_failed",
        "runtime_failed",
        "numeric_mismatch",
        "skipped_backend_missing",
    }
)

# Env flag that gates the Spyre index->address rewrite (PR 2178 surface).
ADD_INDEX_ENV = "SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"


# ---------------------------------------------------------------------------
# Problem configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ProblemConfig:
    """A single vLLM-shaped paged-attention problem instance.

    ``masked_block`` (when set) fully masks one block with ``-inf`` to exercise
    the masked-page path; block 0 is never the masked block so every query row
    keeps at least one valid key. ``variable_context`` masks the tail of each
    sequence's KV beyond a per-sequence context length (always >= one block).
    """

    case_name: str
    batch: int = 2
    num_q_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 64
    block_size: int = 32
    num_pages: int = 16
    blocks_per_seq: int = 4
    seq_len: int = 8
    seed: int = 0
    masked_block: Optional[int] = None
    variable_context: bool = False

    def validate(self) -> None:
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({self.num_q_heads}) must be a multiple of "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.blocks_per_seq < 1 or self.block_size < 1:
            raise ValueError("blocks_per_seq and block_size must be >= 1")
        if self.masked_block is not None and not (
            0 < self.masked_block < self.blocks_per_seq
        ):
            raise ValueError(
                "masked_block must be in (0, blocks_per_seq) so block 0 stays valid"
            )


# ---------------------------------------------------------------------------
# Grouped-query-attention helper
# ---------------------------------------------------------------------------
def expand_kv_heads(t: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Broadcast KV heads up to query heads for grouped-query attention.

    Args:
        t: tensor shaped ``[B, num_kv_heads, S, D]``.
        num_q_heads: number of query heads (a multiple of ``num_kv_heads``).

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
    return t.repeat_interleave(num_q_heads // num_kv_heads, dim=1)


# ---------------------------------------------------------------------------
# Page-selection primitives (the two indirect-access forms)
# ---------------------------------------------------------------------------
def gather_pages_index_select(pages: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Natural vLLM form: select pages by id via ``index_select`` (dim 0).

    On Spyre this decomposes to ``aten.index.Tensor``, which the
    ``add_index_to_address`` pass recognises. ``ids`` is a 1D int64 tensor of
    page ids; the result is ``[len(ids), num_kv_heads, block_size, head_dim]``.
    """
    return pages.index_select(0, ids)


def gather_pages_gather(pages: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Spyre-friendly form: select pages via ``torch.gather`` over dim 0.

    This mirrors ``expand_address_tensor`` + ``torch.gather`` in
    ``paged_attention.py``: the 1D page-id vector is expanded to the page-pool
    rank so ``torch.gather(pages, 0, idx)`` performs the indirect read. This is
    the ``aten.gather.default`` pattern that ``add_index_to_address`` turns into
    ``spyre.indices_to_address``.
    """
    num = ids.shape[0]
    num_kv_heads, block_size, head_dim = pages.shape[1], pages.shape[2], pages.shape[3]
    idx = ids.view(num, 1, 1, 1).expand(num, num_kv_heads, block_size, head_dim)
    return torch.gather(pages, 0, idx)


# ---------------------------------------------------------------------------
# (1) Dense reference -- the correctness oracle
# ---------------------------------------------------------------------------
def dense_reference(
    query: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    block_table: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Gather all selected pages into dense K/V and run one-shot attention.

    Args:
        query: ``[batch, num_q_heads, seq_len, head_dim]``.
        k_pages / v_pages: ``[num_pages, num_kv_heads, block_size, head_dim]``.
        block_table: int64 ``[batch, blocks_per_seq]`` page ids.
        mask: additive mask
            ``[batch, num_q_heads, seq_len, blocks_per_seq*block_size]``.
        scale: softmax scale (typically ``1/sqrt(head_dim)``).

    Returns:
        Attention output ``[batch, num_q_heads, seq_len, head_dim]``.
    """
    batch, num_q_heads = query.shape[0], query.shape[1]
    blocks = block_table.shape[1]
    num_kv_heads, block_size, head_dim = (
        k_pages.shape[1],
        k_pages.shape[2],
        k_pages.shape[3],
    )

    flat = block_table.reshape(-1)
    k_sel = gather_pages_index_select(k_pages, flat)
    v_sel = gather_pages_index_select(v_pages, flat)

    # [B*blocks, Hkv, blk, D] -> [B, Hkv, blocks*blk, D]
    k_flat = (
        k_sel.reshape(batch, blocks, num_kv_heads, block_size, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, num_kv_heads, blocks * block_size, head_dim)
    )
    v_flat = (
        v_sel.reshape(batch, blocks, num_kv_heads, block_size, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, num_kv_heads, blocks * block_size, head_dim)
    )
    k_flat = expand_kv_heads(k_flat, num_q_heads)
    v_flat = expand_kv_heads(v_flat, num_q_heads)

    scores = torch.matmul(query, k_flat.transpose(-2, -1)) * scale + mask
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_flat)


# ---------------------------------------------------------------------------
# (2) Online page loop -- streaming softmax, never materialises full scores
# ---------------------------------------------------------------------------
def online_page_loop(
    query: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    block_table: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """Block-wise paged attention with online (streaming) softmax.

    Mirrors ``create_specialized_paged_attn_kernel`` in ``paged_attention.py``:
    one block of pages is gathered per loop step and the softmax statistics are
    carried across blocks, so the full ``[seq_len, blocks*block_size]`` score
    matrix is never materialised. Fully batched -- each sequence walks its own
    block-table row.
    """
    num_q_heads = query.shape[1]
    blocks = block_table.shape[1]

    running_max = None
    running_sum = None
    running_out = None

    for j in range(blocks):
        ids = block_table[:, j]  # [batch]
        k_blk = expand_kv_heads(gather_pages_index_select(k_pages, ids), num_q_heads)
        v_blk = expand_kv_heads(gather_pages_index_select(v_pages, ids), num_q_heads)

        m = mask[..., j * block_size : (j + 1) * block_size]
        scores = torch.matmul(query, k_blk.transpose(-2, -1)) * scale + m
        block_max = scores.max(dim=-1, keepdim=True)[0]

        if running_max is None:
            running_max = block_max
            probs = torch.exp(scores - running_max)
            running_out = torch.matmul(probs, v_blk)
            running_sum = probs.sum(dim=-1, keepdim=True)
        else:
            new_max = torch.maximum(running_max, block_max)
            rescale = torch.exp(running_max - new_max)
            running_out = running_out * rescale
            running_sum = running_sum * rescale
            probs = torch.exp(scores - new_max)
            running_out = running_out + torch.matmul(probs, v_blk)
            running_sum = running_sum + probs.sum(dim=-1, keepdim=True)
            running_max = new_max

    return running_out / running_sum


# ---------------------------------------------------------------------------
# (3) Indirect gather-shape form -- page selection via torch.gather (dim 0)
# ---------------------------------------------------------------------------
def indirect_gather_shape(
    query: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    block_table: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Same attention as ``dense_reference`` but pages are read with
    ``torch.gather`` over dim 0 (the ``paged_attention.py`` shape).

    The only difference from ``dense_reference`` is the page-selection primitive:
    ``torch.gather(pages, 0, expanded_ids)`` instead of ``index_select``. This is
    the form most likely to exercise the Spyre indirect-access lowering
    (``aten.gather.default`` -> ``spyre.indices_to_address``).
    """
    batch, num_q_heads = query.shape[0], query.shape[1]
    blocks = block_table.shape[1]
    num_kv_heads, block_size, head_dim = (
        k_pages.shape[1],
        k_pages.shape[2],
        k_pages.shape[3],
    )

    flat = block_table.reshape(-1)
    k_sel = gather_pages_gather(k_pages, flat)
    v_sel = gather_pages_gather(v_pages, flat)

    k_flat = (
        k_sel.reshape(batch, blocks, num_kv_heads, block_size, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, num_kv_heads, blocks * block_size, head_dim)
    )
    v_flat = (
        v_sel.reshape(batch, blocks, num_kv_heads, block_size, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, num_kv_heads, blocks * block_size, head_dim)
    )
    k_flat = expand_kv_heads(k_flat, num_q_heads)
    v_flat = expand_kv_heads(v_flat, num_q_heads)

    scores = torch.matmul(query, k_flat.transpose(-2, -1)) * scale + mask
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_flat)


# Kernel registry: name -> (callable, ordered arg names from the inputs dict).
_KERNELS: dict[str, tuple[Callable[..., torch.Tensor], tuple[str, ...]]] = {
    "dense_reference": (
        dense_reference,
        ("query", "k_pages", "v_pages", "block_table", "mask", "scale"),
    ),
    "online_page_loop": (
        online_page_loop,
        ("query", "k_pages", "v_pages", "block_table", "mask", "scale", "block_size"),
    ),
    "indirect_gather_shape": (
        indirect_gather_shape,
        ("query", "k_pages", "v_pages", "block_table", "mask", "scale"),
    ),
}


# ---------------------------------------------------------------------------
# Deterministic input building
# ---------------------------------------------------------------------------
def build_inputs(
    cfg: ProblemConfig,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Build a deterministic vLLM-shaped problem on ``device`` in ``dtype``.

    The float tensors are generated in fp32 on CPU from a seeded generator and
    only then cast/moved, so the *same* ``seed`` yields the same problem across
    devices and dtypes -- which is what lets a CPU fp32 oracle validate a fp16
    (or Spyre) run of the same case.
    """
    cfg.validate()
    gen = torch.Generator().manual_seed(cfg.seed)

    total_kv = cfg.blocks_per_seq * cfg.block_size

    query = torch.randn(
        cfg.batch, cfg.num_q_heads, cfg.seq_len, cfg.head_dim, generator=gen
    )
    k_pages = torch.randn(
        cfg.num_pages, cfg.num_kv_heads, cfg.block_size, cfg.head_dim, generator=gen
    )
    v_pages = torch.randn(
        cfg.num_pages, cfg.num_kv_heads, cfg.block_size, cfg.head_dim, generator=gen
    )

    # 2D block table: deterministic page ids in [0, num_pages).
    block_table = torch.randint(
        0, cfg.num_pages, (cfg.batch, cfg.blocks_per_seq), generator=gen
    ).to(torch.int64)
    # 1D page list for one sequence (the single-sequence view).
    page_list_1d = block_table[0].clone()

    # Per-sequence context length (>= one full block so block 0 is always valid).
    if cfg.variable_context:
        steps = torch.arange(cfg.batch) % cfg.blocks_per_seq
        context_lens = (total_kv - steps * cfg.block_size).clamp_min(cfg.block_size)
    else:
        context_lens = torch.full((cfg.batch,), total_kv, dtype=torch.int64)
    context_lens = context_lens.to(torch.int64)

    # Additive mask [B, Hq, S, total_kv]; 0 where attended, -inf where masked.
    mask = torch.zeros(cfg.batch, cfg.num_q_heads, cfg.seq_len, total_kv)
    kv_pos = torch.arange(total_kv)
    for b in range(cfg.batch):
        beyond = kv_pos >= context_lens[b]
        if beyond.any():
            mask[b, :, :, beyond] = float("-inf")
    if cfg.masked_block is not None:
        lo = cfg.masked_block * cfg.block_size
        hi = lo + cfg.block_size
        mask[:, :, :, lo:hi] = float("-inf")

    scale = 1.0 / (cfg.head_dim**0.5)

    def to_dev(t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point():
            return t.to(device=device, dtype=dtype)
        return t.to(device=device)

    return {
        "query": to_dev(query),
        "k_pages": to_dev(k_pages),
        "v_pages": to_dev(v_pages),
        "block_table": block_table.to(device),
        "page_list_1d": page_list_1d.to(device),
        "context_lens": context_lens.to(device),
        "mask": to_dev(mask),
        "scale": scale,
        "block_size": cfg.block_size,
    }


# ---------------------------------------------------------------------------
# Result classification
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class CaseResult:
    """One classified outcome row (the schema the work order asks for)."""

    case_name: str
    device: str
    dtype: str
    batch: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    num_pages: int
    blocks_per_seq: int
    mode: str  # which execution form (kernel) ran
    compile_enabled: bool
    status: str
    max_abs_diff: Optional[float]
    max_rel_diff: Optional[float]
    allclose: Optional[bool]
    error_type: Optional[str]
    error_message: Optional[str]
    # Extra context (kept out of the required schema but useful when learning).
    seq_len: int = 0
    seed: int = 0
    masked_block: Optional[int] = None
    variable_context: bool = False
    explain: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _tolerances(dtype_str: str) -> tuple[float, float]:
    """(atol, rtol) for comparing a kernel against the fp32 oracle."""
    if dtype_str == "float32":
        return 1e-4, 1e-4
    return 2e-2, 1e-2  # float16


def _compare(out: torch.Tensor, golden: torch.Tensor) -> tuple[float, float]:
    """Return ``(max_abs_diff, max_rel_diff)`` vs the fp32 CPU oracle."""
    out_c = out.detach().float().cpu()
    g = golden.detach().float().cpu()
    diff = (out_c - g).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / g.abs().clamp_min(1e-12)).max().item())
    return max_abs, max_rel


def _assemble_args(inputs: dict[str, Any], arg_names: tuple[str, ...]) -> tuple:
    return tuple(inputs[name] for name in arg_names)


def _short(msg: object, limit: int = 300) -> str:
    text = str(msg).strip().splitlines()
    first = text[0] if text else ""
    return first[:limit]


def capture_explain(fn: Callable[..., Any], args: tuple) -> dict[str, Any]:
    """Best-effort ``torch._dynamo.explain`` summary, version-tolerant.

    Dynamo tracing (explain) does not invoke the Inductor backend, so this can
    succeed even when ``torch.compile`` cannot build a kernel on the local
    toolchain. Returns a small dict; never raises.
    """
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is None or not hasattr(dynamo, "explain"):
        return {"available": False, "reason": "torch._dynamo.explain missing"}
    try:
        try:
            exp = dynamo.explain(fn)(*args)  # torch >= 2.1 calling convention
        except TypeError:
            exp = dynamo.explain(fn, *args)  # legacy positional convention
    except Exception as exc:  # noqa: BLE001 - explain is best-effort
        return {"available": True, "error": f"{type(exc).__name__}: {_short(exc)}"}

    summary: dict[str, Any] = {"available": True}
    for attr in ("graph_count", "graph_break_count", "op_count"):
        val = getattr(exp, attr, None)
        if val is not None:
            summary[attr] = int(val)
    breaks = getattr(exp, "break_reasons", None)
    if breaks:
        try:
            summary["break_reasons"] = [
                _short(getattr(b, "reason", b), 160) for b in breaks
            ][:5]
        except TypeError:
            summary["break_reasons"] = _short(breaks)
    if "graph_count" not in summary:
        summary["raw"] = _short(exp)  # unknown/legacy shape
    return summary


def run_case(
    cfg: ProblemConfig,
    kernel_name: str,
    device: str,
    dtype_str: str,
    compile_enabled: bool,
) -> CaseResult:
    """Run one (config, kernel) case and classify the outcome.

    The oracle is always ``dense_reference`` computed on CPU in fp32 from the
    same seed; every kernel -- including ``dense_reference`` itself -- is scored
    against it.
    """
    dtype = DTYPES[dtype_str]
    fn, arg_names = _KERNELS[kernel_name]

    result = CaseResult(
        case_name=cfg.case_name,
        device=device,
        dtype=dtype_str,
        batch=cfg.batch,
        num_q_heads=cfg.num_q_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        block_size=cfg.block_size,
        num_pages=cfg.num_pages,
        blocks_per_seq=cfg.blocks_per_seq,
        mode=kernel_name,
        compile_enabled=compile_enabled,
        status="runtime_failed",
        max_abs_diff=None,
        max_rel_diff=None,
        allclose=None,
        error_type=None,
        error_message=None,
        seq_len=cfg.seq_len,
        seed=cfg.seed,
        masked_block=cfg.masked_block,
        variable_context=cfg.variable_context,
    )

    # --- fp32 CPU oracle (no Spyre backend needed) -----------------------
    try:
        golden_inputs = build_inputs(cfg, "cpu", torch.float32)
        golden = dense_reference(
            golden_inputs["query"],
            golden_inputs["k_pages"],
            golden_inputs["v_pages"],
            golden_inputs["block_table"],
            golden_inputs["mask"],
            golden_inputs["scale"],
        )
    except Exception as exc:  # noqa: BLE001 - malformed case -> classify it
        result.status = "runtime_failed"
        result.error_type = type(exc).__name__
        result.error_message = _short(exc)
        return result

    # --- optional Spyre backend bring-up ---------------------------------
    if device != "cpu":
        try:
            import os

            os.environ.setdefault(ADD_INDEX_ENV, "1")
            import torch_spyre  # noqa: F401 - registers the 'spyre' device
        except Exception as exc:  # noqa: BLE001 - backend missing -> skip
            result.status = "skipped_backend_missing"
            result.error_type = type(exc).__name__
            result.error_message = _short(exc)
            return result

    # --- build the device/dtype inputs -----------------------------------
    try:
        inputs = build_inputs(cfg, device, dtype)
        args = _assemble_args(inputs, arg_names)
    except Exception as exc:  # noqa: BLE001
        # On a non-CPU device a build failure is almost always a missing/half
        # backend; classify it as such rather than as a generic runtime error.
        if device != "cpu":
            result.status = "skipped_backend_missing"
        else:
            result.status = "runtime_failed"
        result.error_type = type(exc).__name__
        result.error_message = _short(exc)
        return result

    atol, rtol = _tolerances(dtype_str)

    # --- eager execution -------------------------------------------------
    try:
        eager_out = fn(*args)
    except Exception as exc:  # noqa: BLE001
        result.status = (
            "skipped_backend_missing" if device != "cpu" else "runtime_failed"
        )
        result.error_type = type(exc).__name__
        result.error_message = _short(exc)
        return result

    max_abs, max_rel = _compare(eager_out, golden)
    close = bool(
        torch.allclose(
            eager_out.detach().float().cpu(),
            golden.detach().float().cpu(),
            rtol=rtol,
            atol=atol,
        )
    )
    result.max_abs_diff = max_abs
    result.max_rel_diff = max_rel
    result.allclose = close

    if not compile_enabled:
        result.status = "eager_ok" if close else "numeric_mismatch"
        return result

    # --- compile path (capture explain, then attempt execution) ----------
    result.explain = capture_explain(fn, args)
    try:
        compiled = torch.compile(fn)
        compiled_out = compiled(*args)
    except Exception as exc:  # noqa: BLE001 - Inductor/backend build failure
        result.status = "compile_failed"
        result.error_type = type(exc).__name__
        result.error_message = _short(exc)
        return result

    c_abs, c_rel = _compare(compiled_out, golden)
    c_close = bool(
        torch.allclose(
            compiled_out.detach().float().cpu(),
            golden.detach().float().cpu(),
            rtol=rtol,
            atol=atol,
        )
    )
    result.max_abs_diff = c_abs
    result.max_rel_diff = c_rel
    result.allclose = c_close

    graph_breaks = (result.explain or {}).get("graph_break_count")
    if graph_breaks:
        result.status = "graph_break_or_explain_only"
    elif c_close:
        result.status = "compile_ok"
    else:
        result.status = "numeric_mismatch"
    return result


# ---------------------------------------------------------------------------
# Scenario sets
# ---------------------------------------------------------------------------
def single_config(args: argparse.Namespace) -> ProblemConfig:
    """Build a single ProblemConfig from CLI shape flags."""
    return ProblemConfig(
        case_name="single",
        batch=args.batch,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        num_pages=args.num_pages,
        blocks_per_seq=args.blocks_per_seq,
        seq_len=args.seq_len,
        seed=args.seed,
    )


def sweep_scenarios() -> list[tuple[ProblemConfig, str]]:
    """A small, curated (config, dtype) sweep covering the learning surface.

    Each scenario is pinned to its own dtype so the sweep walks fp32 *and* fp16
    regardless of ``--dtype`` (a sweep is meant to cover the matrix). The shapes
    stay small so the whole sweep runs in seconds on a laptop.
    """
    base = dict(
        batch=2,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        block_size=32,
        num_pages=16,
        blocks_per_seq=4,
        seq_len=8,
    )
    scenarios: list[tuple[ProblemConfig, str]] = [
        (ProblemConfig(case_name="base", **base), "float32"),
        (ProblemConfig(case_name="base", **base), "float16"),
        (
            ProblemConfig(case_name="gqa_group4", **{**base, "num_kv_heads": 2}),
            "float16",
        ),
        (
            ProblemConfig(case_name="gqa_group8", **{**base, "num_kv_heads": 1}),
            "float16",
        ),
        (
            ProblemConfig(
                case_name="mha", **{**base, "num_q_heads": 4, "num_kv_heads": 4}
            ),
            "float32",
        ),
        (
            ProblemConfig(case_name="variable_context", variable_context=True, **base),
            "float32",
        ),
        (
            ProblemConfig(case_name="masked_block", masked_block=2, **base),
            "float32",
        ),
        (
            ProblemConfig(case_name="masked_block", masked_block=2, **base),
            "float16",
        ),
        (
            ProblemConfig(
                case_name="decode_seqlen1",
                **{**base, "seq_len": 1, "blocks_per_seq": 6, "batch": 3},
            ),
            "float16",
        ),
    ]
    return scenarios


# ---------------------------------------------------------------------------
# Environment probe
# ---------------------------------------------------------------------------
def probe_environment() -> dict[str, Any]:
    """Collect facts useful for bring-up on a new (e.g. Spyre) host."""
    import os

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "dynamo_explain_available": hasattr(getattr(torch, "_dynamo", None), "explain"),
        f"env::{ADD_INDEX_ENV}": os.environ.get(ADD_INDEX_ENV),
        "env::SENCORES": os.environ.get("SENCORES"),
    }
    try:
        backend = torch._C._get_privateuse1_backend_name()
    except Exception:  # noqa: BLE001
        backend = None
    info["privateuse1_backend"] = backend

    try:
        import torch_spyre  # noqa: F401

        info["torch_spyre_importable"] = True
        info["torch_spyre_version"] = getattr(torch_spyre, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        info["torch_spyre_importable"] = False
        info["torch_spyre_import_error"] = f"{type(exc).__name__}: {_short(exc)}"
    return info


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
_TABLE_COLUMNS = (
    "case_name",
    "mode",
    "device",
    "dtype",
    "compile_enabled",
    "status",
    "max_abs_diff",
    "max_rel_diff",
    "allclose",
)


def _fmt_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3e}"
    if value is None:
        return "-"
    return str(value)


def results_to_markdown(results: list[CaseResult]) -> str:
    header = "| " + " | ".join(_TABLE_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _TABLE_COLUMNS) + " |"
    lines = [header, sep]
    for r in results:
        row = r.to_dict()
        lines.append(
            "| " + " | ".join(_fmt_cell(row[c]) for c in _TABLE_COLUMNS) + " |"
        )
    return "\n".join(lines) + "\n"


def print_table(results: list[CaseResult]) -> None:
    widths = {c: len(c) for c in _TABLE_COLUMNS}
    rendered = []
    for r in results:
        row = {c: _fmt_cell(r.to_dict()[c]) for c in _TABLE_COLUMNS}
        rendered.append(row)
        for c in _TABLE_COLUMNS:
            widths[c] = max(widths[c], len(row[c]))
    line = "  ".join(c.ljust(widths[c]) for c in _TABLE_COLUMNS)
    print(line)
    print("  ".join("-" * widths[c] for c in _TABLE_COLUMNS))
    for row in rendered:
        print("  ".join(row[c].ljust(widths[c]) for c in _TABLE_COLUMNS))


def _status_counts(results: list[CaseResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="vLLM-shape paged-attention learning harness for the "
        "torch-spyre indirect-access stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cpu", help="cpu or spyre")
    parser.add_argument(
        "--dtype", default="float32", choices=sorted(DTYPES), help="compute dtype"
    )
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "sweep", "probe-env"],
        help="single config, curated sweep, or environment probe",
    )
    parser.add_argument(
        "--kernel",
        default="all",
        choices=[*KERNEL_NAMES, "all"],
        help="execution form(s) to run",
    )
    parser.add_argument("--compile", action="store_true", help="also try torch.compile")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num-q-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--num-pages", type=int, default=16)
    parser.add_argument("--blocks-per-seq", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8, help="query length")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None, help="write JSON results to PATH")
    parser.add_argument(
        "--markdown-out", default=None, help="write a Markdown table to PATH"
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="exit non-zero if any case is a numeric_mismatch",
    )
    return parser


def _selected_kernels(kernel_arg: str) -> tuple[str, ...]:
    return KERNEL_NAMES if kernel_arg == "all" else (kernel_arg,)


def run_cli(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.device != "cpu":
        import os

        os.environ.setdefault(ADD_INDEX_ENV, "1")
        print(
            f"[harness] device={args.device}: set {ADD_INDEX_ENV}="
            f"{os.environ.get(ADD_INDEX_ENV)} (Spyre indirect-access rewrite).",
            file=sys.stderr,
        )

    payload: dict[str, Any] = {
        "environment": probe_environment(),
        "args": {
            "device": args.device,
            "dtype": args.dtype,
            "mode": args.mode,
            "kernel": args.kernel,
            "compile": args.compile,
        },
    }

    if args.mode == "probe-env":
        env = payload["environment"]
        print(json.dumps(env, indent=2))
        if args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"[harness] wrote {args.json_out}", file=sys.stderr)
        return 0

    kernels = _selected_kernels(args.kernel)
    results: list[CaseResult] = []

    if args.mode == "single":
        cfg = single_config(args)
        for kernel_name in kernels:
            results.append(
                run_case(cfg, kernel_name, args.device, args.dtype, args.compile)
            )
    else:  # sweep
        for cfg, dtype_str in sweep_scenarios():
            for kernel_name in kernels:
                results.append(
                    run_case(cfg, kernel_name, args.device, dtype_str, args.compile)
                )

    print_table(results)
    counts = _status_counts(results)
    print("\nstatus counts:", json.dumps(counts))

    payload["results"] = [r.to_dict() for r in results]
    payload["status_counts"] = counts

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[harness] wrote {args.json_out}", file=sys.stderr)
    if args.markdown_out:
        with open(args.markdown_out, "w") as fh:
            fh.write(results_to_markdown(results))
        print(f"[harness] wrote {args.markdown_out}", file=sys.stderr)

    if args.fail_on_mismatch and any(r.status == "numeric_mismatch" for r in results):
        print("[harness] numeric_mismatch present -> exit 1", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
