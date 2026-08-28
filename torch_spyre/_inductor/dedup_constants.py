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

from collections import defaultdict

import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation
from torch._inductor.virtualized import V

from .ir import SpyreConstantFallback
from .logging_utils import get_inductor_logger
from .pass_utils import NameSwapHandler
from .provenance import merge_provenance

logger = get_inductor_logger("dedup_constants")


def _constant_key(op: SpyreConstantFallback) -> tuple:
    """Normalised (value, dtype, device) identity key for a SpyreConstantFallback."""
    layout = op.layout
    dev = layout.device
    norm_dev = (
        torch.device(dev.type, dev.index)
        if dev.index is not None
        else torch.device(dev.type)
    )
    return (op.constant_args[0], layout.dtype, norm_dev)


def _patch_inner_fn(consumer: ComputedBuffer, name_map: dict[str, str]) -> None:
    """Wrap consumer's inner_fn to redirect duplicate constant reads to the canonical name."""
    orig_inner = consumer.data.inner_fn

    def _new_inner(*args, _map=name_map, _orig=orig_inner):
        with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
            return _orig(*args)

    object.__setattr__(consumer.data, "inner_fn", _new_inner)
    ComputedBuffer.get_default_sizes_body.clear_cache(consumer)


def _build_reverse_consumer_index(
    operations: list[Operation],
    duplicate_names: set[str],
) -> dict[str, list[Operation]]:
    """Build a name -> [Operations that read this name] index for the given
    duplicate buffer names.

    Runs one ``op.get_read_writes()`` per op in ``operations`` -- the same
    call the pristine algorithm makes inside its per-duplicate scan -- and
    records each match once per (op, buffer name) pair. If a single op's
    reads contain two distinct dependency objects with the same
    ``.name``, that op still appears at most once in
    ``consumers_by_name[name]``. This exactly preserves the pristine
    algorithm's behavior, which patches an op at most once per duplicate.
    """
    idx: dict[str, list[Operation]] = defaultdict(list)
    for op in operations:
        matched_names: set[str] = set()
        for dep in op.get_read_writes().reads:
            if dep.name in duplicate_names:
                matched_names.add(dep.name)
        for name in matched_names:
            idx[name].append(op)
    return idx


def _redirect_consumers(
    consumers: list[Operation],
    dup: SpyreConstantFallback,
    canonical: SpyreConstantFallback,
) -> None:
    """Rewrite every ComputedBuffer consumer of dup to read canonical instead.

    ``consumers`` is ``consumers_by_name[dup.get_name()]`` -- precomputed by
    ``_build_reverse_consumer_index``, so this function does not call
    ``get_read_writes`` and does not scan ``graph.operations``. All other
    semantics (output-name skip, dup/canonical identity skip,
    non-ComputedBuffer AssertionError) are unchanged.
    """
    D = dup.get_name()
    C = canonical.get_name()
    name_map = {D: C}

    # Do not dedup a constant that is itself a graph output.
    if D in V.graph.get_output_names():
        logger.debug("dedup_and_promote_constants: skipping output constant %s", D)
        return

    for op in consumers:
        if op is dup or op is canonical:
            continue
        if isinstance(op, ComputedBuffer):
            _patch_inner_fn(op, name_map)
        else:
            raise AssertionError(
                f"dedup_and_promote_constants: unsupported consumer type "
                f"{type(op).__name__} reads constant {D!r} — cannot rewrite"
            )


def _drop_constant(
    operations: list[Operation],
    dup: SpyreConstantFallback,
    canonical: SpyreConstantFallback,
) -> None:
    """Remove a duplicate constant from the graph and update bookkeeping."""
    D = dup.get_name()
    C = canonical.get_name()
    op_name = dup.get_operation_name()
    merge_provenance(
        [canonical, dup],
        canonical,
        pass_name="dedup_and_promote_constants",
        reason="duplicate constant",
    )
    operations.remove(dup)
    V.graph.removed_buffers.add(D)
    V.graph.name_to_buffer.pop(D, None)
    V.graph.name_to_op.pop(op_name, None)
    # Merge the duplicate's users into the canonical's user list so that passes
    # which iterate name_to_users (e.g. scratchpad planning) see the full set.
    extra_users = V.graph.name_to_users.pop(D, [])
    if extra_users:
        V.graph.name_to_users.setdefault(C, []).extend(extra_users)
    logger.debug("dedup_and_promote_constants: merged %s into canonical %s", D, C)


def dedup_and_promote_constants(graph: GraphLowering) -> None:
    """Deduplicate SpyreConstantFallback ops and move them to the head of operations.

    Steps:
      1. Group SpyreConstantFallback ops by (value, dtype, device).
      2. If any group has >1 instance, build a live reverse consumer index
         once by scanning graph.operations and inspecting each op's live
         get_read_writes. For each duplicate, rewrite its ComputedBuffer
         consumers via the precomputed list to read from canonical, then
         drop the duplicate using the removed_buffers convention.
      3. Move all surviving SpyreConstantFallback ops to the front of
         operations, preserving relative order.

    Mutates operations in place.

    Complexity note. Before: each duplicate triggered an O(N) scan of
    graph.operations, invoking op.get_read_writes() on every op. That
    dominated the pass time on graphs where duplicate count grows with
    graph size (near-quadratic in program size for the affected
    workload). This implementation performs the same get_read_writes
    calls at most once per op, in a single sweep, and consults the
    precomputed reverse index per duplicate. The remaining
    ``operations.remove(dup)`` in ``_drop_constant`` still runs once per
    duplicate; its measured cost is negligible relative to the
    consumer-discovery term and is intentionally not batched in this
    change.
    """
    operations = graph.operations

    # --- Step 1: group by identity key ---
    groups: dict[tuple, list[SpyreConstantFallback]] = {}
    for op in operations:
        if not isinstance(op, SpyreConstantFallback):
            continue
        key = _constant_key(op)
        groups.setdefault(key, []).append(op)

    # Determine the set of duplicate names up front. When there are no
    # duplicates, skip the reverse-index scan and go straight to
    # front-loading; the pristine pass never called get_read_writes in
    # that case either, and it is worth preserving.
    duplicate_names: set[str] = set()
    for group in groups.values():
        if len(group) > 1:
            for dup in group[1:]:
                duplicate_names.add(dup.get_name())

    # --- Step 2: dedup, only when duplicates exist ---
    if duplicate_names:
        consumers_by_name = _build_reverse_consumer_index(operations, duplicate_names)
        for key, group in groups.items():
            if len(group) <= 1:
                continue
            canonical = group[0]
            for dup in group[1:]:
                _redirect_consumers(
                    consumers_by_name.get(dup.get_name(), []),
                    dup,
                    canonical,
                )
                _drop_constant(operations, dup, canonical)

    # --- Step 3: front-load surviving constants ---
    constants = [op for op in operations if isinstance(op, SpyreConstantFallback)]
    if not constants:
        return
    non_constants = [
        op for op in operations if not isinstance(op, SpyreConstantFallback)
    ]
    operations[:] = constants + non_constants
    logger.debug(
        "dedup_and_promote_constants: %d constant(s) promoted to front of operations",
        len(constants),
    )
