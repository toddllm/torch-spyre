# Copyright 2026 The Torch-Spyre Authors.
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

"""Deterministic pass-level tests for dedup_and_promote_constants.

These tests do not depend on Dynamo/Inductor producing a particular
incidental graph shape. Instead they run the real
CustomPreSchedulingPasses pipeline up to (but not including)
dedup_and_promote_constants, snapshot the graph, optionally mutate
it into the exact deterministic condition each test needs, then
call dedup_and_promote_constants directly and assert on the
resulting state.

This hybrid approach was chosen (in preference to hand-built IR
dataclasses) so the tests still exercise real Torch-Spyre / Inductor
constructors, layouts, and provenance mechanisms — i.e. they still
represent pass invariants — while remaining deterministic:

* test_zero_consumer_duplicate — never skips.
* test_one_duplicate_many_consumers — proves consumers_by_name[D]
  fans out correctly when a SINGLE duplicate name D has multiple
  live ComputedBuffer readers.
* test_name_to_users_fold_exact — captures pre-dedup name_to_users
  entries for canonical and every duplicate by object identity,
  asserts post-dedup entries are exactly the concatenation and that
  duplicate keys have been removed.
* test_provenance_transform_appended — asserts merge_provenance
  appended a ProvenanceTransform with pass_name
  "dedup_and_promote_constants" to the canonical.
* test_no_duplicates_fast_path — with the E-only implementation
  we care that when no duplicate group exists we do NOT build the
  reverse index. This test snapshots V.graph state, calls dedup
  with zero duplicates, and asserts the pass exits without
  consuming get_read_writes beyond grouping. Against the pristine
  a9316b3 pass this test passes trivially (the pristine pass never
  called get_read_writes when there are no groups). After the
  E-only refactor, the same assertion must still hold.

Add this file to tests/inductor/ and add its path to
tests/configs/torch_spyre_tests/inductor/ (see
test_dedup_constants_config.yaml for the format).
"""

from typing import Any, Callable, Optional, TypeVarTuple, override

import unittest
from unittest.mock import patch

import torch
from torch._inductor import config as t_inductor_config
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation
from torch._inductor.virtualized import V

from torch_spyre._C import get_elem_in_stick
from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor import passes
from torch_spyre._inductor.dedup_constants import dedup_and_promote_constants
from torch_spyre._inductor.ir import SpyreConstantFallback
from torch_spyre._inductor.passes import CustomPreSchedulingPasses


Ts = TypeVarTuple("Ts")


# ---------------------------------------------------------------------------
# Pipeline hook that stops just before dedup, hands the graph to a test
# callback, then continues (or short-circuits) as the callback wishes.
# ---------------------------------------------------------------------------


class _StopBeforeDedupPasses(CustomPreSchedulingPasses):
    """Runs pre-scheduling passes up to insert_bmm_padding, then hands
    the graph to the test.

    The test callback is stored on the class as ``test_callback``.
    It is invoked with the GraphLowering *inside* V.set_graph_handler,
    which is the same context every pass runs under. The callback
    returns True to continue with the remaining passes (dedup and
    downstream), or False to stop after the callback returns.
    """

    test_callback: Optional[Callable[[GraphLowering], bool]] = None

    @classmethod
    def install(cls, cb: Callable[[GraphLowering], bool]) -> None:
        cls.test_callback = cb

    @override
    def __call__(self, graph: GraphLowering) -> None:
        # Import here so we don't shadow module-level names.
        from torch_spyre._inductor.passes import _operations_have_spyre_device

        if not _operations_have_spyre_device(graph.operations):
            return

        assert self.test_callback is not None, "test_callback not installed"

        # Find the dedup step's index in the pass list; run everything
        # strictly before it.
        pass_list = list(self.passes)
        try:
            dedup_idx = next(
                i
                for i, p in enumerate(pass_list)
                if getattr(p, "__name__", "") == "dedup_and_promote_constants"
            )
        except StopIteration:
            raise AssertionError("dedup_and_promote_constants missing from pipeline")

        # Run everything before dedup — verbatim, no extra observers.
        for pass_fn in pass_list[:dedup_idx]:
            pass_fn(graph)

        # Hand off to the test. Access via type() so Python does not
        # bind self as the first positional arg to a function stored
        # as a class attribute.
        cb = type(self).test_callback
        assert cb is not None
        if not cb(graph):
            return

        # If the callback returned True, continue with dedup and
        # everything after it.
        for pass_fn in pass_list[dedup_idx:]:
            pass_fn(graph)


class _BaseDedupPassTest(unittest.TestCase):
    """Base class installing config patches and the pipeline hook."""

    def setUp(self) -> None:
        torch.manual_seed(0xBEEF)
        self.patchers: list[Any] = []
        self.patchers.append(t_inductor_config.patch("force_disable_caches", True))
        self.patchers.append(ts_inductor_config.patch("sencores", 1))
        self.patchers.append(
            patch.object(passes, "CustomPreSchedulingPasses", _StopBeforeDedupPasses)
        )
        for p in self.patchers:
            p.__enter__()
        torch.compiler.reset()

    def tearDown(self) -> None:
        for p in self.patchers:
            p.__exit__(None, None, None)
        torch.compiler.reset()

    @staticmethod
    def _constants_in(operations: list[Operation]) -> list[SpyreConstantFallback]:
        return [op for op in operations if isinstance(op, SpyreConstantFallback)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDedupConstantsPassLevel(_BaseDedupPassTest):
    """Pass-level tests: run passes up to dedup, then invoke dedup by hand."""

    # ------------------------------------------------------------------
    # test_zero_consumer_duplicate
    # ------------------------------------------------------------------

    def test_zero_consumer_duplicate(self) -> None:
        """A duplicate constant that has no live readers still gets
        cleanly removed and its bookkeeping cleaned.

        Deterministic construction: run the real pipeline up to dedup
        to obtain at least two SpyreConstantFallback ops in one dedup
        group (baseline padded-bmm case). Mark one of them as having
        no live consumer by installing a `V.graph.get_output_names()`
        that includes NOTHING referencing our chosen dup, and — the
        key operation — ARTIFICIALLY DROP any live consumer of the
        chosen duplicate from graph.operations before dedup runs.
        That leaves the dup in the group with zero live readers.

        Asserts:
        - duplicate op removed from operations
        - duplicate buffer name in removed_buffers
        - duplicate buffer name absent from name_to_buffer
        - duplicate operation name absent from name_to_op
        - duplicate buffer name absent from name_to_users
        - canonical survives
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        # Two independent unaligned bmms give two padding constants,
        # both in the same (0.0, fp16, spyre) group. That's our
        # source of a duplicate pair.
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        assertions_ran = {"ok": False}

        def cb(graph: GraphLowering) -> bool:
            constants = self._constants_in(graph.operations)
            self.assertGreaterEqual(
                len(constants),
                2,
                "workload did not produce a duplicate group before "
                "dedup — cannot exercise the zero-consumer case",
            )
            # Group by identity key.
            from torch_spyre._inductor.dedup_constants import _constant_key

            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) > 1]
            self.assertTrue(
                multi,
                "no multi-constant group; cannot test zero-consumer dup",
            )
            group = multi[0]
            canonical, chosen_dup = group[0], group[1]
            D = chosen_dup.get_name()

            # Discover the live consumer of chosen_dup by the same
            # mechanism the pass uses, so the test asserts on the
            # exact ops the pass would have patched.
            live_consumers = [
                op
                for op in graph.operations
                if op is not chosen_dup
                and op is not canonical
                and any(dep.name == D for dep in op.get_read_writes().reads)
            ]
            # Artificially drop them so chosen_dup has zero readers.
            for op in live_consumers:
                graph.operations.remove(op)

            # Snapshot the pre-dedup state that we'll assert against.
            pre_state = {
                "chosen_dup_in_ops": chosen_dup in graph.operations,
                "canonical_in_ops": canonical in graph.operations,
                "chosen_dup_name": D,
                "canonical_name": canonical.get_name(),
                "chosen_dup_op_name": chosen_dup.get_operation_name(),
            }
            self.assertTrue(pre_state["chosen_dup_in_ops"])
            self.assertTrue(pre_state["canonical_in_ops"])

            # Run dedup.
            dedup_and_promote_constants(graph)

            # Assertions.
            self.assertNotIn(
                chosen_dup,
                graph.operations,
                "chosen_dup should be removed from graph.operations",
            )
            self.assertIn(
                canonical,
                graph.operations,
                "canonical should survive",
            )
            self.assertIn(
                D,
                graph.removed_buffers,
                "chosen_dup's buffer name should be in removed_buffers",
            )
            self.assertNotIn(
                D,
                graph.name_to_buffer,
                "chosen_dup's buffer name should be absent from name_to_buffer",
            )
            self.assertNotIn(
                pre_state["chosen_dup_op_name"],
                graph.name_to_op,
                "chosen_dup's operation name should be absent from name_to_op",
            )
            self.assertNotIn(
                D,
                graph.name_to_users,
                "chosen_dup's buffer name should be absent from name_to_users",
            )
            assertions_ran["ok"] = True

            # Stop here — don't run downstream passes; they'd fail
            # because we ripped consumers out of the graph.
            return False

        _StopBeforeDedupPasses.install(cb)

        @torch.compile(fullgraph=True)
        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        # The compile may raise downstream of dedup (we short-circuited
        # by returning False, which means further passes did not run;
        # scheduler still tries to lower the incomplete graph and may
        # error). Swallow anything post-dedup; the assertions ran
        # BEFORE dedup returned, so we know the test either passed or
        # already raised assertion errors.
        try:
            fn(x, w1, w2)
        except Exception:
            # Only re-raise if our assertions didn't get to run.
            if not assertions_ran["ok"]:
                raise
        self.assertTrue(
            assertions_ran["ok"],
            "test callback did not run its assertions",
        )

    # ------------------------------------------------------------------
    # test_one_duplicate_many_consumers
    # ------------------------------------------------------------------

    def test_one_duplicate_many_consumers(self) -> None:
        """A single duplicate name D is read by two or more distinct
        live ComputedBuffers before dedup. Every one of them gets
        redirected to the canonical (patched via NameSwapHandler).

        Deterministic construction: build a padding-constant group of
        two (canonical C and duplicate D) via two unaligned bmms.
        Then re-wire an EXISTING live ComputedBuffer that reads C
        (from bmm #1) to instead read D — this is the exact
        pattern the current pass handles when multiple downstream
        readers happen to reference the same constant. We install
        the extra reader by directly rewriting its inner_fn (the
        same mechanism the pass itself uses via NameSwapHandler).
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        assertions_ran = {"ok": False}

        def cb(graph: GraphLowering) -> bool:
            from torch_spyre._inductor.dedup_constants import _constant_key
            from torch_spyre._inductor.pass_utils import NameSwapHandler

            constants = self._constants_in(graph.operations)
            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) > 1]
            self.assertTrue(multi, "no multi-constant group")
            group = multi[0]
            canonical, dup = group[0], group[1]
            C, D = canonical.get_name(), dup.get_name()

            # Discover the natural reader of `dup` and add a SECOND
            # reader by re-pointing another ComputedBuffer's inner_fn
            # to read D instead of whatever it currently reads. We
            # pick the natural reader of `canonical` and re-wire it
            # by wrapping its inner_fn in a NameSwapHandler({C: D}).
            reader_of_dup = next(
                (
                    op
                    for op in graph.operations
                    if op is not dup
                    and op is not canonical
                    and any(dep.name == D for dep in op.get_read_writes().reads)
                ),
                None,
            )
            reader_of_canonical = next(
                (
                    op
                    for op in graph.operations
                    if op is not dup
                    and op is not canonical
                    and any(dep.name == C for dep in op.get_read_writes().reads)
                ),
                None,
            )
            self.assertIsNotNone(reader_of_dup)
            self.assertIsNotNone(reader_of_canonical)
            self.assertIsInstance(reader_of_canonical, ComputedBuffer)
            self.assertIsNot(reader_of_dup, reader_of_canonical)

            # Re-wire reader_of_canonical to also read D — same
            # mechanism the pass would use, applied in reverse.
            orig_inner = reader_of_canonical.data.inner_fn

            def _new_inner(*args, _map={C: D}, _orig=orig_inner):
                with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
                    return _orig(*args)

            object.__setattr__(reader_of_canonical.data, "inner_fn", _new_inner)
            ComputedBuffer.get_default_sizes_body.clear_cache(reader_of_canonical)

            # Verify: BOTH readers now report D in their live reads.
            reads_dup1 = {dep.name for dep in reader_of_dup.get_read_writes().reads}
            reads_dup2 = {
                dep.name for dep in reader_of_canonical.get_read_writes().reads
            }
            self.assertIn(D, reads_dup1)
            self.assertIn(D, reads_dup2)

            # Run dedup.
            dedup_and_promote_constants(graph)

            # After dedup: both readers should now read C, not D.
            reads_after_1 = {dep.name for dep in reader_of_dup.get_read_writes().reads}
            reads_after_2 = {
                dep.name for dep in reader_of_canonical.get_read_writes().reads
            }
            self.assertNotIn(D, reads_after_1, f"{reader_of_dup} still reads D")
            self.assertNotIn(D, reads_after_2, f"{reader_of_canonical} still reads D")
            self.assertIn(
                C,
                reads_after_1,
                f"{reader_of_dup} should now read canonical {C}",
            )
            self.assertIn(
                C,
                reads_after_2,
                f"{reader_of_canonical} should now read canonical {C}",
            )
            self.assertNotIn(dup, graph.operations)
            self.assertIn(canonical, graph.operations)
            assertions_ran["ok"] = True
            return False

        _StopBeforeDedupPasses.install(cb)

        @torch.compile(fullgraph=True)
        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        try:
            fn(x, w1, w2)
        except Exception:
            if not assertions_ran["ok"]:
                raise
        self.assertTrue(assertions_ran["ok"])

    # ------------------------------------------------------------------
    # test_name_to_users_fold_exact
    # ------------------------------------------------------------------

    def test_name_to_users_fold_exact(self) -> None:
        """name_to_users[canonical] after dedup equals the concatenation
        of pre-dedup name_to_users[canonical] and each duplicate's
        pre-dedup name_to_users entry, compared by object identity.
        Every duplicate name is absent from post-dedup name_to_users.

        This locks in the exact fold behavior of _drop_constant. It
        does NOT claim those entries accurately describe live
        consumers — Phase 2 measured that they do not, on this
        workload — only that the fold behavior is preserved.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        assertions_ran = {"ok": False}

        def cb(graph: GraphLowering) -> bool:
            from torch_spyre._inductor.dedup_constants import _constant_key

            constants = self._constants_in(graph.operations)
            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) > 1]
            self.assertTrue(multi, "no multi-constant group")
            group = multi[0]
            canonical = group[0]
            dups = group[1:]
            C = canonical.get_name()
            dup_names = [d.get_name() for d in dups]

            # Capture pre-dedup entries by object identity.
            pre_C = list(graph.name_to_users.get(C, []))
            pre_D_entries: dict[str, list] = {
                D: list(graph.name_to_users.get(D, [])) for D in dup_names
            }
            expected_C_after = pre_C + [
                entry for D in dup_names for entry in pre_D_entries[D]
            ]

            dedup_and_promote_constants(graph)

            post_C = list(graph.name_to_users.get(C, []))
            # Identity-equal comparison (id-based). The list should be
            # exactly the concatenation.
            self.assertEqual(
                [id(x) for x in post_C],
                [id(x) for x in expected_C_after],
                f"name_to_users[{C!r}] after dedup is not the exact "
                f"identity-preserving concatenation of pre-dedup "
                f"canonical + duplicate entries",
            )
            for D in dup_names:
                self.assertNotIn(
                    D,
                    graph.name_to_users,
                    f"name_to_users still has key for duplicate {D!r}",
                )
            assertions_ran["ok"] = True
            return False

        _StopBeforeDedupPasses.install(cb)

        @torch.compile(fullgraph=True)
        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        try:
            fn(x, w1, w2)
        except Exception:
            if not assertions_ran["ok"]:
                raise
        self.assertTrue(assertions_ran["ok"])

    # ------------------------------------------------------------------
    # test_provenance_transform_appended
    # ------------------------------------------------------------------

    def test_provenance_transform_appended(self) -> None:
        """merge_provenance appends exactly one ProvenanceTransform to
        the canonical constant with pass_name
        "dedup_and_promote_constants" for each duplicate absorbed.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        assertions_ran = {"ok": False}

        def cb(graph: GraphLowering) -> bool:
            from torch_spyre._inductor.dedup_constants import _constant_key

            constants = self._constants_in(graph.operations)
            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) > 1]
            self.assertTrue(multi, "no multi-constant group")
            canonical = multi[0][0]
            n_dups_in_group = len(multi[0]) - 1

            pre_history_len = len(getattr(canonical, "_spyre_prov_history", ()) or ())
            dedup_and_promote_constants(graph)
            post_history = getattr(canonical, "_spyre_prov_history", ())
            self.assertIsNotNone(post_history)
            new_entries = post_history[pre_history_len:]
            dedup_transforms = [
                t
                for t in new_entries
                if getattr(t, "pass_name", "") == "dedup_and_promote_constants"
            ]
            self.assertEqual(
                len(dedup_transforms),
                n_dups_in_group,
                f"expected {n_dups_in_group} new dedup ProvenanceTransform(s), "
                f"got {len(dedup_transforms)}",
            )
            for t in dedup_transforms:
                self.assertEqual(getattr(t, "kind", None), "fusion")
                self.assertEqual(getattr(t, "reason", None), "duplicate constant")
            assertions_ran["ok"] = True
            return False

        _StopBeforeDedupPasses.install(cb)

        @torch.compile(fullgraph=True)
        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        try:
            fn(x, w1, w2)
        except Exception:
            if not assertions_ran["ok"]:
                raise
        self.assertTrue(assertions_ran["ok"])

    # ------------------------------------------------------------------
    # test_no_duplicates_fast_path
    # ------------------------------------------------------------------

    def test_no_duplicates_fast_path(self) -> None:
        """When there are no duplicate groups, dedup must not do the
        expensive N-op reverse-index scan.

        Concretely: after building state where every
        SpyreConstantFallback has a distinct (value, dtype, device)
        key, we assert that `dedup_and_promote_constants` completes
        without calling `op.get_read_writes()` on any non-constant
        operation. We instrument by monkey-patching
        `ComputedBuffer.get_read_writes` to count calls made during
        dedup.

        Against the pristine a9316b3 dedup this passes because the
        pass never walks the operations list when no group has > 1
        entry. After the E-only refactor the same must hold — the
        E implementation MUST decide whether duplicates exist BEFORE
        building the reverse index.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        # Aligned K so no padding constant is generated at all.
        # If any SpyreConstantFallback IS generated, we deliberately
        # remove it (by adjusting canonical selection scope): the
        # test only requires "no duplicate GROUPS", not "no
        # constants".
        k_aligned = stick_size * 2
        x = torch.randn(2, 8, k_aligned, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k_aligned, 32, dtype=dtype, device="spyre")

        assertions_ran = {"ok": False}

        # Counter across ALL ComputedBuffer.get_read_writes calls made
        # during dedup. We install the counter as a class-level wrap
        # and reset it just before the dedup call.
        counter = {"n": 0}
        orig_grw = ComputedBuffer.get_read_writes

        def counted_grw(self):
            counter["n"] += 1
            return orig_grw(self)

        def cb(graph: GraphLowering) -> bool:
            from torch_spyre._inductor.dedup_constants import _constant_key

            constants = self._constants_in(graph.operations)
            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) > 1]
            self.assertFalse(
                multi,
                f"workload produced a duplicate group; cannot test "
                f"no-duplicate fast path ({len(multi)} multi-groups, "
                f"first has {len(multi[0]) if multi else 0} entries)",
            )

            with patch.object(ComputedBuffer, "get_read_writes", counted_grw):
                counter["n"] = 0
                dedup_and_promote_constants(graph)
                calls = counter["n"]

            # The pass may call get_read_writes on constants themselves
            # inside its scan (pristine behavior) or on N ops (an
            # E-only implementation that does NOT gate the reverse
            # index scan on 'duplicates exist' would call ~N of them).
            # The bound we assert is "no non-constant ComputedBuffer
            # get_read_writes calls should happen". Pristine dedup
            # never enters the redirect loop when no group has >1
            # entry, so calls == 0. Any E-only regression that builds
            # the reverse index unconditionally would trigger this.
            self.assertEqual(
                calls,
                0,
                f"no-duplicate fast path violated: dedup made {calls} "
                f"ComputedBuffer.get_read_writes calls when there "
                f"were no duplicate groups",
            )
            assertions_ran["ok"] = True
            return False

        _StopBeforeDedupPasses.install(cb)

        @torch.compile(fullgraph=True)
        def fn(x, w1):
            return torch.bmm(x, w1)

        try:
            fn(x, w1)
        except Exception:
            if not assertions_ran["ok"]:
                raise
        self.assertTrue(assertions_ran["ok"])


# ---------------------------------------------------------------------------
# Unit test for _build_reverse_consumer_index (dedup_and_promote_constants
# E-only refactor). Standalone — does not need the Spyre device.
# ---------------------------------------------------------------------------


class TestBuildReverseConsumerIndex(unittest.TestCase):
    """Guardrail for the E-only reverse-index construction.

    The pristine ``_redirect_consumers`` patches a matching op at most
    once per duplicate constant, regardless of how many separate
    dependency objects in ``op.get_read_writes().reads`` happen to
    share the same buffer name (a single op with two distinct
    ``ops.load(D, ...)`` at different indices produces two MemoryDep
    with the same ``.name``). The E-only index must preserve that:
    each op appears at most once in ``consumers_by_name[name]``.
    """

    def _fake_dep(self, name: str) -> Any:
        """A minimal Dep-like object with just ``.name``."""
        from types import SimpleNamespace

        return SimpleNamespace(name=name)

    def _fake_op(self, deps: list[Any]) -> Any:
        """A minimal Operation-like object whose get_read_writes returns
        an object with ``.reads`` == the given deps.
        """
        from types import SimpleNamespace

        rw = SimpleNamespace(reads=deps)
        return SimpleNamespace(get_read_writes=lambda: rw)

    def test_op_with_two_deps_same_name_appears_once(self) -> None:
        """An op whose reads contain two distinct dep objects with the
        same name D appears exactly once in consumers_by_name[D].
        """
        from torch_spyre._inductor.dedup_constants import (
            _build_reverse_consumer_index,
        )

        op = self._fake_op([self._fake_dep("bufD"), self._fake_dep("bufD")])
        idx = _build_reverse_consumer_index([op], {"bufD"})
        self.assertEqual(len(idx["bufD"]), 1)
        self.assertIs(idx["bufD"][0], op)

    def test_op_with_two_deps_different_names(self) -> None:
        """An op whose reads contain two distinct duplicate names D1
        and D2 appears once in each of consumers_by_name[D1] and
        consumers_by_name[D2].
        """
        from torch_spyre._inductor.dedup_constants import (
            _build_reverse_consumer_index,
        )

        op = self._fake_op([self._fake_dep("bufD1"), self._fake_dep("bufD2")])
        idx = _build_reverse_consumer_index([op], {"bufD1", "bufD2"})
        self.assertEqual(len(idx["bufD1"]), 1)
        self.assertEqual(len(idx["bufD2"]), 1)
        self.assertIs(idx["bufD1"][0], op)
        self.assertIs(idx["bufD2"][0], op)

    def test_op_with_no_duplicate_reads_absent_from_index(self) -> None:
        """An op that reads only non-duplicate names does not appear
        in the index at all.
        """
        from torch_spyre._inductor.dedup_constants import (
            _build_reverse_consumer_index,
        )

        op = self._fake_op([self._fake_dep("bufX"), self._fake_dep("bufY")])
        idx = _build_reverse_consumer_index([op], {"bufD"})
        self.assertNotIn("bufD", idx)
        self.assertNotIn("bufX", idx)
        self.assertNotIn("bufY", idx)

    def test_multiple_ops_deterministic_order(self) -> None:
        """Ops are appended in graph.operations order, preserving
        determinism for later passes and for debugging."""
        from torch_spyre._inductor.dedup_constants import (
            _build_reverse_consumer_index,
        )

        op1 = self._fake_op([self._fake_dep("bufD")])
        op2 = self._fake_op([self._fake_dep("bufD"), self._fake_dep("bufD")])
        op3 = self._fake_op([self._fake_dep("bufOther")])
        op4 = self._fake_op([self._fake_dep("bufD")])
        idx = _build_reverse_consumer_index([op1, op2, op3, op4], {"bufD"})
        self.assertEqual(idx["bufD"], [op1, op2, op4])


if __name__ == "__main__":
    unittest.main()
