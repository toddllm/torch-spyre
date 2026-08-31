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

"""Unit tests for the lazy OR-Tools load in
``ilp_solver_ortools``.

This test file stacks on the certified-greedy fast path introduced in
PR #4139 (``CpSatLayoutSolver.plan_layout`` runs a greedy probe first
and skips CP-SAT entirely when its plan attains the forced-spill
lower bound of the residency objective). It guards the follow-up
change that makes OR-Tools genuinely lazy so certified compiles do
not trigger the ~1.4 s SWIG bootstrap at all.

Guarantees:

- Importing ``ilp_solver_ortools`` does NOT pull in the SWIG-heavy
  ``ortools.sat.python.cp_model`` module.
- Constructing ``CpSatLayoutSolver`` does NOT import it either.
- A certified-greedy ``plan_layout`` (the common case) returns a
  plan without ever importing it.
- A fallback ``plan_layout`` (seed rejects; CP-SAT runs) lazily
  imports it and returns the CP-SAT-optimal objective.
- Joint ``plan_layout_and_core_divisions`` lazily imports it, on both
  the default residency-lex-solve path and the #3810 ``cost_expr``
  branch.
- Repeated CP-SAT solves reuse the already-loaded module; the real
  import (``_do_ortools_import``) runs exactly once across both
  serial repeat calls and concurrent contention.
- Availability check is robust to ``ModuleNotFoundError`` /
  ``ImportError`` / ``ValueError`` from ``find_spec``.
- The first-load critical section is protected by a lock.

Uses subprocess isolation for the ``sys.modules``-membership
assertions so pytest's own imports do not pollute the check.
"""

import json
import os
import subprocess
import sys
import unittest


# Path to the current Python; matches the venv the test runner uses.
_PYTHON = sys.executable


def _run(script: str) -> dict:
    """Run a small python program in a subprocess and return its
    parsed JSON stdout (the last non-empty stdout line). Raises with
    the full transcript on parse or subprocess failure so a broken
    test is diagnosable at a glance.
    """
    env = os.environ.copy()
    # Suppress the PrivateUse1 autoload so subprocess start is fast:
    # nothing in these tests needs the Spyre device backend.
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    result = subprocess.run(
        [_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"subprocess exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise AssertionError(
            f"could not parse subprocess stdout as JSON:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from exc


# Small helper program templates. Each dumps a single JSON dict at
# the end.
_SETUP = """
import json
import sys
import torch  # noqa
import torch_spyre  # noqa


def _has(name):
    return name in sys.modules


def snap():
    return {
        "ortools": _has("ortools"),
        "cp_model": _has("ortools.sat.python.cp_model"),
        "cp_model_helper": _has("ortools.sat.python.cp_model_helper"),
    }
"""


class TestLazyOrtoolsLoad(unittest.TestCase):
    def test_import_ilp_solver_ortools_does_not_import_cp_model(self):
        """Just importing the scratchpad ilp solver module must not
        trigger the SWIG ``cp_model`` import. It is safe for
        ``importlib.util.find_spec`` to walk the ``ortools`` /
        ``ortools.sat`` / ``ortools.sat.python`` package tree (those
        packages are cheap ``__init__.py`` files); ``cp_model`` itself
        is where the ~1.4 s cost lives."""
        program = (
            _SETUP
            + """
from torch_spyre._inductor.scratchpad import ilp_solver_ortools  # noqa
print(json.dumps(snap()))
"""
        )
        r = _run(program)
        self.assertFalse(r["cp_model"], r)
        self.assertFalse(r["cp_model_helper"], r)

    def test_constructing_solver_does_not_import_cp_model(self):
        """Building a ``CpSatLayoutSolver`` instance runs the
        availability check but not the load."""
        program = (
            _SETUP
            + """
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
    CpSatLayoutSolver,
)
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer

bufs = [LifetimeBoundBuffer(f"b{i}", 100, [i, i+2]) for i in range(4)]
_ = CpSatLayoutSolver(bufs, 100_000)
print(json.dumps(snap()))
"""
        )
        r = _run(program)
        self.assertFalse(r["cp_model"], r)
        self.assertFalse(r["cp_model_helper"], r)

    def test_certified_plan_layout_does_not_import_cp_model(self):
        """The common case: capacity is generous, the certified
        greedy seed accepts, and CP-SAT is skipped entirely. No
        OR-Tools SWIG import is triggered."""
        program = (
            _SETUP
            + """
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
    CpSatLayoutSolver,
)
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer

bufs = [LifetimeBoundBuffer(f"b{i}", 100, [i, i+2]) for i in range(4)]
plan = CpSatLayoutSolver(bufs, 100_000).plan_layout()
# Sanity: every buffer placed (capacity dwarfs live footprint).
placed = sum(1 for b in plan if b.address is not None)
result = snap()
result["n_placed"] = placed
print(json.dumps(result))
"""
        )
        r = _run(program)
        self.assertFalse(r["cp_model"], r)
        self.assertFalse(r["cp_model_helper"], r)
        self.assertEqual(r["n_placed"], 4)

    def test_fallback_plan_layout_lazily_imports_cp_model(self):
        """When the seed rejects, ``_plan_layout_generic`` calls
        ``_load_ortools`` which imports ``cp_model`` and
        ``cp_model_helper`` on demand.

        Fixture: the classic constrained-spill case reused from
        ``test_cpsat_certified_greedy_seed``. Three computed
        intermediates a=10, b=20, c=30, all live across [0, 3),
        alignment=1, capacity=50. Combined live footprint 60 > 50,
        so at least one buffer must be spilled. Each buffer's
        ``spill_cost`` = ``(read_count + is_intermediate) * size``
        = ``(1 + 1) * size`` = ``2 * size``. Greedy spills the
        largest (c) reaching objective ``2 * 30 = 60``; CP-SAT's
        forced-spill lower bound is 0 (nothing pinned by
        ``record_exclusions``), so ``60 > 0`` and the seed rejects.
        CP-SAT then finds the true optimum: spill ``a`` alone
        (``2 * 10 = 20``), which fits ``b`` (20) and ``c`` (30) in
        the 50-byte capacity.
        """
        program = (
            _SETUP
            + """
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
    CpSatLayoutSolver, _hbm_spill_cost,
)
from dataclasses import replace
from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer, ceil_div,
)

bufs = [
    LifetimeBoundBuffer("a", 10, [0, 3]),
    LifetimeBoundBuffer("b", 20, [0, 3]),
    LifetimeBoundBuffer("c", 30, [0, 3]),
]
alignment = 1
plan = CpSatLayoutSolver(bufs, 50, alignment=alignment).plan_layout()
obj = sum(
    _hbm_spill_cost(replace(b, size=ceil_div(b.size, alignment)))
    for b in plan
    if b.address is None
)
result = snap()
result["objective_units"] = obj
result["n_placed"] = sum(1 for b in plan if b.address is not None)
result["placed_names"] = sorted(
    b.name for b in plan if b.address is not None
)
print(json.dumps(result))
"""
        )
        r = _run(program)
        self.assertTrue(r["cp_model"], r)
        self.assertTrue(r["cp_model_helper"], r)
        # CP-SAT reaches objective 20 (spill a; place b and c).
        self.assertEqual(r["objective_units"], 20, r)

    def test_joint_path_lazily_imports_cp_model(self):
        """``plan_layout_and_core_divisions`` unconditionally reaches
        ``_plan_layout_generic`` and therefore lazily loads
        OR-Tools."""
        program = (
            _SETUP
            + """
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision, CoreDivisionBuffer,
)
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
    CpSatLayoutSolver,
)

whole = [CoreDivision()]
bufs = [
    CoreDivisionBuffer(f"c{i}", 100, [i, i+2], core_divisions=whole)
    for i in range(3)
]
plan = CpSatLayoutSolver(bufs, 100_000).plan_layout_and_core_divisions()
result = snap()
result["n_ops"] = len(plan)
print(json.dumps(result))
"""
        )
        r = _run(program)
        self.assertTrue(r["cp_model"], r)
        self.assertEqual(r["n_ops"], 3)

    def test_joint_path_with_cost_expr_lazily_imports_and_runs(self):
        """#3810's ``cost_expr`` branch: pass a small nonconstant
        sympy expression to ``plan_layout_and_core_divisions``,
        prove that OR-Tools was not loaded before the call, that
        the call triggers the lazy load, and that the returned plan
        respects the cost_expr's preference.

        The cost expression is ``-(sym_is_lx_a + sym_is_lx_b + sym_is_lx_c)``
        -- ``_minimize_cost_expr`` maps each buffer's
        ``sym_is_lx.name`` to its CP-SAT ``in_buffer`` bool var, so
        minimizing the negation is equivalent to maximizing the
        residency count. Capacity is generous, so the optimum is
        every buffer resident; that is behaviorally testable
        without depending on solver arithmetic beyond "resident is
        preferred over spilled."
        """
        program = (
            _SETUP
            + """
import sympy
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision, CoreDivisionBuffer,
)
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
    CpSatLayoutSolver,
)

whole = [CoreDivision()]
bufs = [
    CoreDivisionBuffer(f"c{i}", 100, [i, i+2], core_divisions=whole)
    for i in range(3)
]

# Before the joint call: ortools must not be loaded yet.
before = snap()

# Build the cost_expr against the buffers' sym_is_lx symbols.
# ``_minimize_cost_expr`` in ilp_solver_ortools.py resolves
# ``buffer.sym_is_lx.name`` to the CP-SAT ``in_buffer`` var. A
# negated sum maximizes residency.
cost_expr = -sum(b.sym_is_lx for b in bufs)
plan = CpSatLayoutSolver(
    bufs, 100_000,
).plan_layout_and_core_divisions(cost_expr=cost_expr)

# After: cp_model must be loaded.
after = snap()

result = {
    "before": before,
    "after": after,
    "n_ops": len(plan),
    "n_resident": sum(1 for b in plan if b.address is not None),
    "n_spilled": sum(1 for b in plan if b.address is None),
}
print(json.dumps(result))
"""
        )
        r = _run(program)
        self.assertFalse(
            r["before"]["cp_model"], "cp_model must not be loaded before joint call"
        )
        self.assertFalse(
            r["before"]["cp_model_helper"],
            "cp_model_helper must not be loaded before joint call",
        )
        self.assertTrue(
            r["after"]["cp_model"], "joint call must have triggered the lazy load"
        )
        self.assertTrue(r["after"]["cp_model_helper"])
        self.assertEqual(r["n_ops"], 3)
        # cost_expr said "prefer residency" and capacity was generous,
        # so every buffer should be resident.
        self.assertEqual(r["n_resident"], 3, r)
        self.assertEqual(r["n_spilled"], 0, r)

    def test_repeated_cpsat_solves_load_exactly_once(self):
        """Two CP-SAT solves in one process must trigger the real
        OR-Tools import exactly once. Proved deterministically by
        patching ``_do_ortools_import`` (the tiny helper that owns
        the actual ``from ortools.sat.python import ...`` statement)
        with a counter wrapper: after two solves, the counter must
        read 1.

        No wall-clock assertion. Module-identity is checked as a
        supporting invariant but is not the primary proof.
        """
        program = (
            _SETUP
            + """
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
    CpSatLayoutSolver,
)
from torch_spyre._inductor.scratchpad import ilp_solver_ortools as m
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer

# Install a counter around _do_ortools_import BEFORE anything else
# runs. The lazy loader has not been called yet on this process, so
# cp_model / cp_model_helper are still None: the first solve will
# unavoidably reach _do_ortools_import, and the second must NOT.
call_count = {"n": 0}
_real_import = m._do_ortools_import


def counted_import():
    call_count["n"] += 1
    return _real_import()


m._do_ortools_import = counted_import


def force_fallback():
    bufs = [
        LifetimeBoundBuffer("a", 10, [0, 3]),
        LifetimeBoundBuffer("b", 20, [0, 3]),
        LifetimeBoundBuffer("c", 30, [0, 3]),
    ]
    return CpSatLayoutSolver(bufs, 50, alignment=1).plan_layout()

force_fallback()
first_cp = m.cp_model
first_cph = m.cp_model_helper
count_after_first = call_count["n"]

force_fallback()
second_cp = m.cp_model
second_cph = m.cp_model_helper
count_after_second = call_count["n"]

print(json.dumps({
    "count_after_first": count_after_first,
    "count_after_second": count_after_second,
    "same_cp_model": first_cp is second_cp,
    "same_cp_model_helper": first_cph is second_cph,
    "cp_model_present": snap()["cp_model"],
    "cp_model_helper_present": snap()["cp_model_helper"],
}))
"""
        )
        r = _run(program)
        self.assertEqual(
            r["count_after_first"],
            1,
            "first solve must trigger exactly one real import",
        )
        self.assertEqual(
            r["count_after_second"], 1, "second solve must not trigger any real import"
        )
        # Supporting invariants.
        self.assertTrue(r["same_cp_model"], r)
        self.assertTrue(r["same_cp_model_helper"], r)
        self.assertTrue(r["cp_model_present"])
        self.assertTrue(r["cp_model_helper_present"])


class TestAvailabilityContractPreserved(unittest.TestCase):
    """Preserve the existing ``ImportError`` contract that
    :func:`allocator._make_cpsat_solver` catches to fall back to the
    greedy allocator when ortools is not installed on this arch.

    We can't actually uninstall ortools inside a test, so we test the
    helpers' behavior on a variety of ``find_spec`` returns."""

    def test_availability_helper_matches_find_spec(self):
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            _ortools_available,
        )
        import importlib.util

        # Fresh call has to line up with find_spec's answer.
        expected = importlib.util.find_spec("ortools.sat.python.cp_model") is not None
        self.assertEqual(_ortools_available(), expected)

    def test_availability_helper_is_cached(self):
        """The cheap ``find_spec`` runs once; subsequent calls hit the
        module-level cache. Verified by patching ``find_spec`` to
        raise if called after the cache is primed."""
        from unittest.mock import patch

        from torch_spyre._inductor.scratchpad import ilp_solver_ortools as m

        # Prime the cache once.
        first = m._ortools_available()

        # Now any call would hit the cache -- find_spec must not run.
        def _boom(*args, **kwargs):
            raise AssertionError(
                "find_spec must not run again after the first cache hit"
            )

        with patch("importlib.util.find_spec", side_effect=_boom):
            self.assertEqual(m._ortools_available(), first)
            self.assertEqual(m._ortools_available(), first)

    def test_availability_helper_absent_returns_false(self):
        """When ``find_spec`` returns ``None`` (package genuinely
        absent), ``_ortools_available`` returns False and caches
        that."""
        from unittest.mock import patch

        from torch_spyre._inductor.scratchpad import ilp_solver_ortools as m

        # Reset the cache to force the probe to run again.
        with (
            patch.object(m, "_ORTOOLS_AVAILABLE", None),
            patch(
                "importlib.util.find_spec",
                return_value=None,
            ),
        ):
            self.assertIs(m._ortools_available(), False)

    def test_availability_helper_module_not_found_returns_false(self):
        """When a parent package on the dotted lookup is missing,
        ``find_spec`` raises ``ModuleNotFoundError``. The helper must
        translate that into a False result rather than let it
        escape."""
        from unittest.mock import patch

        from torch_spyre._inductor.scratchpad import ilp_solver_ortools as m

        with (
            patch.object(m, "_ORTOOLS_AVAILABLE", None),
            patch(
                "importlib.util.find_spec",
                side_effect=ModuleNotFoundError("No module named 'ortools'"),
            ),
        ):
            self.assertIs(m._ortools_available(), False)

    def test_availability_helper_value_error_returns_false(self):
        """Some frozen distributions can produce ``ValueError`` from
        ``find_spec`` when a parent package's ``__spec__`` is
        ``None``. Treated the same as absent."""
        from unittest.mock import patch

        from torch_spyre._inductor.scratchpad import ilp_solver_ortools as m

        with (
            patch.object(m, "_ORTOOLS_AVAILABLE", None),
            patch(
                "importlib.util.find_spec",
                side_effect=ValueError("__spec__ is None"),
            ),
        ):
            self.assertIs(m._ortools_available(), False)

    def test_load_ortools_is_idempotent(self):
        """``_load_ortools`` populates the module globals on first
        call; repeated calls return immediately without re-importing.
        """
        from torch_spyre._inductor.scratchpad import ilp_solver_ortools as m

        m._load_ortools()
        first_cp = m.cp_model
        first_cph = m.cp_model_helper
        m._load_ortools()
        # Same object identity: the module was not re-fetched.
        self.assertIs(m.cp_model, first_cp)
        self.assertIs(m.cp_model_helper, first_cph)

    def test_load_ortools_publishes_both_globals(self):
        """After ``_load_ortools`` returns successfully, both
        ``cp_model`` and ``cp_model_helper`` must be non-None -- no
        caller may observe a half-initialized state."""
        from torch_spyre._inductor.scratchpad import ilp_solver_ortools as m

        m._load_ortools()
        self.assertIsNotNone(m.cp_model)
        self.assertIsNotNone(m.cp_model_helper)


class TestConcurrentFirstLoad(unittest.TestCase):
    """The first-load critical section is protected by a
    ``threading.Lock``. Every thread that returns from
    ``_load_ortools`` must observe both ``cp_model`` and
    ``cp_model_helper`` bound; no successful caller may see a half-
    published pair.

    Guarding the actual invariant, not just "two threads finish".
    Uses a slow inner import to enlarge the race window, then
    stress-tests with many threads and asserts each one saw both
    globals non-None immediately after its own ``_load_ortools``
    returned. The number of real imports done under contention must
    still be exactly one (the same guarantee
    :meth:`test_repeated_cpsat_solves_load_exactly_once` checks in
    the serial case).
    """

    def test_concurrent_first_load_publishes_atomically(self):
        """Widened race window: patch ``_do_ortools_import`` to
        sleep briefly before returning, then race N threads through
        ``_load_ortools`` and assert:

        * every worker saw ``cp_model is not None`` AND
          ``cp_model_helper is not None`` right after its own
          ``_load_ortools`` returned;
        * every worker saw identical object identities for both
          globals;
        * ``_do_ortools_import`` was invoked exactly once total
          (proves the lock actually serialised, not that all threads
          happened to skip the import section).
        """
        program = (
            _SETUP
            + """
import threading
import time

from torch_spyre._inductor.scratchpad import ilp_solver_ortools as m

# Widen the race window by slowing the actual import. A concurrent
# caller entering the lock after the slow import completes must see
# both globals bound in one visit -- an outer fast-path check would
# let it see the first assignment before the second.
_real_import = m._do_ortools_import
call_count = {"n": 0}
count_lock = threading.Lock()


def slow_import():
    with count_lock:
        call_count["n"] += 1
    result = _real_import()
    time.sleep(0.10)  # 100 ms of extra window
    return result


m._do_ortools_import = slow_import

N = 8
barrier = threading.Barrier(N)
results = [None] * N
errors = [None] * N


def worker(idx):
    try:
        barrier.wait()  # synchronize the start
        m._load_ortools()
        # Read both globals immediately in the worker so its
        # observation is captured atomically with respect to this
        # worker's own return.
        cp = m.cp_model
        cph = m.cp_model_helper
        results[idx] = {
            "cp_is_none": cp is None,
            "cph_is_none": cph is None,
            "cp_id": id(cp) if cp is not None else None,
            "cph_id": id(cph) if cph is not None else None,
        }
    except Exception as exc:  # noqa: BLE001
        errors[idx] = f"{type(exc).__name__}: {exc}"


threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()

cp_ids = sorted({r["cp_id"] for r in results if r})
cph_ids = sorted({r["cph_id"] for r in results if r})
any_saw_half = any(
    r and (r["cp_is_none"] or r["cph_is_none"])
    for r in results
)
print(json.dumps({
    "results": results,
    "errors": errors,
    "call_count": call_count["n"],
    "unique_cp_ids": cp_ids,
    "unique_cph_ids": cph_ids,
    "any_saw_half_published": any_saw_half,
    "n_workers": N,
}))
"""
        )
        r = _run(program)
        # No worker raised.
        for err in r["errors"]:
            self.assertIsNone(err, r)
        # No worker observed a half-published pair.
        self.assertFalse(r["any_saw_half_published"], r)
        # All workers agreed on the identities of both globals.
        self.assertEqual(len(r["unique_cp_ids"]), 1, r)
        self.assertEqual(len(r["unique_cph_ids"]), 1, r)
        # The lock actually serialised: exactly one real import ran
        # under contention.
        self.assertEqual(r["call_count"], 1, r)


if __name__ == "__main__":
    unittest.main()
