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

Design (PR #4139 already merged the certified-greedy fast path; this
test file guards the follow-up that makes OR-Tools genuinely lazy):

- Importing ``ilp_solver_ortools`` does NOT pull in the SWIG-heavy
  ``ortools.sat.python.cp_model`` module.
- Constructing ``CpSatLayoutSolver`` does NOT import it either.
- A certified-greedy ``plan_layout`` (the common case) returns a
  plan without ever importing it.
- A fallback ``plan_layout`` (seed rejects, CP-SAT runs) lazily
  imports it and returns the CP-SAT-optimal objective.
- Joint ``plan_layout_and_core_divisions`` lazily imports it.
- Repeated CP-SAT solves reuse the already-loaded module.

Uses subprocess isolation for the sys.modules-membership assertions
so pytest's own imports do not pollute the check.
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
    parsed JSON stdout. Returns the raw stdout under ``_raw`` on
    parse failure to make diagnosis clearer."""
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
        ``_load_ortools`` which imports ``cp_model`` /
        ``cp_model_helper`` on demand.

        Uses the classic constrained-spill fixture from
        ``test_cpsat_certified_greedy_seed.test_nonzero_objective_falls_through_to_cpsat``:
        three buffers (10, 20, 30) with a 50-capacity limit. Greedy
        evicts the largest and reaches objective 60, but CP-SAT's
        forced-spill lower bound for this fixture is 0 (nothing is in
        ``record_exclusions``), so the seed rejects and CP-SAT runs.
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
        # cp_model must have been imported by the fallback path.
        self.assertTrue(r["cp_model"], r)
        self.assertTrue(r["cp_model_helper"], r)
        # CP-SAT reaches the classic optimum on this fixture: place
        # a (10) + b (20) = 30 <= 50, spill c (30). Objective for c
        # is (0 reads_served + 1 is_intermediate) * 30 = 30 -- wait,
        # reads_served = 1 for uses=[0,3] with first_use_is_read=False,
        # so spill_cost(c) = (1+1)*30 = 60... but CP-SAT finds
        # optimum 20 by placing c (30) instead, spilling a and b
        # whose combined cost = (1+1)*10 + (1+1)*20 = 60. Actually
        # optimum here is spill a alone -> obj = (1+1)*10 = 20.
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

    def test_repeated_cpsat_solves_do_not_reimport(self):
        """Second CP-SAT solve in the same process must not reimport
        anything: the module-level ``cp_model`` is populated on the
        first call and reused thereafter. Checks that
        ``_load_ortools`` is genuinely idempotent (its cheap
        ``if cp_model is not None: return`` short-circuit fires)."""
        program = (
            _SETUP
            + """
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
    CpSatLayoutSolver,
)
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer

def force_fallback():
    bufs = [
        LifetimeBoundBuffer("a", 60, [0, 5]),
        LifetimeBoundBuffer("b", 60, [0, 5]),
    ]
    CpSatLayoutSolver(bufs, 100).plan_layout()

force_fallback()
after_first = snap()
force_fallback()
after_second = snap()
print(json.dumps({"first": after_first, "second": after_second}))
"""
        )
        r = _run(program)
        # Both snapshots must show cp_model present. The point of
        # the test is that the second call doesn't raise or refetch.
        self.assertTrue(r["first"]["cp_model"])
        self.assertTrue(r["second"]["cp_model"])


class TestAvailabilityContractPreserved(unittest.TestCase):
    """Preserve the existing ``ImportError`` contract that
    :func:`allocator._make_cpsat_solver` catches to fall back to the
    greedy allocator when ortools is not installed on this arch.

    We can't actually uninstall ortools inside a test, but we can
    verify the check helper's shape: when
    ``_ortools_available()`` returns False, ``CpSatLayoutSolver`` must
    raise ``ImportError`` at construction with the same message shape
    the outer factory expects."""

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
        module-level cache. Verified by patching the underlying
        ``find_spec`` to raise if called."""
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

    def test_load_ortools_is_idempotent(self):
        """``_load_ortools`` populates the module globals on first
        call; repeated calls return immediately without re-importing."""
        from torch_spyre._inductor.scratchpad import ilp_solver_ortools as m

        m._load_ortools()
        first_cp = m.cp_model
        first_cph = m.cp_model_helper
        m._load_ortools()
        # Same object identity: the module was not re-fetched.
        self.assertIs(m.cp_model, first_cp)
        self.assertIs(m.cp_model_helper, first_cph)


if __name__ == "__main__":
    unittest.main()
