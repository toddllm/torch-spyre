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

"""CPU-only tests for the paged-attention learning harness.

These tests need only ``torch`` -- no ``torch_spyre._C`` and no Spyre hardware.
They validate the numerics of the three execution forms, the indirect-access
gather primitives, grouped-query expansion, masking, the CLI's JSON output, and
the outcome classifier (both a success case and controlled non-ok cases).

Run directly (no repo conftest / Spyre runtime required)::

    python tests/indirect_access/test_paged_attention_learning_harness.py

Or under pytest with the repo conftest bypassed (it imports Spyre backend
pieces that are not built locally)::

    python -m pytest tests/indirect_access/test_paged_attention_learning_harness.py \\
        -p no:cacheprovider -c /dev/null -q
"""

import json
import os
import sys

import torch

# Make the harness module importable without installing the package.
_EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples",
    "indirect_access",
)
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

import paged_attention_learning_harness as harness  # noqa: E402

REQUIRED_FIELDS = (
    "case_name",
    "device",
    "dtype",
    "batch",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "num_pages",
    "blocks_per_seq",
    "mode",
    "compile_enabled",
    "status",
    "max_abs_diff",
    "max_rel_diff",
    "allclose",
    "error_type",
    "error_message",
)


def _base_cfg(**overrides):
    params = dict(case_name="test", seed=0)
    params.update(overrides)
    return harness.ProblemConfig(**params)


# ---------------------------------------------------------------------------
# Numerics: the three execution forms agree
# ---------------------------------------------------------------------------
def test_dense_equals_online_fp32():
    """dense_reference and online_page_loop match tightly in fp32."""
    cfg = _base_cfg()
    inp = harness.build_inputs(cfg, "cpu", torch.float32)
    dense = harness.dense_reference(
        inp["query"],
        inp["k_pages"],
        inp["v_pages"],
        inp["block_table"],
        inp["mask"],
        inp["scale"],
    )
    online = harness.online_page_loop(
        inp["query"],
        inp["k_pages"],
        inp["v_pages"],
        inp["block_table"],
        inp["mask"],
        inp["scale"],
        inp["block_size"],
    )
    assert dense.shape == online.shape
    assert torch.allclose(dense, online, rtol=1e-4, atol=1e-4)


def test_dense_equals_online_fp16():
    """dense_reference and online_page_loop stay close in fp16."""
    cfg = _base_cfg()
    inp = harness.build_inputs(cfg, "cpu", torch.float16)
    dense = harness.dense_reference(
        inp["query"],
        inp["k_pages"],
        inp["v_pages"],
        inp["block_table"],
        inp["mask"],
        inp["scale"],
    )
    online = harness.online_page_loop(
        inp["query"],
        inp["k_pages"],
        inp["v_pages"],
        inp["block_table"],
        inp["mask"],
        inp["scale"],
        inp["block_size"],
    )
    assert torch.allclose(dense.float(), online.float(), rtol=1e-2, atol=2e-2)


def test_indirect_gather_matches_dense_fp32():
    """The gather-based form equals the index_select-based dense form."""
    cfg = _base_cfg()
    inp = harness.build_inputs(cfg, "cpu", torch.float32)
    dense = harness.dense_reference(
        inp["query"],
        inp["k_pages"],
        inp["v_pages"],
        inp["block_table"],
        inp["mask"],
        inp["scale"],
    )
    indirect = harness.indirect_gather_shape(
        inp["query"],
        inp["k_pages"],
        inp["v_pages"],
        inp["block_table"],
        inp["mask"],
        inp["scale"],
    )
    assert torch.equal(dense, indirect)


# ---------------------------------------------------------------------------
# Indirect-access primitives: 2D block-table gather == direct indexing
# ---------------------------------------------------------------------------
def test_block_table_gather_matches_indexing():
    """Both page-selection forms equal plain advanced indexing for a 2D table."""
    gen = torch.Generator().manual_seed(3)
    pages = torch.randn(12, 2, 16, 32, generator=gen)
    block_table = torch.tensor([[0, 4, 9, 1], [7, 1, 2, 11]], dtype=torch.int64)
    flat = block_table.reshape(-1)

    via_index_select = harness.gather_pages_index_select(pages, flat)
    via_gather = harness.gather_pages_gather(pages, flat)
    reference = pages[flat]

    assert tuple(via_index_select.shape) == (8, 2, 16, 32)
    assert torch.equal(via_index_select, reference)
    assert torch.equal(via_gather, reference)
    # The two indirect forms must agree with each other as well.
    assert torch.equal(via_index_select, via_gather)


# ---------------------------------------------------------------------------
# Grouped-query attention head expansion
# ---------------------------------------------------------------------------
def test_grouped_query_head_expansion():
    """expand_kv_heads broadcasts each kv head across its query-head group."""
    t = torch.randn(2, 2, 5, 8)  # [B, num_kv_heads=2, S, D]
    expanded = harness.expand_kv_heads(t, num_q_heads=8)
    assert tuple(expanded.shape) == (2, 8, 5, 8)
    # repeat_interleave => heads [0,0,0,0,1,1,1,1].
    for group_start, kv in ((0, 0), (4, 1)):
        for offset in range(4):
            assert torch.equal(expanded[:, group_start + offset], t[:, kv])
    # Equal head counts must be a no-op.
    assert torch.equal(harness.expand_kv_heads(t, num_q_heads=2), t)


def test_grouped_query_rejects_non_multiple():
    """A non-divisible head count is a clear ValueError, not silent garbage."""
    t = torch.randn(1, 3, 4, 8)
    try:
        harness.expand_kv_heads(t, num_q_heads=8)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-multiple head counts")


# ---------------------------------------------------------------------------
# Masking: a masked block changes the output and produces no NaNs
# ---------------------------------------------------------------------------
def test_masked_block_changes_output_no_nan():
    """Fully masking a block changes attention output and never yields NaNs."""
    unmasked_cfg = _base_cfg()
    masked_cfg = _base_cfg(masked_block=2)
    un = harness.build_inputs(unmasked_cfg, "cpu", torch.float32)
    ma = harness.build_inputs(masked_cfg, "cpu", torch.float32)

    out_unmasked = harness.online_page_loop(
        un["query"],
        un["k_pages"],
        un["v_pages"],
        un["block_table"],
        un["mask"],
        un["scale"],
        un["block_size"],
    )
    out_masked = harness.online_page_loop(
        ma["query"],
        ma["k_pages"],
        ma["v_pages"],
        ma["block_table"],
        ma["mask"],
        ma["scale"],
        ma["block_size"],
    )
    assert torch.isfinite(out_masked).all()
    assert not torch.allclose(out_unmasked, out_masked)
    # Dense and online must still agree under the mask.
    dense_masked = harness.dense_reference(
        ma["query"],
        ma["k_pages"],
        ma["v_pages"],
        ma["block_table"],
        ma["mask"],
        ma["scale"],
    )
    assert torch.allclose(dense_masked, out_masked, rtol=1e-4, atol=1e-4)


def test_variable_context_is_finite_and_consistent():
    """Variable per-sequence context lengths stay finite and self-consistent."""
    cfg = _base_cfg(variable_context=True)
    inp = harness.build_inputs(cfg, "cpu", torch.float32)
    assert inp["context_lens"].min().item() >= cfg.block_size
    dense = harness.dense_reference(
        inp["query"],
        inp["k_pages"],
        inp["v_pages"],
        inp["block_table"],
        inp["mask"],
        inp["scale"],
    )
    online = harness.online_page_loop(
        inp["query"],
        inp["k_pages"],
        inp["v_pages"],
        inp["block_table"],
        inp["mask"],
        inp["scale"],
        inp["block_size"],
    )
    assert torch.isfinite(dense).all()
    assert torch.allclose(dense, online, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Classification: success and controlled non-ok cases
# ---------------------------------------------------------------------------
def test_run_case_success_schema_and_status():
    """A good CPU case is eager_ok and carries every required schema field."""
    cfg = _base_cfg()
    res = harness.run_case(
        cfg, "dense_reference", "cpu", "float32", compile_enabled=False
    )
    assert res.status == "eager_ok"
    assert res.status in harness.STATUSES
    assert res.allclose is True
    row = res.to_dict()
    for field in REQUIRED_FIELDS:
        assert field in row, f"missing required field {field!r}"
    assert row["mode"] == "dense_reference"
    assert row["error_type"] is None


def test_run_case_all_kernels_eager_ok():
    """All three execution forms classify as eager_ok on a good fp16 case."""
    cfg = _base_cfg()
    for kernel in harness.KERNEL_NAMES:
        res = harness.run_case(cfg, kernel, "cpu", "float16", compile_enabled=False)
        assert res.status == "eager_ok", (kernel, res.status, res.error_message)


def test_run_case_skips_when_backend_missing():
    """device=spyre with no built backend classifies as skipped_backend_missing."""
    cfg = _base_cfg()
    res = harness.run_case(
        cfg, "dense_reference", "spyre", "float16", compile_enabled=False
    )
    assert res.status == "skipped_backend_missing"
    assert res.error_type is not None


def test_run_case_runtime_failure_is_classified():
    """A malformed config (non-divisible heads) classifies as runtime_failed."""
    bad = _base_cfg(num_q_heads=6, num_kv_heads=4)  # 6 % 4 != 0
    res = harness.run_case(
        bad, "dense_reference", "cpu", "float32", compile_enabled=False
    )
    assert res.status == "runtime_failed"
    assert res.error_type == "ValueError"
    assert res.allclose is None


# ---------------------------------------------------------------------------
# CLI: JSON output for a small sweep
# ---------------------------------------------------------------------------
def test_cli_writes_json_for_sweep(tmp_path):
    """run_cli(--mode sweep --json-out ...) writes a well-formed JSON payload."""
    out = tmp_path / "sweep.json"
    rc = harness.run_cli(
        [
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--kernel",
            "dense_reference",
            "--mode",
            "sweep",
            "--json-out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert "results" in payload and payload["results"]
    assert "status_counts" in payload
    assert "environment" in payload
    for row in payload["results"]:
        for field in REQUIRED_FIELDS:
            assert field in row
        assert row["status"] in harness.STATUSES


def test_cli_probe_env_writes_json(tmp_path):
    """probe-env mode emits an environment payload as JSON."""
    out = tmp_path / "env.json"
    rc = harness.run_cli(["--mode", "probe-env", "--json-out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["environment"]["torch_version"]
    assert "dynamo_explain_available" in payload["environment"]


# ---------------------------------------------------------------------------
# Direct runner (no pytest required)
# ---------------------------------------------------------------------------
def _run_all():
    import pathlib
    import tempfile

    tests = [
        test_dense_equals_online_fp32,
        test_dense_equals_online_fp16,
        test_indirect_gather_matches_dense_fp32,
        test_block_table_gather_matches_indexing,
        test_grouped_query_head_expansion,
        test_grouped_query_rejects_non_multiple,
        test_masked_block_changes_output_no_nan,
        test_variable_context_is_finite_and_consistent,
        test_run_case_success_schema_and_status,
        test_run_case_all_kernels_eager_ok,
        test_run_case_skips_when_backend_missing,
        test_run_case_runtime_failure_is_classified,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    with tempfile.TemporaryDirectory() as root:
        for t in (test_cli_writes_json_for_sweep, test_cli_probe_env_writes_json):
            try:
                t(pathlib.Path(root))
                print(f"PASS {t.__name__}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    total = len(tests) + 2
    print(f"\n{total - failures}/{total} passed")
    return failures == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
