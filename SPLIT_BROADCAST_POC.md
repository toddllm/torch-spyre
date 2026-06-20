# torch-spyre split-broadcast POC — Public README

This describes a throwaway, probe-only patch on top of `torch-spyre` that
routes `SpyreCCLBackend::broadcast` through the split-collectives API
(`broadcast(TensorInfo)` returning `WorkScheduleInfo`, plus
`broadcast_applyTensor(WorkScheduleInfo&, Tensor&)` returning
`WorkSchedule`) instead of the single-call form.

The patch is used to characterise compute/comm overlap on Spyre 2-rank
single-pod configurations. It is NOT a candidate for upstream merge.

## Where The Patch Lives

```
   github.com/toddllm/torch-spyre
       |
       +-- upstream-base-93c376a2     (snapshot of upstream torch-spyre @ 93c376a2)
       |
       +-- tdeshane/split-broadcast-poc   (POC + split patches applied on top)
            |
            +-- da695dd  WIP: route SpyreCCLBackend::broadcast through split-collectives API
```

## What The Patch Does

Two changes stacked on `torch-spyre@93c376a2`:

1. **Async-broadcast POC**:
   - Removes the `if (opts.asyncOp) throw NotSupported(...)` guard in
     `SpyreCCLBackend::broadcast`.
   - Holds `at::Tensor` and the underlying `spyre_comms::Tensor` lifetime in
     `SpyreCCLWork` so they survive past the call return for true async
     execution.
   - Defers `work_schedule_->wait()` to `SpyreCCLWork::wait()` when
     `opts.asyncOp == true`.

2. **Split-broadcast routing**:
   - Replaces the `group_context_->broadcast(tensor, rootRank)` single-call
     with a try-block that:
     - calls `group_context_->broadcast(TensorInfo, rootRank)` to obtain a
       `WorkScheduleInfo`,
     - then `group_context_->broadcast_applyTensor(*info, tensor)` to obtain
       the `WorkSchedule`.
   - On any exception in that block (e.g. older ABI), falls back to the
     single-call form.
   - Adds a `SplitCacheKey` / `SplitCacheKeyHash` plus an
     `unordered_map<SplitCacheKey, shared_ptr<WorkScheduleInfo>>` cache to
     `SpyreCCLBackend` for future reuse (currently unused — see "Caveats").

## Build And Run Requirements

The patched torch-spyre links against a `libspyre_comms.so.1` that exposes
the split-collectives ABI: `Context::broadcast(const TensorInfo&, ...)` and
`Context::broadcast_applyTensor(const WorkScheduleInfo&, Tensor&)`. Use
spyre-comms commit `41278fa1` or any later commit that retains those public
symbols.

Build env on a Spyre dev pod:

```bash
# spyre-comms install path (provides include/spyre_comms.hpp + lib/libspyre_comms.so.1)
export SPYRE_COMMS_INSTALL_DIR=<path-to-spyre-comms-install>

# Spyre runtime + senlib + deeptools (system-installed at /opt/ibm/spyre/...)
export RUNTIME_INSTALL_DIR=/opt/ibm/spyre/runtime
export SENLIB_INSTALL_DIR=/opt/ibm/spyre/senlib
export DEEPTOOLS_INSTALL_DIR=/opt/ibm/spyre/deeptools
export SEN_COMMON_HEADERS=/opt/ibm/spyre/runtime/include

# pip install editable
pip install -e <path-to-tdeshane/split-broadcast-poc-checkout> \
    --no-build-isolation --no-deps --force-reinstall
```

Runtime env when launching torchrun (2-rank example):

```bash
unset TORCH_DEVICE_BACKEND_AUTOLOAD
export LD_LIBRARY_PATH="$SPYRE_COMMS_INSTALL_DIR/lib:..."
export PYTHONPATH="<path-to-checkout>:$PYTHONPATH"

# Required: spyre-comms 41278fa+ reads PCI addrs from AIU_WORLD_RANK_<n>
IFS=, read -r AIU_WORLD_RANK_0 AIU_WORLD_RANK_1 _rest <<< "$PCIDEVICE_IBM_COM_AIU_PF"
export AIU_WORLD_RANK_0 AIU_WORLD_RANK_1

torchrun --standalone --nproc-per-node 2 my_probe.py
```

## Caveats

1. **Cache is currently disabled.** The `broadcast_info_cache_` map is
   wired into the class but the active hot path constructs a fresh
   `WorkScheduleInfo` per call. Reusing a single `WorkScheduleInfo` across
   multiple `broadcast_applyTensor` calls trips an
   `InputSentinalEnvelope::setAddress` assertion in spyre-comms `41278fa`
   because the schedule's input address is set the first time and the
   second `applyTensor` tries to set a different address. If a future
   spyre-comms commit makes `applyTensor` reentrant on a single
   `WorkScheduleInfo`, flip the active body to use the cache map.
2. **Steady-state performance is identical** to the legacy single-call
   path on a 2-rank single-pod stack with broadcasts in the 64 KiB to
   8 MiB range. Per-call cost is dominated by something other than
   schedule planning. The split API's value is amortising the ~190 ms
   first-iteration cold cost; cache reuse (gated on the bug above) would
   amortise it across iterations.
3. **The patch enables compute/comm overlap measurements** but the actual
   overlap mechanism is the user-side N-way pipelining of async broadcasts
   (issuing N concurrent works and waiting all N), which works through
   either the legacy or split path. At N=128 with 1 MiB tensors and a 13 ms
   compute window, the configuration sees ~30% wall-clock reduction vs
   serial; below N=16 the saving is below noise.

## License

The patch sits on top of `torch-spyre`, which is Apache-2.0 licensed
(see `LICENSE.txt` in the upstream repo). The patch itself is offered
under the same license.

## Author

Todd Deshane <todd.deshane@ibm.com>, 2026-06-19.
