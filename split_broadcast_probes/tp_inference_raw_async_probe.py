"""TP=2 with raw async broadcasts (bypassing SpyreCommunicator wrapper).

Same vllm distributed init, but instead of calling
`tensor_model_parallel_all_reduce` (which is sync inside SpyreCommunicator's
manual fallback), this uses `dist.broadcast(..., async_op=True)` against the
device_group directly. Each "allreduce" becomes a 2-broadcast pattern but
they can be in flight concurrently.

NOTE: this gives wrong numerical results — broadcast doesn't add — but
the timing measurement is valid for determining whether async overlap
is achievable on the inference path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist


def now() -> float:
    return time.perf_counter()


def emit(fh, row):
    fh.write(json.dumps(row, sort_keys=True) + "\n")
    fh.flush()


def run_iter(mode, *, rank, device, num_layers, hidden_size, iters_per_layer,
             hidden, device_group):
    info = {"mode": mode, "num_layers": num_layers, "hidden_size": hidden_size}
    bias = torch.tensor(0.0009765625, dtype=torch.float16, device=device)
    scale = torch.tensor(1.0009765625, dtype=torch.float16, device=device)

    def _compute(x):
        for _ in range(iters_per_layer):
            x = (x + bias) * scale
            x = x + bias
        return x

    n_colls = 2 * num_layers

    if mode == "compute_only":
        dist.barrier(group=device_group, device_ids=[device.index])
        t0 = now()
        x = hidden
        for _ in range(num_layers):
            x = _compute(x)
        out_sum = float(x.detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        info["iter_ms"] = (t1 - t0) * 1000.0
        info["correct_compute"] = (out_sum == out_sum) and not math.isinf(out_sum)
    elif mode == "comm_only_blocking_raw_bcast":
        # Each iteration issues 2N blocking broadcasts on the device_group
        dist.barrier(group=device_group, device_ids=[device.index])
        t0 = now()
        for layer in range(num_layers):
            for i in range(2):
                t = torch.full((hidden_size,), float(rank + 1), dtype=torch.float16, device=device)
                dist.broadcast(t, src=0, group=device_group)
        t1 = now()
        info["iter_ms"] = (t1 - t0) * 1000.0
    elif mode == "blocking_serial_raw_bcast":
        # Per-layer compute then 2 blocking broadcasts (analog of blocking_serial
        # but using raw broadcasts not allreduce)
        dist.barrier(group=device_group, device_ids=[device.index])
        t0 = now()
        x = hidden
        for layer in range(num_layers):
            x = _compute(x)
            for i in range(2):
                t = torch.full((hidden_size,), float(rank + 1), dtype=torch.float16, device=device)
                dist.broadcast(t, src=0, group=device_group)
        out_sum = float(x.detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        info["iter_ms"] = (t1 - t0) * 1000.0
        info["correct_compute"] = (out_sum == out_sum) and not math.isinf(out_sum)
    elif mode == "async_batched_raw_bcast":
        # Issue all 2N async broadcasts upfront, run all compute, wait all
        dist.barrier(group=device_group, device_ids=[device.index])
        t0 = now()
        bufs = []
        works = []
        for layer in range(num_layers):
            for i in range(2):
                t = torch.full((hidden_size,), float(rank + 1), dtype=torch.float16, device=device)
                bufs.append(t)
                works.append(dist.broadcast(t, src=0, group=device_group, async_op=True))
        x = hidden
        for layer in range(num_layers):
            x = _compute(x)
        for w in works:
            w.wait()
        out_sum = float(x.detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        info["iter_ms"] = (t1 - t0) * 1000.0
        info["correct_compute"] = (out_sum == out_sum) and not math.isinf(out_sum)
    else:
        raise ValueError(mode)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-out", required=True)
    p.add_argument("--summary-out", required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timed", type=int, default=10)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--iters-per-layer", type=int, default=8)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    import torch_spyre  # noqa: F401
    torch.spyre.set_device(local_rank)

    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

    load_general_plugins()
    cfg = EngineArgs(
        model="facebook/opt-125m",
        tensor_parallel_size=world_size,
        dtype="float16",
        enforce_eager=True,
        distributed_executor_backend="external_launcher",
    ).create_engine_config()

    with set_current_vllm_config(cfg):
        init_worker_distributed_environment(
            cfg,
            rank,
            distributed_init_method="env://",
            local_rank=local_rank,
            backend=current_platform.dist_backend,
        )
        import vllm.distributed.parallel_state as ps
        device_group = ps._TP.device_group

        device = torch.device(f"spyre:{local_rank}")
        hidden = torch.full((args.hidden_size,), 0.5, dtype=torch.float16, device=device)

        # Warmup
        for _ in range(args.warmup):
            t = torch.full((args.hidden_size,), float(rank + 1), dtype=torch.float16, device=device)
            dist.broadcast(t, src=0, group=device_group)

        fh = open(f"{args.jsonl_out}.rank{rank}", "w", buffering=1)
        cfg_meta = {
            "num_layers": args.num_layers, "hidden_size": args.hidden_size,
            "iters_per_layer": args.iters_per_layer, "world_size": world_size,
        }
        for mode in ("compute_only", "comm_only_blocking_raw_bcast",
                     "blocking_serial_raw_bcast", "async_batched_raw_bcast"):
            for it in range(args.timed):
                row_info = run_iter(
                    mode, rank=rank, device=device,
                    num_layers=args.num_layers, hidden_size=args.hidden_size,
                    iters_per_layer=args.iters_per_layer,
                    hidden=hidden, device_group=device_group,
                )
                emit(fh, {**cfg_meta, "rank": rank, "iteration": it, **row_info})
            print(f"DONE rank={rank} mode={mode}", flush=True)
        fh.close()
        dist.barrier(group=device_group, device_ids=[device.index])

        if rank == 0:
            all_rows = []
            for r in range(world_size):
                with open(f"{args.jsonl_out}.rank{r}") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_rows.append(json.loads(line))
            from collections import defaultdict
            bucket = defaultdict(list)
            for row in all_rows:
                key = (row["mode"], row["rank"])
                bucket[key].append(row["iter_ms"])
            summary = {"config": cfg_meta, "modes": {}}
            for mode in ("compute_only", "comm_only_blocking_raw_bcast",
                         "blocking_serial_raw_bcast", "async_batched_raw_bcast"):
                per_mode = {}
                for rk in (0, 1):
                    iters_list = bucket.get((mode, rk), [])
                    if not iters_list:
                        per_mode[f"rank{rk}"] = None
                        continue
                    ss = iters_list[2:] if len(iters_list) > 2 else iters_list
                    per_mode[f"rank{rk}"] = {
                        "ss_count": len(ss),
                        "ss_median": statistics.median(ss),
                    }
                summary["modes"][mode] = per_mode
            with open(args.summary_out, "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            print("SUMMARY written", flush=True)

        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
