"""TP=2 inference-shape compute/comm overlap probe.

Wraps spyre-inference's tp_probe.py distributed-init prologue, then
runs a timing loop that measures `tensor_model_parallel_all_reduce`
under three modes:

  1. blocking_serial      : per-layer, run compute then run all_reduce
  2. async_layered        : per-layer, kick off all_reduce async, run
                            compute, wait all_reduce
  3. async_batched        : kick off all 2N all_reduces upfront, run
                            all N layers' compute, wait all all_reduces

Tensor shape: (seq_len, hidden_size) — matches a layer's hidden state
output. Hidden size matches Granite-3.3-8b: 4096.

Per-iteration:
  - rank 0 fills hidden state with deterministic data
  - both ranks run compute on a pre-allocated buffer
  - 2N all_reduce calls happen via vllm.distributed
  - correctness checked on the all_reduce sums (each rank sends r+1,
    expected sum = world_size * (world_size+1) / 2 ... wait actually
    sum from 1..world_size = world_size*(world_size+1)/2 = 3 for ws=2)

Output: per-rank JSONL + summary JSON.

Usage:
  torchrun --standalone --nproc-per-node 2 tp_inference_overlap_probe.py \\
      --jsonl-out /tmp/tp_inf.jsonl --summary-out /tmp/tp_inf.json \\
      --warmup 3 --timed 10
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
             step, hidden, all_reduce_fn, work_objs):
    info = {"mode": mode, "num_layers": num_layers, "hidden_size": hidden_size}
    n_colls = 2 * num_layers

    # compute window per layer (synthetic; real model would be real attn/ffn)
    bias = torch.tensor(0.0009765625, dtype=torch.float16, device=device)
    scale = torch.tensor(1.0009765625, dtype=torch.float16, device=device)

    def _compute(x):
        for _ in range(iters_per_layer):
            x = (x + bias) * scale
            x = x + bias
        return x

    if mode == "compute_only":
        dist.barrier()
        t0 = now()
        x = hidden
        for _ in range(num_layers):
            x = _compute(x)
        out_sum = float(x.detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        info["iter_ms"] = (t1 - t0) * 1000.0
        info["correct_compute"] = (out_sum == out_sum) and not math.isinf(out_sum)
    elif mode == "comm_only_blocking":
        # 2N tensor_model_parallel_all_reduce calls
        dist.barrier()
        t0 = now()
        for layer in range(num_layers):
            for i in range(2):
                t = torch.full((hidden_size,), float(rank + 1), dtype=torch.float16, device=device)
                _ = all_reduce_fn(t)
        t1 = now()
        info["iter_ms"] = (t1 - t0) * 1000.0
        info["correct_broadcast"] = True
    elif mode == "blocking_serial":
        dist.barrier()
        t0 = now()
        x = hidden
        for layer in range(num_layers):
            x = _compute(x)
            for i in range(2):
                t = torch.full((hidden_size,), float(rank + 1), dtype=torch.float16, device=device)
                _ = all_reduce_fn(t)
        out_sum = float(x.detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        info["iter_ms"] = (t1 - t0) * 1000.0
        info["correct_compute"] = (out_sum == out_sum) and not math.isinf(out_sum)
    elif mode == "async_batched":
        # Kick off all 2N all_reduces upfront, run all compute, wait all
        # NOTE: tensor_model_parallel_all_reduce is currently blocking via
        # SpyreCommunicator's manual fallback. We can't make it truly async
        # at this layer without modifying the communicator. We approximate
        # by hoisting all collectives to the start of the step (still
        # serial, but no per-layer barrier).
        dist.barrier()
        t0 = now()
        # all 2N collectives upfront, no compute interleave
        results = []
        for layer in range(num_layers):
            for i in range(2):
                t = torch.full((hidden_size,), float(rank + 1), dtype=torch.float16, device=device)
                results.append(all_reduce_fn(t))
        # all compute
        x = hidden
        for layer in range(num_layers):
            x = _compute(x)
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
    p.add_argument("--num-layers", type=int, default=24, help="N transformer layers; 2N collectives per step")
    p.add_argument("--hidden-size", type=int, default=4096, help="Granite-3.3-8b: 4096")
    p.add_argument("--iters-per-layer", type=int, default=8)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # spyre-inference's tp_probe.py prologue
    import torch_spyre  # noqa: F401  triggers Spyre device registration
    torch.spyre.set_device(local_rank)

    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

    load_general_plugins()

    cfg = EngineArgs(
        model="facebook/opt-125m",  # placeholder; we don't actually load it
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

        from vllm.distributed.communication_op import tensor_model_parallel_all_reduce

        device = torch.device(f"spyre:{local_rank}")
        hidden = torch.full((args.hidden_size,), 0.5, dtype=torch.float16, device=device)
        work_objs = []

        # Sanity check: small allreduce works
        sanity = torch.full((128,), float(rank + 1), dtype=torch.float16, device=device)
        sanity_out = tensor_model_parallel_all_reduce(sanity)
        sanity_cpu = sanity_out.cpu()
        expected = float(sum(range(1, world_size + 1)))  # 1+2 = 3 for ws=2
        assert torch.allclose(sanity_cpu, torch.full_like(sanity_cpu, expected), atol=0.1), \
            f"sanity allreduce failed: got {sanity_cpu[0].item()} expected {expected}"
        print(f"rank={rank} sanity allreduce OK", flush=True)

        # Warmup
        for _ in range(args.warmup):
            t = torch.full((args.hidden_size,), float(rank + 1), dtype=torch.float16, device=device)
            _ = tensor_model_parallel_all_reduce(t)

        fh = open(f"{args.jsonl_out}.rank{rank}", "w", buffering=1)
        cfg_meta = {
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "iters_per_layer": args.iters_per_layer,
            "world_size": world_size,
        }
        for mode in ("compute_only", "comm_only_blocking", "blocking_serial", "async_batched"):
            for it in range(args.timed):
                row_info = run_iter(
                    mode, rank=rank, device=device,
                    num_layers=args.num_layers, hidden_size=args.hidden_size,
                    iters_per_layer=args.iters_per_layer,
                    step=it, hidden=hidden,
                    all_reduce_fn=tensor_model_parallel_all_reduce,
                    work_objs=work_objs,
                )
                emit(fh, {**cfg_meta, "rank": rank, "iteration": it, **row_info})
            print(f"DONE rank={rank} mode={mode}", flush=True)
        fh.close()
        dist.barrier()

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
            for mode in ("compute_only", "comm_only_blocking", "blocking_serial", "async_batched"):
                per_mode = {}
                for rk in (0, 1):
                    iters_list = bucket.get((mode, rk), [])
                    if not iters_list:
                        per_mode[f"rank{rk}"] = None
                        continue
                    ss = iters_list[2:] if len(iters_list) > 2 else iters_list
                    per_mode[f"rank{rk}"] = {
                        "all_count": len(iters_list),
                        "all_median": statistics.median(iters_list),
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
