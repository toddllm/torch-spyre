"""Tensor-parallel toy transformer probe.

Realistic shape for compute/comm overlap measurement on inference.

A Megatron-style tensor-parallel transformer block has two collectives
per layer in the forward pass:
  - allreduce after the attention MLP output projection
  - allreduce after the FFN output projection

(In a 2-rank TP setup, each rank computes half of the output and the
allreduce sums them. Equivalent topologically to a broadcast of the
sum result; we use broadcast for simplicity since broadcast is what
our patched torch-spyre routes through the split-collectives API.)

This probe:
  1. builds a stack of N transformer-like layers (linear + relu + linear)
  2. simulates the per-layer comm by issuing 2 broadcasts of (hidden_dim,)
     fp16 tensors after each layer
  3. compares three execution modes:
       (a) blocking_serial: per-layer, run compute then run comm
       (b) async_layered:   per-layer, kick off comm async, run compute,
                            wait comm.  This is the "overlap" path that
                            real TP frameworks use.
       (c) async_batched:   issue all 2*N comms async at the start of
                            the step, run all N layers, then wait all
                            comms.  Tests whether collapsing collectives
                            to one big batch improves over per-layer
                            overlap.

Per-iteration:
  - rank 0 fills the comm buffers with deterministic data
  - both ranks run the linear-relu-linear compute window
  - correctness is verified for the comm buffers and a sentinel of the
    compute output

Output: per-rank JSONL + summary JSON, same shape as the prior probes.

Usage:
  torchrun --standalone --nproc-per-node 2 tp_transformer_probe.py \\
      --jsonl-out /tmp/tp.jsonl --summary-out /tmp/tp.json \\
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
import torch_spyre  # noqa: F401  registers spyre backend


def now() -> float:
    return time.perf_counter()


def emit(fh, row):
    fh.write(json.dumps(row, sort_keys=True) + "\n")
    fh.flush()


def fill_send(rank, buf, step):
    v = 0.5 + 0.001 * float(step % 64)
    if rank == 0:
        buf.fill_(v)
    else:
        buf.zero_()


def expected_sum(numel, step):
    return (0.5 + 0.001 * float(step % 64)) * float(numel)


def make_layer_compute(hidden_dim, intermediate_dim, iters, device):
    """Build a callable that simulates one transformer layer's compute window.

    Loosely models linear -> relu -> linear with `iters` repetitions of
    elementwise on a hidden_dim vector. We don't actually do matmul to
    avoid hitting register_torch_compile_kernel routes that need
    dxp_standalone for a fresh shape.
    """
    bias = torch.tensor(0.0009765625, dtype=torch.float16, device=device)
    scale = torch.tensor(1.0009765625, dtype=torch.float16, device=device)

    def fn(x):
        for _ in range(iters):
            x = (x + bias) * scale
            x = x + bias
        return x

    return fn


def run_iter(mode, *, rank, device, num_layers, hidden_dim, intermediate_dim,
             iters_per_layer, step, layer_compute,
             comm_bufs, hidden, out):
    """One iteration (= one TP forward step over `num_layers` layers)."""
    info = {"mode": mode}
    # 2 collectives per layer (attn output projection + FFN output projection)
    n_colls = 2 * num_layers

    if mode == "blocking_serial":
        # per-layer: compute then comm
        dist.barrier()
        t0 = now()
        x = hidden
        for layer in range(num_layers):
            x = layer_compute(x)
            # 2 sequential blocking broadcasts after the layer
            for i in range(2):
                ix = layer * 2 + i
                fill_send(rank, comm_bufs[ix], step + ix)
                dist.broadcast(comm_bufs[ix], src=0)
        out.copy_(x)
        os_ = float(out.detach().to("cpu").to(torch.float32).sum().item())
        # also verify a few comm buffers
        bs_sample = float(comm_bufs[0].detach().to("cpu").to(torch.float32).sum().item())
        bs_last = float(comm_bufs[n_colls-1].detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        info["iter_ms"] = (t1 - t0) * 1000.0
        info["correct_compute"] = (os_ == os_) and not math.isinf(os_)
        info["correct_broadcast"] = (
            abs(bs_sample - expected_sum(hidden_dim, step)) < max(1.0, hidden_dim * 1e-2)
            and abs(bs_last - expected_sum(hidden_dim, step + n_colls - 1)) < max(1.0, hidden_dim * 1e-2)
        )

    elif mode == "async_layered":
        # per-layer: kick off comm async, run compute on hidden, wait comm
        dist.barrier()
        t0 = now()
        x = hidden
        in_flight = []
        for layer in range(num_layers):
            # before layer's compute: drain any work from prior layer
            for w in in_flight: w.wait()
            in_flight = []
            # kick off this layer's 2 comms async
            for i in range(2):
                ix = layer * 2 + i
                fill_send(rank, comm_bufs[ix], step + ix)
                in_flight.append(dist.broadcast(comm_bufs[ix], src=0, async_op=True))
            # run compute on hidden, while comms are in flight
            x = layer_compute(x)
        # final drain
        for w in in_flight: w.wait()
        out.copy_(x)
        os_ = float(out.detach().to("cpu").to(torch.float32).sum().item())
        bs_sample = float(comm_bufs[0].detach().to("cpu").to(torch.float32).sum().item())
        bs_last = float(comm_bufs[n_colls-1].detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        info["iter_ms"] = (t1 - t0) * 1000.0
        info["correct_compute"] = (os_ == os_) and not math.isinf(os_)
        info["correct_broadcast"] = (
            abs(bs_sample - expected_sum(hidden_dim, step)) < max(1.0, hidden_dim * 1e-2)
            and abs(bs_last - expected_sum(hidden_dim, step + n_colls - 1)) < max(1.0, hidden_dim * 1e-2)
        )

    elif mode == "async_batched":
        # all 2*N comms in flight at once, then run all N layers, then wait all
        dist.barrier()
        t0 = now()
        # fill all comm buffers and kick off all 2*N comms
        for ix in range(n_colls):
            fill_send(rank, comm_bufs[ix], step + ix)
        works = [dist.broadcast(comm_bufs[ix], src=0, async_op=True)
                 for ix in range(n_colls)]
        # run all N layers' compute
        x = hidden
        for layer in range(num_layers):
            x = layer_compute(x)
        # wait all
        for w in works: w.wait()
        out.copy_(x)
        os_ = float(out.detach().to("cpu").to(torch.float32).sum().item())
        bs_sample = float(comm_bufs[0].detach().to("cpu").to(torch.float32).sum().item())
        bs_last = float(comm_bufs[n_colls-1].detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        info["iter_ms"] = (t1 - t0) * 1000.0
        info["correct_compute"] = (os_ == os_) and not math.isinf(os_)
        info["correct_broadcast"] = (
            abs(bs_sample - expected_sum(hidden_dim, step)) < max(1.0, hidden_dim * 1e-2)
            and abs(bs_last - expected_sum(hidden_dim, step + n_colls - 1)) < max(1.0, hidden_dim * 1e-2)
        )

    elif mode == "compute_only":
        dist.barrier()
        t0 = now()
        x = hidden
        for layer in range(num_layers):
            x = layer_compute(x)
        out.copy_(x)
        os_ = float(out.detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        info["iter_ms"] = (t1 - t0) * 1000.0
        info["correct_compute"] = (os_ == os_) and not math.isinf(os_)

    elif mode == "comm_only_blocking":
        dist.barrier()
        t0 = now()
        for ix in range(n_colls):
            fill_send(rank, comm_bufs[ix], step + ix)
            dist.broadcast(comm_bufs[ix], src=0)
        bs_sample = float(comm_bufs[0].detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        info["iter_ms"] = (t1 - t0) * 1000.0
        info["correct_broadcast"] = abs(bs_sample - expected_sum(hidden_dim, step)) < max(1.0, hidden_dim * 1e-2)

    return info


def run_config(*, fh, rank, device, num_layers, hidden_dim, intermediate_dim,
               iters_per_layer, warmup, timed):
    cfg = {"num_layers": num_layers, "hidden_dim": hidden_dim,
           "intermediate_dim": intermediate_dim, "iters_per_layer": iters_per_layer}
    layer_compute = make_layer_compute(hidden_dim, intermediate_dim, iters_per_layer, device)
    n_colls = 2 * num_layers
    comm_bufs = [torch.zeros((hidden_dim,), dtype=torch.float16, device=device)
                 for _ in range(n_colls)]
    hidden = torch.full((hidden_dim,), 0.5, dtype=torch.float16, device=device)
    out = torch.zeros((hidden_dim,), dtype=torch.float16, device=device)

    # warm comm + compute
    for s in range(warmup):
        fill_send(rank, comm_bufs[0], s); dist.broadcast(comm_bufs[0], src=0)
    for _ in range(warmup):
        _ = layer_compute(hidden)
        _ = float(out.detach().to("cpu").to(torch.float32).sum().item())

    for mode in ("compute_only", "comm_only_blocking",
                 "blocking_serial", "async_layered", "async_batched"):
        for it in range(timed):
            try:
                row_info = run_iter(mode, rank=rank, device=device,
                                    num_layers=num_layers, hidden_dim=hidden_dim,
                                    intermediate_dim=intermediate_dim,
                                    iters_per_layer=iters_per_layer,
                                    step=it * n_colls + 9000,
                                    layer_compute=layer_compute,
                                    comm_bufs=comm_bufs, hidden=hidden, out=out)
            except Exception as e:
                row_info = {"mode": mode, "error": f"{type(e).__name__}: {str(e).splitlines()[0]}"}
            emit(fh, {**cfg, "rank": rank, "iteration": it, **row_info})
        print(f"DONE rank={rank} mode={mode} L={num_layers} H={hidden_dim} iter_per_layer={iters_per_layer}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-out", required=True)
    p.add_argument("--summary-out", required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timed", type=int, default=10)
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"spyre:{local_rank}")
    dist.init_process_group(backend="spyreccl", device_id=device)
    dist.barrier()

    fh = open(f"{args.jsonl_out}.rank{rank}", "w", buffering=1)

    # Configurations: realistic small-LLM shapes
    # GPT-2-medium  : L=24, H=1024
    # Llama-7B       : L=32, H=4096 (we scale H down for speed of probe)
    # Llama-70B      : L=80, H=8192
    # Qwen-Coder-7B  : L=28, H=4096
    # Granite-3B     : L=40, H=2048
    configs = [
        # (num_layers, hidden_dim, intermediate_dim, iters_per_layer)
        (12, 1024, 4096, 8),     # small model, tiny per-layer compute
        (24, 1024, 4096, 8),     # GPT-2-medium-ish, small compute
        (32, 2048, 8192, 8),     # 7B-class (small), small compute
        (32, 4096, 16384, 8),    # 7B-class, more comm per layer
        (32, 8192, 32768, 8),    # 70B-shard-class
    ]
    for nl, hd, id_, it in configs:
        run_config(fh=fh, rank=rank, device=device,
                   num_layers=nl, hidden_dim=hd, intermediate_dim=id_,
                   iters_per_layer=it, warmup=args.warmup, timed=args.timed)
    fh.close()
    dist.barrier()

    if rank == 0:
        all_rows = []
        for r in range(world_size):
            try:
                with open(f"{args.jsonl_out}.rank{r}") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try: all_rows.append(json.loads(line))
                        except: pass
            except FileNotFoundError: pass
        from collections import defaultdict
        bucket = defaultdict(list)
        for row in all_rows:
            if row.get("error") or "iteration" not in row: continue
            key = (row["num_layers"], row["hidden_dim"], row["mode"], row["rank"])
            bucket[key].append(row)
        configs_summary = []
        for (nl, hd, id_, it) in configs:
            entry = {"num_layers": nl, "hidden_dim": hd,
                     "intermediate_dim": id_, "iters_per_layer": it,
                     "modes": {}}
            for mode in ("compute_only", "comm_only_blocking",
                         "blocking_serial", "async_layered", "async_batched"):
                per_mode = {}
                for rk in (0, 1):
                    rows = bucket.get((nl, hd, mode, rk), [])
                    if not rows:
                        per_mode[f"rank{rk}"] = None; continue
                    iters_list = [r["iter_ms"] for r in rows if "iter_ms" in r]
                    if not iters_list:
                        per_mode[f"rank{rk}"] = None; continue
                    ss = iters_list[2:] if len(iters_list) > 2 else iters_list
                    correct_b = [r.get("correct_broadcast") for r in rows if r.get("correct_broadcast") is not None]
                    correct_c = [r.get("correct_compute") for r in rows if r.get("correct_compute") is not None]
                    per_mode[f"rank{rk}"] = {
                        "all_count": len(iters_list),
                        "all_median": statistics.median(iters_list),
                        "ss_count": len(ss),
                        "ss_median": statistics.median(ss),
                        "all_correct_broadcast": all(correct_b) if correct_b else None,
                        "all_correct_compute": all(correct_c) if correct_c else None,
                    }
                entry["modes"][mode] = per_mode
            configs_summary.append(entry)
        summary = {"configs": configs_summary, "warmup_iters": args.warmup, "timed_iters": args.timed}
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("SUMMARY written", flush=True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
