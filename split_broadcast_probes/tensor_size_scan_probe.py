"""Tensor-size scan probe.

Answers: at what tensor size does broadcast time start to approach
compute time on the current 2-rank single-pod stack? Reuses the same
build environment as P5/P6 (no rebuild required); just sweeps numel
and iters across a much wider range than the prior probes.

For each (numel, iters) pair, runs:
  1. comm_only_blocking : blocking broadcast + readback, no compute.
                          Measures pure broadcast cost as a function
                          of tensor size.
  2. compute_only       : Spyre eager compute on a separate buffer,
                          no broadcast.
  3. blocking_pipeline  : blocking broadcast + compute. Establishes
                          baseline e2e.
  4. single_inflight    : async broadcast + compute + wait.

Steady-state median (drop first 2 timed iters). 5 warmup + 15 timed
to keep total run time bounded with 5 sizes x 4 modes x 2 ranks =
40 mode-config-rank combinations.

The size sweep:
  64 KiB    (16384 fp16  = numel 16K)  - small, comm dominated by overhead
  256 KiB   (65536 fp16)               - prior probes baseline
  1 MiB     (262144 fp16)              - prior probes' largest
  4 MiB     (1048576 fp16)             - new
  16 MiB    (4194304 fp16)             - new

If broadcast remains <5% of e2e through 16 MiB, the conclusion that
overlap is uninteresting at single-pod scale extends to a much wider
range. If it grows past ~30%, we have a regime where the split API
(or Lane C concurrency) starts to give back time.
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
import torch_spyre  # noqa: F401


def now() -> float:
    return time.perf_counter()


def emit(fh, row):
    fh.write(json.dumps(row, sort_keys=True) + "\n")
    fh.flush()


def make_compute(numel, iters, device):
    bias = torch.tensor(0.0009765625, dtype=torch.float16, device=device)
    scale = torch.tensor(1.0009765625, dtype=torch.float16, device=device)
    def fn(buf):
        x = buf
        for _ in range(iters):
            x = (x + bias) * scale
            x = x + bias
        return x
    return fn


def fill_send(rank, buf, step):
    v = 0.5 + 0.001 * float(step % 64)
    if rank == 0:
        buf.fill_(v)
    else:
        buf.zero_()


def expected_sum(numel, step):
    return (0.5 + 0.001 * float(step % 64)) * float(numel)


def run_iter(mode, *, rank, device, numel, step, fn, buf_a, buf_c, out):
    info = {"mode": mode}
    if mode == "comm_only_blocking":
        fill_send(rank, buf_a, step)
        dist.barrier()
        t0 = now()
        dist.broadcast(buf_a, src=0)
        s = float(buf_a.detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        info["bcast_ms"] = (t1 - t0) * 1000.0
        info["iter_ms"]  = info["bcast_ms"]
        info["correct_broadcast"] = abs(s - expected_sum(numel, step)) < max(1.0, numel * 1e-2)
    elif mode == "compute_only":
        dist.barrier()
        t0 = now()
        y = fn(buf_c)
        out.copy_(y)
        os_ = float(out.detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        info["compute_ms"] = (t1 - t0) * 1000.0
        info["iter_ms"]    = info["compute_ms"]
        info["correct_compute"] = (os_ == os_) and not math.isinf(os_)
    elif mode == "blocking_pipeline":
        fill_send(rank, buf_a, step)
        dist.barrier()
        t0 = now()
        dist.broadcast(buf_a, src=0)
        t1 = now()
        y = fn(buf_c)
        t2 = now()
        s = float(buf_a.detach().to("cpu").to(torch.float32).sum().item())
        out.copy_(y)
        os_ = float(out.detach().to("cpu").to(torch.float32).sum().item())
        t3 = now()
        info["bcast_ms"]    = (t1 - t0) * 1000.0
        info["compute_ms"]  = (t2 - t1) * 1000.0
        info["readback_ms"] = (t3 - t2) * 1000.0
        info["iter_ms"]     = (t3 - t0) * 1000.0
        info["correct_broadcast"] = abs(s - expected_sum(numel, step)) < max(1.0, numel * 1e-2)
        info["correct_compute"]   = (os_ == os_) and not math.isinf(os_)
    elif mode == "single_inflight":
        fill_send(rank, buf_a, step)
        dist.barrier()
        t0 = now()
        work = dist.broadcast(buf_a, src=0, async_op=True)
        t1 = now()
        y = fn(buf_c)
        t2 = now()
        work.wait()
        t3 = now()
        s = float(buf_a.detach().to("cpu").to(torch.float32).sum().item())
        out.copy_(y)
        os_ = float(out.detach().to("cpu").to(torch.float32).sum().item())
        t4 = now()
        info["launch1_ms"]  = (t1 - t0) * 1000.0
        info["compute_ms"]  = (t2 - t1) * 1000.0
        info["wait1_ms"]    = (t3 - t2) * 1000.0
        info["readback_ms"] = (t4 - t3) * 1000.0
        info["iter_ms"]     = info["launch1_ms"] + info["compute_ms"] \
                            + info["wait1_ms"] + info["readback_ms"]
        info["correct_broadcast"] = abs(s - expected_sum(numel, step)) < max(1.0, numel * 1e-2)
        info["correct_compute"]   = (os_ == os_) and not math.isinf(os_)
    else:
        raise ValueError(f"unknown mode {mode}")
    return info


def run_config(*, fh, rank, device, numel, iters, warmup, timed):
    cfg = {"numel": numel, "iters": iters}
    fn = make_compute(numel, iters, device)
    buf_a = torch.zeros((numel,), dtype=torch.float16, device=device)
    buf_c = torch.full((numel,), 0.5, dtype=torch.float16, device=device)
    out = torch.zeros((numel,), dtype=torch.float16, device=device)
    for s in range(warmup):
        fill_send(rank, buf_a, s)
        dist.broadcast(buf_a, src=0)
    for _ in range(warmup):
        _ = fn(buf_c)
        _ = float(out.detach().to("cpu").to(torch.float32).sum().item())
    for mode in ("comm_only_blocking", "compute_only",
                 "blocking_pipeline", "single_inflight"):
        for it in range(timed):
            try:
                row_info = run_iter(mode, rank=rank, device=device,
                                    numel=numel, step=it + 9000, fn=fn,
                                    buf_a=buf_a, buf_c=buf_c, out=out)
            except Exception as exc:
                row_info = {"mode": mode, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"}
            emit(fh, {**cfg, "rank": rank, "iteration": it, **row_info})
        print(f"DONE rank={rank} mode={mode} numel={numel} iters={iters}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-out", required=True)
    p.add_argument("--summary-out", required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timed", type=int, default=15)
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"spyre:{local_rank}")
    dist.init_process_group(backend="spyreccl", device_id=device)
    dist.barrier()

    fh = open(f"{args.jsonl_out}.rank{rank}", "w", buffering=1)
    configs = [
        (16384,   32),    # 32 KiB
        (65536,   32),    # 128 KiB (prior probes baseline)
        (262144,  32),    # 512 KiB (prior probes' largest)
        (1048576, 32),    # 2 MiB
        (4194304, 32),    # 8 MiB
    ]
    for numel, iters in configs:
        run_config(fh=fh, rank=rank, device=device, numel=numel, iters=iters,
                   warmup=args.warmup, timed=args.timed)
    fh.close()
    dist.barrier()

    if rank == 0:
        all_rows = []
        for r in range(world_size):
            try:
                with open(f"{args.jsonl_out}.rank{r}") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            all_rows.append(json.loads(line))
                        except Exception:
                            pass
            except FileNotFoundError:
                pass
        from collections import defaultdict
        bucket = defaultdict(list)
        for row in all_rows:
            if row.get("error") or "iteration" not in row:
                continue
            key = (row["numel"], row["iters"], row["mode"], row["rank"])
            bucket[key].append(row)
        configs_summary = []
        for (numel, iters) in configs:
            entry = {"numel": numel, "iters": iters, "modes": {}}
            for mode in ("comm_only_blocking", "compute_only",
                         "blocking_pipeline", "single_inflight"):
                per_mode = {}
                for rk in (0, 1):
                    rows = bucket.get((numel, iters, mode, rk), [])
                    if not rows:
                        per_mode[f"rank{rk}"] = None
                        continue
                    iters_list = [r["iter_ms"] for r in rows if "iter_ms" in r]
                    if not iters_list:
                        per_mode[f"rank{rk}"] = None
                        continue
                    ss = iters_list[2:] if len(iters_list) > 2 else iters_list
                    per_mode[f"rank{rk}"] = {
                        "all_count": len(iters_list),
                        "all_min": min(iters_list),
                        "all_max": max(iters_list),
                        "all_median": statistics.median(iters_list),
                        "ss_count": len(ss),
                        "ss_median": statistics.median(ss),
                    }
                entry["modes"][mode] = per_mode
            configs_summary.append(entry)
        summary = {"configs": configs_summary,
                   "warmup_iters": args.warmup,
                   "timed_iters": args.timed}
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("SUMMARY written", flush=True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
