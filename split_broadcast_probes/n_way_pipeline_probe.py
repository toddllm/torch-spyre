"""N-way pipeline probe.

Issues N async broadcasts back to back, runs one compute window,
then waits all N. Tests how per-call overhead scales: linear (each
broadcast costs the full launch+wait) or sub-linear (host-side
parallelism amortizes the launches).

For each (numel, N) configuration, three modes:

  1. n_blocking : N back-to-back blocking broadcasts (no compute).
                  Establishes the linear-cost baseline.
  2. n_async    : N async broadcasts, no compute, then wait all N.
                  Tests pure host-side launch parallelism.
  3. n_async_compute : N async broadcasts + 1 compute + wait all N.
                       Tests whether compute can run concurrently
                       with N in-flight broadcasts.

Steady-state median of 25 timed iters (drop first 2). Compute is the
same Spyre eager (x+bias)*scale loop used in P5/P6.

Today we know N=2 works at ~0.10 ms per extra launch (P6). This
probe finds where that breaks down. Specifically, if 10 broadcasts
cost 1.0 ms (linear scaling) the API split would help; if 10 cost
0.3 ms (heavy sub-linear scaling) the host-side path is already
near-optimal.
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


def run_iter(mode, *, rank, device, numel, step, fn, bufs, buf_c, out, N):
    info = {"mode": mode, "N": N}
    if mode == "n_blocking":
        for i in range(N):
            fill_send(rank, bufs[i], step + i)
        dist.barrier()
        t0 = now()
        for i in range(N):
            dist.broadcast(bufs[i], src=0)
        t1 = now()
        # readback + verify all
        all_ok = True
        for i in range(N):
            s = float(bufs[i].detach().to("cpu").to(torch.float32).sum().item())
            if abs(s - expected_sum(numel, step + i)) > max(1.0, numel * 1e-2):
                all_ok = False
        t2 = now()
        info["bcasts_ms"]   = (t1 - t0) * 1000.0
        info["readback_ms"] = (t2 - t1) * 1000.0
        info["iter_ms"]     = (t2 - t0) * 1000.0
        info["correct_broadcast"] = all_ok
    elif mode == "n_async":
        for i in range(N):
            fill_send(rank, bufs[i], step + i)
        dist.barrier()
        t0 = now()
        works = []
        for i in range(N):
            works.append(dist.broadcast(bufs[i], src=0, async_op=True))
        t1 = now()
        for w in works:
            w.wait()
        t2 = now()
        all_ok = True
        for i in range(N):
            s = float(bufs[i].detach().to("cpu").to(torch.float32).sum().item())
            if abs(s - expected_sum(numel, step + i)) > max(1.0, numel * 1e-2):
                all_ok = False
        t3 = now()
        info["launches_ms"] = (t1 - t0) * 1000.0
        info["waits_ms"]    = (t2 - t1) * 1000.0
        info["readback_ms"] = (t3 - t2) * 1000.0
        info["iter_ms"]     = (t3 - t0) * 1000.0
        info["correct_broadcast"] = all_ok
    elif mode == "n_async_compute":
        for i in range(N):
            fill_send(rank, bufs[i], step + i)
        dist.barrier()
        t0 = now()
        works = []
        for i in range(N):
            works.append(dist.broadcast(bufs[i], src=0, async_op=True))
        t1 = now()
        y = fn(buf_c)
        t2 = now()
        for w in works:
            w.wait()
        t3 = now()
        all_ok = True
        for i in range(N):
            s = float(bufs[i].detach().to("cpu").to(torch.float32).sum().item())
            if abs(s - expected_sum(numel, step + i)) > max(1.0, numel * 1e-2):
                all_ok = False
        out.copy_(y)
        os_ = float(out.detach().to("cpu").to(torch.float32).sum().item())
        t4 = now()
        info["launches_ms"] = (t1 - t0) * 1000.0
        info["compute_ms"]  = (t2 - t1) * 1000.0
        info["waits_ms"]    = (t3 - t2) * 1000.0
        info["readback_ms"] = (t4 - t3) * 1000.0
        info["iter_ms"]     = (t4 - t0) * 1000.0
        info["correct_broadcast"] = all_ok
        info["correct_compute"]   = (os_ == os_) and not math.isinf(os_)
    else:
        raise ValueError(f"unknown mode {mode}")
    return info


def run_config(*, fh, rank, device, numel, iters, N, warmup, timed):
    cfg = {"numel": numel, "iters": iters, "N": N}
    fn = make_compute(numel, iters, device)
    bufs = [torch.zeros((numel,), dtype=torch.float16, device=device) for _ in range(N)]
    buf_c = torch.full((numel,), 0.5, dtype=torch.float16, device=device)
    out = torch.zeros((numel,), dtype=torch.float16, device=device)
    for s in range(warmup):
        fill_send(rank, bufs[0], s)
        dist.broadcast(bufs[0], src=0)
    for _ in range(warmup):
        _ = fn(buf_c)
        _ = float(out.detach().to("cpu").to(torch.float32).sum().item())
    for mode in ("n_blocking", "n_async", "n_async_compute"):
        for it in range(timed):
            try:
                step = it * N + 9000
                row_info = run_iter(mode, rank=rank, device=device,
                                    numel=numel, step=step, fn=fn,
                                    bufs=bufs, buf_c=buf_c, out=out, N=N)
            except Exception as exc:
                row_info = {"mode": mode, "N": N,
                            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"}
            emit(fh, {**cfg, "rank": rank, "iteration": it, **row_info})
        print(f"DONE rank={rank} mode={mode} N={N} numel={numel}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-out", required=True)
    p.add_argument("--summary-out", required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timed", type=int, default=20)
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"spyre:{local_rank}")
    dist.init_process_group(backend="spyreccl", device_id=device)
    dist.barrier()

    fh = open(f"{args.jsonl_out}.rank{rank}", "w", buffering=1)
    # Sweep N to see how host-side launches scale.
    # Fix numel and iters at the "prior probes" config.
    configs = [
        (65536, 32, 1),
        (65536, 32, 2),
        (65536, 32, 4),
        (65536, 32, 8),
        (65536, 32, 16),
    ]
    for numel, iters, N in configs:
        run_config(fh=fh, rank=rank, device=device, numel=numel, iters=iters,
                   N=N, warmup=args.warmup, timed=args.timed)
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
                        except Exception: pass
            except FileNotFoundError: pass
        from collections import defaultdict
        bucket = defaultdict(list)
        for row in all_rows:
            if row.get("error") or "iteration" not in row: continue
            key = (row["numel"], row["iters"], row["N"], row["mode"], row["rank"])
            bucket[key].append(row)
        configs_summary = []
        for (numel, iters, N) in configs:
            entry = {"numel": numel, "iters": iters, "N": N, "modes": {}}
            for mode in ("n_blocking", "n_async", "n_async_compute"):
                per_mode = {}
                for rk in (0, 1):
                    rows = bucket.get((numel, iters, N, mode, rk), [])
                    if not rows:
                        per_mode[f"rank{rk}"] = None; continue
                    iters_list = [r["iter_ms"] for r in rows if "iter_ms" in r]
                    if not iters_list:
                        per_mode[f"rank{rk}"] = None; continue
                    ss = iters_list[2:] if len(iters_list) > 2 else iters_list
                    per_mode[f"rank{rk}"] = {
                        "all_count": len(iters_list),
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
