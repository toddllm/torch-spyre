"""Very large N at 1MB tensors."""
import argparse, json, math, os, statistics, sys, time
import torch, torch.distributed as dist, torch_spyre

def now(): return time.perf_counter()

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
    if rank == 0: buf.fill_(v)
    else: buf.zero_()

p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--warmup", type=int, default=3); p.add_argument("--timed", type=int, default=10)
args = p.parse_args()
rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"]); world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"spyre:{local_rank}")
dist.init_process_group(backend="spyreccl", device_id=device); dist.barrier()

# Fix numel=1MB, iters=64. Sweep N.
NUMEL = 1048576
ITERS = 64
fn = make_compute(NUMEL, ITERS, device)
buf_c = torch.full((NUMEL,), 0.5, dtype=torch.float16, device=device)
out = torch.zeros((NUMEL,), dtype=torch.float16, device=device)

results = {}
for N in (4, 8, 16, 32):
    bufs = [torch.zeros((NUMEL,), dtype=torch.float16, device=device) for _ in range(N)]
    # warmup
    for s in range(args.warmup):
        fill_send(rank, bufs[0], s); dist.broadcast(bufs[0], src=0)
    for _ in range(args.warmup):
        _ = fn(buf_c); _ = float(out.detach().to("cpu").to(torch.float32).sum().item())

    times = {"n_blocking": [], "compute_only": [], "n_async_compute": []}
    for it in range(args.timed):
        # blocking
        for i in range(N): fill_send(rank, bufs[i], it*N + 9000 + i)
        dist.barrier()
        t0 = now()
        for i in range(N): dist.broadcast(bufs[i], src=0)
        t1 = now()
        times["n_blocking"].append(1000*(t1-t0))
        # compute only
        dist.barrier()
        t0 = now()
        y = fn(buf_c)
        out.copy_(y)
        _ = float(out.detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        times["compute_only"].append(1000*(t1-t0))
        # n_async_compute
        for i in range(N): fill_send(rank, bufs[i], it*N + 9000 + i)
        dist.barrier()
        t0 = now()
        works = [dist.broadcast(bufs[i], src=0, async_op=True) for i in range(N)]
        y = fn(buf_c)
        for w in works: w.wait()
        out.copy_(y)
        _ = float(out.detach().to("cpu").to(torch.float32).sum().item())
        t1 = now()
        times["n_async_compute"].append(1000*(t1-t0))

    results[N] = {k: statistics.median(v[2:]) if len(v) > 2 else statistics.median(v) for k,v in times.items()}
    print(f"rank={rank} N={N}: blocking={results[N]['n_blocking']:.2f} compute={results[N]['compute_only']:.2f} async_compute={results[N]['n_async_compute']:.2f}", flush=True)

dist.barrier()
if rank == 0:
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
dist.destroy_process_group()
