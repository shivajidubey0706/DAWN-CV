"""Cost protocol (paper Sec. V-D, experiment E7): params, FLOPs, latency, FPS, memory.

FLOPs convention: torch.utils.flop_counter.FlopCounterMode, which counts 2 FLOPs per
multiply-accumulate for convolutions and matmuls and does not count element-wise
ops, pooling, interpolation or searchsorted. GMACs = GFLOPs / 2.
Latency: batch 1, warm-up then timed iterations with device synchronisation;
median and 90th percentile; FPS = 1000 / median_ms.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def count_flops(fn, *args) -> int | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            fn(*args)
        return int(fc.get_total_flops())
    except Exception:  # noqa: BLE001
        return None


def time_fn(fn, x: torch.Tensor, warmup: int = 50, iters: int = 500, half: bool = False) -> dict:
    dev = x.device
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    ctx = torch.autocast("cuda", dtype=torch.float16) if half and dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad(), ctx:
        for _ in range(warmup):
            fn(x)
        sync()
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(x)
            sync()
            times.append((time.perf_counter() - t0) * 1e3)
    t = np.array(times)
    out = {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)),
           "fps": float(1000 / np.median(t)), "warmup": warmup, "iters": iters,
           "precision": "fp16-autocast" if half and dev.type == "cuda" else "fp32"}
    if dev.type == "cuda":
        out["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return out
