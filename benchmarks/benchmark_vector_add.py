#!/usr/bin/env python3
"""Measure PyTorch vector addition with warmup, synchronization and statistics."""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--n", type=int, default=1 << 24, help="number of float elements")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(samples_ms: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p50_ms": percentile(samples_ms, 0.50),
        "p95_ms": percentile(samples_ms, 0.95),
        "p99_ms": percentile(samples_ms, 0.99),
        "std_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "max_ms": max(samples_ms),
    }


def main() -> None:
    args = parse_args()
    if args.n <= 0 or args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--n and --iters must be positive; --warmup cannot be negative")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false")

    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    a = torch.ones(args.n, device=device, dtype=dtype)
    b = torch.full_like(a, 2)
    out = torch.empty_like(a)

    def run_once() -> None:
        torch.add(a, b, out=out)

    # Correctness is checked before timing. It also performs one non-timed call.
    run_once()
    expected = torch.full_like(out, 3)
    torch.testing.assert_close(out, expected)

    for _ in range(args.warmup):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples_ms: list[float] = []
    if device.type == "cuda":
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_once()
            end.record()
            end.synchronize()
            samples_ms.append(start.elapsed_time(end))
    else:
        for _ in range(args.iters):
            start = time.perf_counter()
            run_once()
            samples_ms.append((time.perf_counter() - start) * 1000)

    result = summarize(samples_ms)
    bytes_per_iteration = args.n * 3 * torch.tensor([], dtype=dtype).element_size()
    bandwidth_gb_s = bytes_per_iteration / (result["p50_ms"] / 1000) / 1e9

    print("=== vector add benchmark ===")
    print(f"device: {device}")
    print(f"dtype: {args.dtype}")
    print(f"elements: {args.n}")
    print(f"warmup: {args.warmup}")
    print(f"iterations: {args.iters}")
    print("correctness: PASS")
    for name, value in result.items():
        print(f"{name}: {value:.4f}")
    print(f"effective_read_write_bandwidth_GB_s: {bandwidth_gb_s:.2f}")


if __name__ == "__main__":
    main()
