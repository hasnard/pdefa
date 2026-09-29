#!/usr/bin/env python3
"""
PyTorch Matmul Benchmark — motorla AYNI koşullar.
12 thread, 1024x1024x1024, 5 warmup + 10 ölçüm.
"""

import time
import numpy as np
import torch


def main():
    torch.set_num_threads(12)

    M, K, N = 1024, 1024, 1024
    WARMUP = 5
    RUNS = 10

    print("=== PyTorch Matmul Benchmark ===\n")
    print(f"PyTorch:  {torch.__version__}")
    print(f"Threads:  {torch.get_num_threads()}")
    print(f"Matrix:   {M} x {K} x {N}")
    print(f"Warmup:   {WARMUP}, Runs: {RUNS}\n")

    np.random.seed(42)
    A = torch.from_numpy(np.random.uniform(-1, 1, (M, K)).astype(np.float32))
    B = torch.from_numpy(np.random.uniform(-1, 1, (K, N)).astype(np.float32))

    for _ in range(WARMUP):
        _ = A @ B

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _ = A @ B
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    t_min = times[0]
    t_med = times[RUNS // 2]
    t_max = times[-1]

    flops = 2.0 * M * N * K
    print("--- Results ---")
    print(f"Min:    {t_min:7.3f} ms  →  {flops / (t_min * 1e-3) / 1e9:7.2f} GFLOPS")
    print(f"Median: {t_med:7.3f} ms  →  {flops / (t_med * 1e-3) / 1e9:7.2f} GFLOPS")
    print(f"Max:    {t_max:7.3f} ms  →  {flops / (t_max * 1e-3) / 1e9:7.2f} GFLOPS")
    print()


if __name__ == "__main__":
    main()