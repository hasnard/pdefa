#!/usr/bin/env python3
"""BigDetector — PyTorch vs ONNX Runtime benchmark."""

import time
import numpy as np
import torch
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

from big_model import BigDetector


def bench_pytorch(iters=10, warmup=3):
    print("\n=== PyTorch (MKL) ===")
    torch.set_num_threads(12)
    model = BigDetector(num_classes=2).eval()
    dummy = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times.sort()
    print(f"Min:      {times[0]:8.3f} ms  ->  {1000.0/times[0]:6.1f} FPS")
    print(f"Median:   {times[len(times)//2]:8.3f} ms  ->  {1000.0/times[len(times)//2]:6.1f} FPS")
    print(f"Max:      {times[-1]:8.3f} ms  ->  {1000.0/times[-1]:6.1f} FPS")


def bench_onnx(iters=10, warmup=3):
    print("\n=== ONNX Runtime ===")
    opts = SessionOptions()
    opts.intra_op_num_threads = 12
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = InferenceSession("../converter/big_model.onnx", opts,
                             providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {"input": dummy})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {"input": dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    print(f"Min:      {times[0]:8.3f} ms  ->  {1000.0/times[0]:6.1f} FPS")
    print(f"Median:   {times[len(times)//2]:8.3f} ms  ->  {1000.0/times[len(times)//2]:6.1f} FPS")
    print(f"Max:      {times[-1]:8.3f} ms  ->  {1000.0/times[-1]:6.1f} FPS")


if __name__ == "__main__":
    print("=== BigDetector 640x640 Benchmark ===")
    print("Threads: 12")
    bench_pytorch()
    bench_onnx()