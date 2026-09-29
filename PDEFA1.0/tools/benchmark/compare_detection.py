#!/usr/bin/env python3
"""
MiniDetector — PyTorch vs ONNX Runtime karşılaştırma.
Motorla AYNI model, AYNI koşullar.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import subprocess
import re
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel


class MiniDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3,    16,  3, 2, 1)
        self.conv2 = nn.Conv2d(16,   32,  3, 2, 1)
        self.conv3 = nn.Conv2d(32,   64,  3, 2, 1)
        self.conv4 = nn.Conv2d(64,   128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128,  256, 3, 2, 1)
        self.relu = nn.ReLU()
        head_ch = 3 * (5 + num_classes)
        self.head = nn.Conv2d(256, head_ch, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.head(x)
        return x


def bench_engine():
    exe = os.path.join("..", "..", "build", "bin", "Release", "benchmark_detection.exe")
    if not os.path.exists(exe): return None
    out = subprocess.check_output([exe], text=True, cwd=os.path.dirname(exe))
    for line in out.splitlines():
        m = re.match(r'\s*Median:\s+([\d.]+)\s+ms', line)
        if m:
            med = float(m.group(1))
            return {'median': med}
    return None   

def bench_pytorch(iterations=10, warmup=3):
    print("\n=== PyTorch (MKL) ===")
    torch.set_num_threads(12)

    model = MiniDetector().eval()
    dummy = torch.randn(1, 3, 640, 640)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times.sort()
    t_min = times[0]
    t_med = times[len(times) // 2]
    t_max = times[-1]
    var = (t_max - t_min) / t_min * 100.0

    print(f"Min:      {t_min:7.3f} ms  ->  {1000.0/t_min:6.1f} FPS")
    print(f"Median:   {t_med:7.3f} ms  ->  {1000.0/t_med:6.1f} FPS")
    print(f"Max:      {t_max:7.3f} ms  ->  {1000.0/t_max:6.1f} FPS")
    print(f"Variance: {var:.2f}%")


def bench_onnx(iterations=10, warmup=3):
    print("\n=== ONNX Runtime ===")

    opts = SessionOptions()
    opts.intra_op_num_threads = 12
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = InferenceSession("../converter/mini_detector.onnx", opts, providers=["CPUExecutionProvider"])

    dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # Warm-up
    for _ in range(warmup):
        sess.run(None, {"input": dummy})

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        sess.run(None, {"input": dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    t_min = times[0]
    t_med = times[len(times) // 2]
    t_max = times[-1]
    var = (t_max - t_min) / t_min * 100.0

    print(f"Min:      {t_min:7.3f} ms  ->  {1000.0/t_min:6.1f} FPS")
    print(f"Median:   {t_med:7.3f} ms  ->  {1000.0/t_med:6.1f} FPS")
    print(f"Max:      {t_max:7.3f} ms  ->  {1000.0/t_max:6.1f} FPS")
    print(f"Variance: {var:.2f}%")


if __name__ == "__main__":
    import os
    if not os.path.exists("../converter/mini_detector.onnx"):
        print("HATA: mini_detector.onnx bulunamadı. Önce mini_detector.py çalıştır.")
        raise SystemExit(1)

    print("=== MiniDetector 640x640 Benchmark ===")
    print("CPU: AMD Ryzen 5 7535HS")
    print("Threads: 12")

    bench_pytorch()
    bench_onnx()