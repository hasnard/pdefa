#!/usr/bin/env python3
"""
Conv2d Benchmark — PyTorch vs ONNX Runtime vs Engine-AI
Aynı YOLOv5n layer'ları, motorumuzun benchmark'ı ile birebir aynı.
"""

import os
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn

from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
import onnx
from onnx import helper, TensorProto, numpy_helper


# Motorun test ettiği layer'lar
LAYERS = [
    # (name, N, Cin, H, W, Cout, KH, KW, stride, pad)
    ("Conv1  3->16  640x640",  1, 3,   640, 640, 16,  3, 3, 2, 1),
    ("Conv2  16->32 320x320",  1, 16,  320, 320, 32,  3, 3, 2, 1),
    ("Conv3  32->64 160x160",  1, 32,  160, 160, 64,  3, 3, 2, 1),
    ("Conv4  64->128 80x80",   1, 64,  80,  80,  128, 3, 3, 2, 1),
    ("Conv5  128->256 40x40",  1, 128, 40,  40,  256, 3, 3, 2, 1),
    ("Conv6  256->512 20x20",  1, 256, 20,  20,  512, 3, 3, 2, 1),
    ("1x1   64->128 80x80",    1, 64,  80,  80,  128, 1, 1, 1, 0),
    ("1x1   128->256 40x40",   1, 128, 40,  40,  256, 1, 1, 1, 0),
    ("3x3   64->64 80x80",     1, 64,  80,  80,  64,  3, 3, 1, 1),
    ("3x3   128->128 40x40",   1, 128, 40,  40,  128, 3, 3, 1, 1),
]


def bench_pytorch(name, N, Cin, H, W, Cout, KH, KW, stride, pad, warmup=3, runs=10):
    """PyTorch (MKL) benchmark."""
    conv = nn.Conv2d(Cin, Cout, (KH, KW), stride=stride, padding=pad, bias=True).eval()
    x = torch.randn(N, Cin, H, W)

    Hout = (H + 2 * pad - KH) // stride + 1
    Wout = (W + 2 * pad - KW) // stride + 1
    flops = 2.0 * N * Cout * Cin * KH * KW * Hout * Wout / 1e9

    with torch.no_grad():
        for _ in range(warmup):
            _ = conv(x)

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = conv(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times.sort()
    t_med = times[len(times) // 2]
    t_min = times[0]
    t_max = times[-1]
    var = (t_max - t_min) / t_min * 100.0

    print(f"  {name:30s} {t_med:7.3f} ms (med)  {flops / (t_med * 1e-3):6.2f} GFLOPS  var={var:.1f}%")
    return t_med, flops


def bench_onnx(name, N, Cin, H, W, Cout, KH, KW, stride, pad, warmup=3, runs=10):
    """ONNX Runtime benchmark."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [N, Cin, H, W])
    W_arr = np.random.randn(Cout, Cin, KH, KW).astype(np.float32)
    B_arr = np.random.randn(Cout).astype(np.float32)
    W_init = numpy_helper.from_array(W_arr, name="W")
    B_init = numpy_helper.from_array(B_arr, name="B")

    Hout = (H + 2 * pad - KH) // stride + 1
    Wout = (W + 2 * pad - KW) // stride + 1
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [N, Cout, Hout, Wout])

    conv_node = helper.make_node(
        "Conv", ["input", "W", "B"], ["output"],
        kernel_shape=[KH, KW], strides=[stride, stride], pads=[pad, pad, pad, pad]
    )
    graph = helper.make_graph([conv_node], "conv_bench", [X], [Y], [W_init, B_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, f"_tmp_{name.replace(' ', '_').replace('>', '_')}.onnx")

    opts = SessionOptions()
    opts.intra_op_num_threads = 12
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = InferenceSession(f"_tmp_{name.replace(' ', '_').replace('>', '_')}.onnx",
                            opts, providers=["CPUExecutionProvider"])

    x_np = np.random.randn(N, Cin, H, W).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {"input": x_np})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": x_np})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    t_med = times[len(times) // 2]
    t_min = times[0]
    t_max = times[-1]
    var = (t_max - t_min) / t_min * 100.0

    flops = 2.0 * N * Cout * Cin * KH * KW * Hout * Wout / 1e9

    print(f"  {name:30s} {t_med:7.3f} ms (med)  {flops / (t_med * 1e-3):6.2f} GFLOPS  var={var:.1f}%")
    return t_med, flops


def bench_engine():
    """Motorun benchmark_conv2d.exe'sini çağır, CSV satırlarını parse et."""
    candidates = [
        r"D:\PROJE\engineforaiincpu\PDEFA1.0\build\bin\Release\benchmark_conv2d.exe",
        "benchmark_conv2d.exe",
        "./benchmark_conv2d",
        "../../build/bin/Release/benchmark_conv2d.exe",
    ]
    exe = next((p for p in candidates if os.path.exists(p)), None)
    if exe is None:
        print("  [!] benchmark_conv2d.exe bulunamadi — motor kismi atlanacak.")
        return {}

    try:
        out = subprocess.check_output([exe], text=True, timeout=300)
    except Exception as e:
        print(f"  [!] Motor exe hata verdi: {e}")
        return {}

    results = {}
    for line in out.splitlines():
        if not line.startswith("CSV,"):
            continue
        parts = line.split(",")
        if len(parts) != 4:
            continue
        _, name, ms_s, gf_s = parts
        try:
            results[name.strip()] = (float(ms_s), float(gf_s))
        except ValueError:
            continue
    return results


def main():
    torch.set_num_threads(12)

    print("=== Conv2d Benchmark: PyTorch vs ONNX Runtime vs Engine-AI ===\n")
    print(f"PyTorch: {torch.__version__}")
    print(f"Threads: 12\n")

    print("--- PyTorch (MKL) ---")
    pt_results = {}
    for layer in LAYERS:
        name = layer[0]
        pt_results[name] = bench_pytorch(*layer)

    print("\n--- ONNX Runtime ---")
    ort_results = {}
    for layer in LAYERS:
        name = layer[0]
        ort_results[name] = bench_onnx(*layer)

    print("\n--- Engine-AI (bizim motor) ---")
    eng_results = bench_engine()
    for layer in LAYERS:
        name = layer[0]
        if name in eng_results:
            ms, gf = eng_results[name]
            print(f"  {name:30s} {ms:7.3f} ms (med)  {gf:6.2f} GFLOPS")
        else:
            print(f"  {name:30s} {'N/A':>7s}")

    # Özet
    print("\n" + "=" * 100)
    print("  ÖZET: PyTorch vs ONNX Runtime vs Engine-AI")
    print("=" * 100)
    print(f"  {'Katman':30s} {'PyTorch':>10s} {'ONNX':>10s} {'Engine':>10s} {'Eng/PT':>9s} {'Eng/ONNX':>10s}")
    print("-" * 100)

    pt_total = ort_total = eng_total = 0.0
    eng_count = 0

    for layer in LAYERS:
        name = layer[0]
        pt_t  = pt_results[name][0]
        ort_t = ort_results[name][0]
        pt_total  += pt_t
        ort_total += ort_t

        if name in eng_results:
            eng_t = eng_results[name][0]
            eng_total += eng_t
            eng_count += 1
            print(f"  {name:30s} {pt_t:8.3f} ms {ort_t:8.3f} ms {eng_t:8.3f} ms "
                  f"{eng_t/pt_t:7.2f}x {eng_t/max(ort_t,1e-9):8.2f}x")
        else:
            print(f"  {name:30s} {pt_t:8.3f} ms {ort_t:8.3f} ms {'N/A':>8s}")

    print("-" * 100)
    if eng_count > 0:
        print(f"  {'TOPLAM':30s} {pt_total:8.3f} ms {ort_total:8.3f} ms {eng_total:8.3f} ms "
              f"{eng_total/max(pt_total,1e-9):7.2f}x {eng_total/max(ort_total,1e-9):8.2f}x")

    # _tmp_*.onnx temizle
    import glob
    for f in glob.glob("_tmp_*.onnx"):
        try:
            os.remove(f)
        except OSError:
            pass
    print("\n[cleanup] _tmp_*.onnx silindi.")


if __name__ == "__main__":
    main()