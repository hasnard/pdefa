#!/usr/bin/env python3
"""
ONNX Runtime Matmul Benchmark — motorla AYNI koşullar.
12 thread, 1024x1024x1024, 5 warmup + 10 ölçüm.
"""

import time
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort


def make_matmul_model(path, M, K, N):
    """Tek MatMul node'u içeren ONNX modeli oluştur."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    B_init = numpy_helper.from_array(
        np.random.randn(K, N).astype(np.float32), name="B"
    )
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])

    node = helper.make_node("MatMul", ["A", "B"], ["Y"])

    graph = helper.make_graph(
        [node], "matmul_bench",
        [A], [Y],
        initializer=[B_init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return path


def main():
    M, K, N = 1024, 1024, 1024
    WARMUP = 5
    RUNS = 10

    print("=== ONNX Runtime Matmul Benchmark ===\n")
    print(f"ONNX Runtime: {ort.__version__}")

    # Thread ayarı (motorla aynı: 12)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 12
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Model oluştur
    model_path = "matmul_bench.onnx"
    make_matmul_model(model_path, M, K, N)
    print(f"Threads:      {opts.intra_op_num_threads}")
    print(f"Matrix:       {M} x {K} x {N}")
    print(f"Warmup:       {WARMUP}, Runs: {RUNS}\n")

    # Session
    sess = ort.InferenceSession(
        model_path, opts, providers=["CPUExecutionProvider"]
    )

    # Input
    A_np = np.random.randn(M, K).astype(np.float32)

    # Warm-up
    for _ in range(WARMUP):
        sess.run(None, {"A": A_np})

    # Ölçüm
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {"A": A_np})
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