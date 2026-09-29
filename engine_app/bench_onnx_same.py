#!/usr/bin/env python3
"""
ONNX Runtime benchmark — Engine-AI ile ayni kosullar
640x640, 12 thread, ayni warmup
"""
import time
import numpy as np
import onnxruntime as ort

MODEL = r"D:\PROJE\tiny-yolo-cpp\yolov8n.onnx"
THREADS = 12
WARMUP = 5
RUNS = 30

# --- Session ayarlari (Engine-AI ile ayni) ---
opts = ort.SessionOptions()
opts.intra_op_num_threads = THREADS
opts.inter_op_num_threads = 1
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

sess = ort.InferenceSession(MODEL, opts, providers=["CPUExecutionProvider"])

in_name = sess.get_inputs()[0].name
print(f"Input:  {in_name}  shape={sess.get_inputs()[0].shape}")
for o in sess.get_outputs():
    print(f"Output: {o.name}  shape={o.shape}")
print()

# --- Ayni input: random, [1, 3, 640, 640] ---
# (Engine-AI'de de ayni input var)
x = np.random.randn(1, 3, 640, 640).astype(np.float32)

# --- Warmup ---
print(f"Warmup ({WARMUP} run)...")
for _ in range(WARMUP):
    sess.run(None, {in_name: x})

# --- Benchmark ---
print(f"Benchmark ({RUNS} run)...")
times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    sess.run(None, {in_name: x})
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000.0)

times.sort()
n = len(times)
print()
print("=== ONNX Runtime (YOLOv8n, 640x640, 12 thread) ===")
print(f"  min    = {times[0]:7.3f} ms")
print(f"  median = {times[n//2]:7.3f} ms")
print(f"  max    = {times[-1]:7.3f} ms")
print(f"  FPS    = {1000.0/times[n//2]:7.1f}")