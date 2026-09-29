#!/usr/bin/env python3
"""
YOLOv8s benchmark — PyTorch vs ONNX Runtime vs Engine-AI.
Tek komut: python bench_yolo.py

FIX'ler:
  - torch.inference_mode() (autograd overhead kaldırıldı)
  - ONNX ExecutionMode.ORT_PARALLEL (doğru enum)
  - ONNX export sonrası model re-fuse (export state'i bozuyor)
  - Sanity check: tek forward süresi print
  - Hardcoded 5.5 yerine gerçek motor ölçümü
"""
import os, sys, time, subprocess, re
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CONV = os.path.join(ROOT, 'tools', 'converter')
BIN  = os.path.join(ROOT, 'build', 'bin', 'Release')
sys.path.insert(0, CONV)

from onnx_to_engine import convert as onnx_to_engine
from ultralytics import YOLO

WARMUP, RUNS, THREADS = 10, 50, 12

# ============================================================
# 1) Model + export
# ============================================================
print("=== YOLOv8s hazırlanıyor ===")
model = YOLO("yolov8s.pt")
torch_model = model.model.eval().fuse()
dummy = torch.randn(1, 3, 640, 640)

onnx_path = os.path.join(CONV, "yolov8s.onnx")
torch.onnx.export(torch_model, dummy, onnx_path,
                  input_names=['images'], output_names=['output0'],
                  opset_version=17, dynamo=False)
print(f"ONNX: {onnx_path}")

engine_path = os.path.join(CONV, "yolov8s.engine")
onnx_to_engine(onnx_path, engine_path)
print(f"Engine: {engine_path}")

dummy.numpy().astype(np.float32).tofile(os.path.join(CONV, "yolov8s_input.f32"))

# Referans (inference_mode ile)
with torch.inference_mode():
    ref = torch_model(dummy)
    if isinstance(ref, (tuple, list)): ref = ref[0]
    ref_np = ref.numpy().astype(np.float32)
ref_np.tofile(os.path.join(CONV, "yolov8s_pt.f32"))
print(f"PyTorch ref: shape={ref_np.shape}  sum={ref_np.sum():.4f}")

# ============================================================
# 2) Benchmark yardımcıları
# ============================================================
def stats(times):
    t = sorted(times)
    n = len(t)
    return {
        'min':    t[0],
        'median': t[n // 2],
        'p95':    t[int(0.95 * n)],
        'max':    t[-1],
    }

def bench(label, fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter(); fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    s = stats(ts)
    print(f"  {label:25s}  median={s['median']:7.3f} ms  "
          f"({1000/s['median']:6.1f} FPS)  p95={s['p95']:7.3f}")
    return s

# ============================================================
# 3) PyTorch
# ============================================================
print("\n=== YOLOv8s 640×640 ===")
torch.set_num_threads(THREADS)
torch_model = torch_model.eval().fuse()   # ONNX export sonrası re-init

# Sanity check — tek forward
with torch.inference_mode():
    _ = torch_model(dummy)
t0 = time.perf_counter()
with torch.inference_mode():
    _ = torch_model(dummy)
dt = (time.perf_counter() - t0) * 1000.0
print(f"  [sanity] PyTorch tek forward = {dt:.2f} ms")

def _pt_forward():
    with torch.inference_mode():
        return torch_model(dummy)

pt = bench("PyTorch", _pt_forward)

# ============================================================
# 4) ONNX Runtime
# ============================================================
import onnxruntime as ort
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

so = SessionOptions()
so.intra_op_num_threads = THREADS
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
so.add_session_config_entry("session.intra_op.allow_spinning", "1")

sess = InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
dnp = dummy.numpy()

def _onnx_forward():
    return sess.run(None, {'images': dnp})

on = bench("ONNX Runtime", _onnx_forward)

# ============================================================
# 5) Engine-AI (C++ exe)
# ============================================================
eng = None
yolo_exe = os.path.join(BIN, "benchmark_yolo.exe")
if not os.path.exists(yolo_exe):
    print(f"  [uyarı] {yolo_exe} yok — Engine benchmark atlanıyor")
else:
    try:
        out = subprocess.check_output(
            [yolo_exe], text=True, cwd=ROOT, timeout=600,
            stderr=subprocess.STDOUT)
        m = re.search(r'Engine-AI\s+([\d.]+)\s+ms', out)
        if m:
            med = float(m.group(1))
            eng = {'median': med, 'min': med, 'p95': med, 'max': med}
            print(f"  {'Engine-AI':25s}  median={med:7.3f} ms  "
                  f"({1000/med:6.1f} FPS)")
            # Engine output satırı varsa göster
            for line in out.splitlines():
                if 'Engine out' in line:
                    print(f"  {line.strip()}")
        else:
            print(f"  [uyarı] Engine çıktısı parse edilemedi:")
            print(out)
    except Exception as e:
        print(f"  [hata] {e}")

# ============================================================
# 6) Karşılaştırma tablosu
# ============================================================
print()
print("=" * 78)
print(f"  YOLOv8s 640×640 — Karşılaştırma")
print("=" * 78)
print(f"  {'Framework':16s} {'Min (ms)':>10s} {'Median (ms)':>12s} "
      f"{'P95 (ms)':>10s} {'FPS':>8s} {'E/P':>6s} {'E/O':>6s}")
print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*6} {'-'*6}")

rows = [("PyTorch", pt), ("ONNX Runtime", on)]
if eng:
    rows.append(("Engine-AI", eng))

for name, s in rows:
    fps = 1000.0 / s['median'] if s['median'] > 0 else 0
    ep = (eng['median'] / s['median']) if (eng and s['median'] > 0) else 0
    eo = (eng['median'] / on['median']) if (eng and on['median'] > 0) else 0
    ep_str = f"{ep:5.2f}x" if ep else "  -  "
    eo_str = f"{eo:5.2f}x" if eo else "  -  "
    print(f"  {name:16s} {s['min']:10.3f} {s['median']:12.3f} "
          f"{s['p95']:10.3f} {fps:8.1f} {ep_str:>6s} {eo_str:>6s}")

print("=" * 78)
if eng:
    print(f"  Engine ÷ PyTorch : {eng['median']/pt['median']:.2f}x  "
          f"({'Engine hızlı' if eng['median'] < pt['median'] else 'Engine yavaş'})")
    print(f"  Engine ÷ ONNX    : {eng['median']/on['median']:.2f}x  "
          f"({'Engine hızlı' if eng['median'] < on['median'] else 'Engine yavaş'})")

# ============================================================
# 7) Engine doğruluk karşılaştırması
# ============================================================
eng_out_path = os.path.join(CONV, "yolov8s_eng.f32")
if os.path.exists(eng_out_path):
    try:
        eng_np = np.fromfile(eng_out_path, dtype=np.float32).reshape(ref_np.shape)
        if eng_np.shape == ref_np.shape:
            diff = np.abs(ref_np - eng_np)
            cos = np.dot(ref_np.flatten(), eng_np.flatten()) / (
                np.linalg.norm(ref_np) * np.linalg.norm(eng_np) + 1e-9)
            print()
            print("=== Engine vs PyTorch Doğruluk ===")
            print(f"  max abs diff  = {diff.max():.4e}")
            print(f"  mean abs diff = {diff.mean():.4e}")
            print(f"  cosine sim    = {cos:.9f}")
            if cos > 0.9999:
                print("  SONUÇ: MÜKEMMEL")
            elif cos > 0.999:
                print("  SONUÇ: İYİ")
            elif cos > 0.99:
                print("  SONUÇ: KABUL EDİLEBİLİR")
            else:
                print("  SONUÇ: BAŞARISIZ — eksik op olabilir")
        else:
            print(f"\n[!] Engine out shape {eng_np.shape} != ref {ref_np.shape}")
            print(f"    → Converter bazı node'ları atlamış olabilir")
    except Exception as e:
        print(f"\n[!] Doğruluk karşılaştırma hatası: {e}")

