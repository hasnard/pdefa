#!/usr/bin/env python3
"""
YOLOv8n — 4'lü karşılaştırma:
   PyTorch | ONNX Runtime | Engine-AI (senin) | tiny-yolo-cpp (QiuMr)

Ön koşullar:
  1. D:\\PROJE\\tiny-yolo-cpp\\test_tiny.exe  (derlenmiş)
  2. D:\\PROJE\\tiny-yolo-cpp\\yolov8n.onnx
  3. D:\\PROJE\\tiny-yolo-cpp\\zidane.jpg
  4. tools\\converter\\yolov8n.engine  (senin motorun)
  5. benchmark_yolo.exe yolov8n.engine'i okuyacak şekilde derlenmiş
"""
import os, sys, time, subprocess, re
import numpy as np
import torch

ROOT_ENGINE = r"D:\PROJE\engineforaiincpu\PDEFA1.0"
CONVERTER   = os.path.join(ROOT_ENGINE, "tools", "converter")
RELEASE_BIN = os.path.join(ROOT_ENGINE, "build", "bin", "Release")

TINY_REPO = r"D:\PROJE\tiny-yolo-cpp"
TINY_EXE  = os.path.join(TINY_REPO, "test_tiny.exe")
TINY_ONNX = os.path.join(TINY_REPO, "yolov8n.onnx")
TINY_IMG  = os.path.join(TINY_REPO, "zidane.jpg")
TINY_SIZE = 640
TINY_CONF = 0.25

MODEL_PT  = os.path.join(TINY_REPO, "yolov8n.pt")
MODEL_ONNX = TINY_ONNX
MODEL_ENG  = os.path.join(CONVERTER, "yolov8n.engine")

THREADS = 12
RUNS    = 30
WARMUP  = 5

def stats(times):
    t = sorted(times)
    n = len(t)
    return {'min': t[0], 'median': t[n//2], 'max': t[-1]}

def row(name, s):
    if s is None:
        print(f"  {name:<18} {'N/A':>12}")
        return
    fps = 1000.0 / s['median'] if s['median'] > 0 else 0
    print(f"  {name:<18} med={s['median']:8.3f} ms   "
          f"min={s['min']:8.3f}   max={s['max']:8.3f}   {fps:6.1f} FPS")

# ============================================================
#  PyTorch
# ============================================================
def bench_torch():
    try:
        from ultralytics import YOLO
        from PIL import Image
        import torchvision.transforms.functional as TF
    except ImportError as e:
        print(f"  [!] ultralytics/PIL/torchvision eksik: {e}")
        return None

    if not os.path.exists(MODEL_PT):
        print(f"  [!] {MODEL_PT} yok")
        return None
    if not os.path.exists(TINY_IMG):
        print(f"  [!] {TINY_IMG} yok")
        return None

    torch.set_num_threads(THREADS)
    m = YOLO(MODEL_PT).model.eval().fuse()

    img = Image.open(TINY_IMG).convert("RGB").resize((640, 640))
    x = TF.to_tensor(img).unsqueeze(0)

    with torch.inference_mode():
        for _ in range(WARMUP):
            out = m(x)
            if isinstance(out, (tuple, list)): out = out[0]
        ts = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = m(x)
            if isinstance(out, (tuple, list)): out = out[0]
            ts.append((time.perf_counter() - t0) * 1000.0)
    return stats(ts)

# ============================================================
#  ONNX Runtime
# ============================================================
def bench_onnx():
    try:
        from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
        from PIL import Image
    except ImportError:
        print("  [!] onnxruntime/PIL yok")
        return None

    if not os.path.exists(MODEL_ONNX):
        return None

    opts = SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = InferenceSession(MODEL_ONNX, opts, providers=["CPUExecutionProvider"])

    img = Image.open(TINY_IMG).convert("RGB").resize((640, 640))
    x = np.array(img, dtype=np.float32).transpose(2,0,1)[None] / 255.0
    in_name = sess.get_inputs()[0].name

    for _ in range(WARMUP):
        sess.run(None, {in_name: x})
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {in_name: x})
        ts.append((time.perf_counter() - t0) * 1000.0)
    return stats(ts)

# ============================================================
#  Engine-AI (senin) — benchmark_yolo.exe yolov8n.engine okumalı
# ============================================================
def bench_engine():
    """Engine-AI — DLL üzerinden (exe yok)."""
    try:
        import pdefa
    except ImportError:
        print("  [!] pdefa kurulu değil")
        return None

    if not os.path.exists(MODEL_ENG):
        print(f"  [!] {MODEL_ENG} yok")
        return None
    if not os.path.exists(TINY_IMG):
        print(f"  [!] {TINY_IMG} yok")
        return None

    # Model + session
    ctx     = pdefa.Context()
    model   = pdefa.Model(ctx, MODEL_ENG)
    session = pdefa.Session(ctx, model)

    # Preprocess (bir kere — aynı resim, aynı tensor)
    tensor, meta = pdefa.prepare_for_model(model, TINY_IMG)

    # Warmup
    for _ in range(WARMUP):
        session.set_input(0, tensor)
        session.run()

    # Benchmark
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        session.set_input(0, tensor)
        session.run()
        ts.append((time.perf_counter() - t0) * 1000.0)

    return stats(ts)

# ============================================================
#  tiny-yolo-cpp (QiuMr)
# ============================================================
def bench_tiny():
    if not os.path.exists(TINY_EXE):
        print(f"  [!] {TINY_EXE} yok")
        return None
    if not os.path.exists(TINY_ONNX):
        print(f"  [!] {TINY_ONNX} yok")
        return None
    if not os.path.exists(TINY_IMG):
        print(f"  [!] {TINY_IMG} yok")
        return None

    cmd = [TINY_EXE, TINY_ONNX, TINY_IMG, str(TINY_SIZE), str(TINY_CONF)]
    try:
        out = subprocess.check_output(cmd, text=True,
                                       cwd=TINY_REPO, timeout=600,
                                       stderr=subprocess.STDOUT)
    except Exception as e:
        print(f"  [!] tiny-yolo hata: {e}")
        return None

    # "time=116 ms" satırlarından ms çek
    ms_vals = []
    for line in out.splitlines():
        m = re.search(r'time\s*=\s*([\d.]+)\s*ms', line)
        if m:
            ms_vals.append(float(m.group(1)))

    if not ms_vals:
        print("  [!] tiny-yolo ms çıktısı bulunamadı. Ham çıktı:")
        print(out[-400:])
        return None

    return stats(ms_vals)

# ============================================================
def main():
    print("=" * 78)
    print("  YOLOv8n — 4'lü Karşılaştırma  (aynı makine, aynı input, 12 thread)")
    print("=" * 78)

    print("\n[1/4] PyTorch...")
    pt = bench_torch()
    row("PyTorch", pt)

    print("\n[2/4] ONNX Runtime...")
    on = bench_onnx()
    row("ONNX Runtime", on)

    print("\n[3/4] Engine-AI (senin)...")
    eg = bench_engine()
    row("Engine-AI", eg)

    print("\n[4/4] tiny-yolo-cpp (QiuMr)...")
    ty = bench_tiny()
    row("tiny-yolo-cpp", ty)

    print("\n" + "=" * 78)
    print("  ÖZET")
    print("=" * 78)
    print(f"  {'Motor':<18} {'Median(ms)':>12} {'FPS':>8} {'vs PyTorch':>12} {'vs ONNX':>10}")
    print("  " + "-" * 66)
    ref_pt = pt['median'] if pt else 0
    ref_on = on['median'] if on else 0
    for name, s in [("PyTorch", pt), ("ONNX Runtime", on),
                    ("Engine-AI", eg), ("tiny-yolo-cpp", ty)]:
        if s is None:
            print(f"  {name:<18} {'N/A':>12}")
            continue
        fps = 1000.0 / s['median'] if s['median'] > 0 else 0
        rpt = s['median'] / ref_pt if ref_pt > 0 else 0
        ron = s['median'] / ref_on if ref_on > 0 else 0
        print(f"  {name:<18} {s['median']:>12.3f} {fps:>8.1f} "
              f"{rpt:>12.2f}x {ron:>10.2f}x")

    print("\n  Not: Daha düşük = daha hızlı. 1.00x = referans ile eşit.")
    if eg and ty:
        winner = "Engine-AI" if eg['median'] < ty['median'] else "tiny-yolo-cpp"
        ratio = max(eg['median'], ty['median']) / min(eg['median'], ty['median'])
        print(f"\n  🏆 Kazanan: {winner}  ({ratio:.2f}x hızlı)")

if __name__ == "__main__":
    main()