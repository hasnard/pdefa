"""
PyTorch YOLOv8n vs pdefa engine — çıktı karşılaştırması.
Aynı resim, aynı preprocess, aynı model → aynı çıktı olmalı.
"""
import numpy as np
import torch
import pdefa
from ultralytics import YOLO
from pathlib import Path

PNG = r"D:\png\sayfa_1.png"
PT  = r"D:\png\yolov8n.pt"
ENG = r"D:\png\yolov8n.engine"

# ---- 1) PyTorch referans ----
print("[1] PyTorch inference...")
model = YOLO(PT)
torch_model = model.model.eval().fuse()

# Preprocess: pdefa ile AYNI olmalı
tensor, meta = pdefa.prepare_for_engine(PNG, 640, 640)
np_input = tensor.numpy()  # (1, 3, 640, 640) float32, 0..1

with torch.no_grad():
    x = torch.from_numpy(np_input)
    y = torch_model(x)
    if isinstance(y, (list, tuple)): y = y[0]
    pt_out = y.numpy()

print(f"    PT out shape: {pt_out.shape}")

# ---- 2) pdefa engine ----
print("[2] pdefa inference...")
ctx = pdefa.Context()
m = pdefa.Model(ctx, ENG)
s = pdefa.Session(ctx, m)
s.set_input(0, tensor)
s.run()
eng_out = s.output(0).numpy()
print(f"    Eng out shape: {eng_out.shape}")

# ---- 3) Karşılaştır ----
print("\n[3] Karşılaştırma")
assert pt_out.shape == eng_out.shape, "Shape uyuşmuyor!"

diff = np.abs(pt_out - eng_out)
cos = (np.dot(pt_out.flatten(), eng_out.flatten()) /
       (np.linalg.norm(pt_out) * np.linalg.norm(eng_out) + 1e-9))

print(f"  Max abs diff : {diff.max():.4e}")
print(f"  Mean abs diff: {diff.mean():.4e}")
print(f"  Cosine sim   : {cos:.9f}")
print(f"  PT  sum      : {pt_out.sum():.4f}")
print(f"  Eng sum      : {eng_out.sum():.4f}")

# Kanallara göre cos
if pt_out.ndim == 3 and pt_out.shape[1] == 84:
    print("\n  Kanal bazlı cosine:")
    for c in [0, 1, 2, 3, 4, 5, 50, 83]:
        a, b = pt_out[0, c], eng_out[0, c]
        cc = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        print(f"    ch{c:3d}: {cc:.6f}")

# ---- 4) Sonuç ----
print()
if cos > 0.9999 and diff.max() < 1e-2:
    print("  ✅ MÜKEMMEL — PyTorch ile sayısal olarak aynı")
elif cos > 0.999:
    print("  ✅ İYİ — ufak sayısal fark")
elif cos > 0.99:
    print("  ⚠️  KABUL EDİLEBİLİR — küçük farklar var")
else:
    print("  ❌ BAŞARISIZ — motor yanlış hesaplıyor")