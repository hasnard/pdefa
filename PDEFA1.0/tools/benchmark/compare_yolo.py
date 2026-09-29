#!/usr/bin/env python3
"""Engine vs PyTorch YOLO çıktı karşılaştırması."""
import os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONV = os.path.abspath(os.path.join(HERE, '..', 'converter'))

ref = np.fromfile(os.path.join(CONV, 'yolov8s_pt.f32'),  dtype=np.float32)
eng = np.fromfile(os.path.join(CONV, 'yolov8s_eng.f32'), dtype=np.float32)

# Engine .f32 düz yazılmış — ref shape'ine göre reshape
ref = ref.reshape(1, 84, 8400)
eng = eng.reshape(1, 84, 8400)

print(f"PyTorch: n={ref.size:,}  sum={ref.sum():.4f}  min={ref.min():.4f}  max={ref.max():.4f}")
print(f"Engine : n={eng.size:,}  sum={eng.sum():.4f}  min={eng.min():.4f}  max={eng.max():.4f}")

diff = np.abs(ref - eng)
cos = np.dot(ref.flatten(), eng.flatten()) / (np.linalg.norm(ref) * np.linalg.norm(eng) + 1e-9)
print(f"\nmax abs diff = {diff.max():.4e}")
print(f"mean abs diff= {diff.mean():.4e}")
print(f"cosine sim   = {cos:.9f}")

if cos > 0.9999:
    print("SONUÇ: MÜKEMMEL — aynı çıktı")
elif cos > 0.999:
    print("SONUÇ: İYİ")
elif cos > 0.99:
    print("SONUÇ: KABUL EDİLEBİLİR")
else:
    print("SONUÇ: BAŞARISIZ — yanlış çıktı")