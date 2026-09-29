#!/usr/bin/env python3
"""PyTorch vs Engine çıktı karşılaştırması."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(HERE, '..', 'converter')
RELEASE = os.path.join(HERE, '..', '..', 'build', 'bin', 'Release')

ref_path = os.path.join(CONVERTER, 'test_pytorch_output.f32')
eng_path = os.path.join(RELEASE, 'test_engine_output.f32')

if not os.path.exists(ref_path):
    print(f"❌ PyTorch referansı yok: {ref_path}")
    print("   Önce: python generate_reference.py")
    sys.exit(1)

if not os.path.exists(eng_path):
    print(f"❌ Engine çıktısı yok: {eng_path}")
    print("   Önce: benchmark_big_model.exe çalıştır")
    sys.exit(1)

ref = np.fromfile(ref_path, dtype=np.float32)
eng = np.fromfile(eng_path, dtype=np.float32)

print("=" * 60)
print("  DOĞRULUK TESTİ: PyTorch vs Engine")
print("=" * 60)
print(f"PyTorch shape: {ref.shape},  sum={ref.sum():.6f}")
print(f"Engine  shape: {eng.shape},  sum={eng.sum():.6f}")

if ref.shape != eng.shape:
    print(f"❌ BOYUT UYUŞMUYOR!")
    sys.exit(1)

# Farklar
abs_diff = np.abs(ref - eng)
max_abs = abs_diff.max()
mean_abs = abs_diff.mean()

# Relatif (sıfıra bölme korumalı)
denom = np.maximum(np.abs(ref), 1e-6)
rel_diff = abs_diff / denom
max_rel = rel_diff.max()

# Cosine similarity
dot = np.dot(ref.flatten(), eng.flatten())
norm = np.linalg.norm(ref) * np.linalg.norm(eng)
cos_sim = dot / norm if norm > 0 else 0.0

print("\n--- FARK ANALİZİ ---")
print(f"Max abs diff:  {max_abs:.3e}")
print(f"Mean abs diff: {mean_abs:.3e}")
print(f"Max rel diff:  {max_rel:.3e}")
print(f"Cosine sim:    {cos_sim:.9f}")

# Karar
print("\n--- SONUÇ ---")
if cos_sim > 0.9999 and max_abs < 1e-3:
    print("✅ MÜKEMMEL — PyTorch ile aynı çıktı (fark < 1e-3)")
elif cos_sim > 0.999 and max_abs < 1e-2:
    print("✅ İYİ — Küçük sayısal hata (fark < 1e-2)")
elif cos_sim > 0.99:
    print("⚠️  KABUL EDİLEBİLİR — Winograd kaynaklı hata olabilir")
else:
    print("❌ BAŞARISIZ — Motor yanlış sonuç veriyor!")

# Yüzde fark
pct_diff = max_abs / (np.abs(ref).max() + 1e-8) * 100
print(f"\nMax yüzde fark: {pct_diff:.4f}%")