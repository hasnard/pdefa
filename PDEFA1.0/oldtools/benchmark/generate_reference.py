#!/usr/bin/env python3
"""
Tam pipeline: input üret → model oluştur → ONNX export → .engine üret
              → PyTorch referansı hesapla → dosyalara yaz.

Bu sayede PyTorch ve Engine AYNI modeli, AYNI input'u kullanır.
"""
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(HERE, '..', 'converter')
sys.path.insert(0, CONVERTER)

from big_model import BigDetector
from onnx_to_engine import convert

# ---- 1) Seed sabit ----
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

# ---- 2) Modeli oluştur (SABİT ağırlıklar) ----
model = BigDetector(num_classes=2).eval()

# ---- 3) Input oluştur ----
dummy = torch.randn(1, 3, 640, 640)

# ---- 4) Input'u kaydet (engine bunu okuyacak) ----
input_path = os.path.join(CONVERTER, 'test_input.f32')
dummy.numpy().astype(np.float32).tofile(input_path)
print(f"✅ Input yazıldı: {input_path}")
print(f"   Shape: {tuple(dummy.shape)}, sum={dummy.numpy().sum():.6f}")

# ---- 5) ONNX export (AYNI model → AYNI ağırlıklar) ----
onnx_path = os.path.join(CONVERTER, 'big_model.onnx')
print(f"\n📤 ONNX export...")
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input"], output_names=["output"],
    opset_version=17, dynamo=False,
)
print(f"✅ ONNX: {onnx_path}")

# ---- 6) .engine üret (AYNI ONNX → AYNI ağırlıklar) ----
engine_path = os.path.join(CONVERTER, 'big_model.engine')
print(f"\n📤 .engine üret...")
convert(onnx_path, engine_path)
print(f"✅ Engine: {engine_path}")

# ---- 7) PyTorch referans çıktısı ----
with torch.no_grad():
    ref = model(dummy).numpy().astype(np.float32)

ref_path = os.path.join(CONVERTER, 'test_pytorch_output.f32')
ref.tofile(ref_path)
print(f"\n✅ PyTorch referansı: {ref_path}")
print(f"   Shape: {ref.shape}, sum={ref.sum():.6f}")

print("\n" + "=" * 60)
print("  HAZIR — şimdi benchmark_big_model.exe çalıştır")
print("=" * 60)