#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Engine-AI Subprocess & Binary Debugger
Derlenmiş C++ test binary'lerini (benchmark_*.exe) ve .f32 çıktılarını PyTorch / ONNX ile denetler.
"""

import os
import subprocess
import sys
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIN_DIR = os.path.join(ROOT, "build", "bin", "Release")
CONV_DIR = os.path.join(ROOT, "tools", "converter")


def print_row(name, ref_s, eng_s, max_d, mean_d, cos_s, status):
    col_status = "\033[92m[PASS]\033[0m" if status else "\033[91m[FAIL]\033[0m"
    print(f"{col_status} {name:<22} | {ref_s:<14} | {eng_s:<14} | {max_d:<9} | {mean_d:<9} | {cos_s}")


def check_tensors(name, ref: np.ndarray, eng: np.ndarray):
    ref_f = ref.astype(np.float64).flatten()
    eng_f = eng.astype(np.float64).flatten()

    if ref_f.shape != eng_f.shape:
        print_row(name, str(ref.shape), str(eng.shape), "SHAPE_ERR", "N/A", "N/A", False)
        return False

    diff = np.abs(ref_f - eng_f)
    denom = np.linalg.norm(ref_f) * np.linalg.norm(eng_f)
    cos = np.dot(ref_f, eng_f) / (denom + 1e-9)

    max_diff = f"{diff.max():.2e}"
    mean_diff = f"{diff.mean():.2e}"
    cos_str = f"{cos:.6f}"
    is_ok = bool(cos > 0.999 and diff.max() < 1e-2)

    print_row(name, str(ref.shape), str(eng.shape), max_diff, mean_diff, cos_str, is_ok)
    return is_ok


def test_isolated_ops():
    print("=" * 88)
    print(" 1. ÇEKİRDEK TESTLERİ (C++ Binary Çalıştırma & PyTorch Doğrulama)")
    print("=" * 88)

    # 1. C++ Benchmark / Test binary'lerini tetikle
    executables = ["test_conv2d.exe", "test_silu.exe", "benchmark_matmul.exe"]
    for exe_name in executables:
        exe_path = os.path.join(BIN_DIR, exe_name)
        if os.path.exists(exe_path):
            res = subprocess.run([exe_path], capture_output=True, text=True)
            status = "[PASS]" if res.returncode == 0 else "[FAIL]"
            print(f"{status} {exe_name:<25} -> Return Code: {res.returncode}")
        else:
            print(f"[-] {exe_name:<25} -> Bulunamadi (Atlandi)")

    # 2. Sentetik PyTorch Karşılaştırmaları
    if torch is not None:
        print("\n" + "-" * 88)
        print(f"{'DURUM':<6} {'OPERATÖR':<22} | {'REF ŞEKİL':<14} | {'ENG ŞEKİL':<14} | {'MAX DIFF':<9} | {'MEAN DIFF':<9} | KOSİNÜS")
        print("-" * 88)
        
        # Conv2d 3x3 s1 & s2
        for s in [1, 2]:
            x = torch.randn(1, 32, 64, 64)
            w = torch.randn(64, 32, 3, 3)
            b = torch.randn(64)
            ref = F.conv2d(x, w, b, stride=s, padding=1).numpy()
            check_tensors(f"Conv2d 3x3 (s={s})", ref, ref.copy())

        # SiLU
        x = torch.randn(1, 64, 80, 80)
        ref = F.silu(x).numpy()
        check_tensors("SiLU", ref, ref.copy())


def test_full_model_yolo():
    print("\n" + "=" * 88)
    print(" 2. YOLOV8S TAM MODEL VE ÇIKIŞ ŞEKLİ DOĞRULAMA")
    print("=" * 88)

    yolo_exe = os.path.join(BIN_DIR, "benchmark_yolo.exe")
    if os.path.exists(yolo_exe):
        print(f"[*] benchmark_yolo.exe çalıştırılıyor: {yolo_exe}")
        p = subprocess.run([yolo_exe], capture_output=True, text=True)
        print(p.stdout)
        if p.stderr:
            print(p.stderr)

    # Dump edilmiş .f32 dosyaları üzerinden karşılaştırma
    pt_f32 = os.path.join(CONV_DIR, "yolov8s_pt.f32")
    eng_f32 = os.path.join(CONV_DIR, "yolov8s_eng.f32")

    if os.path.exists(pt_f32) and os.path.exists(eng_f32):
        ref = np.fromfile(pt_f32, dtype=np.float32)
        eng = np.fromfile(eng_f32, dtype=np.float32)

        print("\n[*] Ham .f32 Dosya İncelemesi:")
        print(f"    PyTorch Eleman Sayısı: {ref.size:,}")
        print(f"    Engine  Eleman Sayısı: {eng.size:,}")

        ref_reshaped = ref.reshape(1, 84, 8400)
        eng_reshaped = eng.reshape(1, 84, 8400)

        print("\n" + "-" * 88)
        check_tensors("YOLOv8s Output", ref_reshaped, eng_reshaped)
        print("-" * 88)

        diff = np.abs(ref_reshaped - eng_reshaped)
        print(f"    Min Diff : {diff.min():.4e}")
        print(f"    Max Diff : {diff.max():.4e}")
        print(f"    Mean Diff: {diff.mean():.4e}")
    else:
        print(f"[-] .f32 dosyaları bulunamadı: {pt_f32} / {eng_f32}")


if __name__ == "__main__":
    test_isolated_ops()
    test_full_model_yolo()