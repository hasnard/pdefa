#!/usr/bin/env python3
"""
MatMul karşılaştırma: PyTorch vs ONNX Runtime vs Engine-AI.
C++ tarafındaki 8 şeklin aynısı ile birebir ve hatasız çalışır.
"""
import os
import re
import subprocess
import time
import numpy as np
import torch
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

HERE = os.path.dirname(os.path.abspath(__file__))

# Exe yolunu kendi sistemine göre buraya tam yazıyoruz
ENGINE_EXE = os.path.abspath(os.path.join(HERE, "..", "..", "build", "bin", "Release", "benchmark_matmul.exe"))

# C++ tarafındaki kShapes yapısının birebir aynısı[cite: 9]
SHAPES = [
    (128,  128,  128,  "128^3"),
    (256,  256,  256,  "256^3"),
    (512,  512,  512,  "512^3"),
    (1024, 1024, 1024, "1024^3"),
    (2048, 2048, 2048, "2048^3"),
    (1,    4096, 4096, "1x4096x4096 (LLM)"),
    (196,  768,  768,  "196x768x768 (ViT)"),
    (1024, 2048, 512,  "1024x2048x512"),
]

WARMUP = 5
RUNS = 20
THREADS = 12


def calc_stats(times):
    ts = sorted(times)
    idx_p95 = int(np.ceil(0.95 * len(ts))) - 1
    return {"min": ts[0], "med": ts[len(ts) // 2],
            "p95": ts[max(0, idx_p95)], "max": ts[-1]}


# ---------- PyTorch ----------
def bench_torch(M, K, N):
    torch.set_num_threads(THREADS)
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = A @ B
        ts = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            _ = A @ B
            ts.append((time.perf_counter() - t0) * 1000.0)
    return calc_stats(ts)


# ---------- ONNX Runtime ----------
def bench_onnx(M, K, N):
    A_in = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    B_init = numpy_helper.from_array(np.random.randn(K, N).astype(np.float32), "B")
    Y_out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    g = helper.make_graph([node], "mm", [A_in], [Y_out], [B_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    tmp = os.path.join(HERE, f"_tmp_mm_{M}_{K}_{N}.onnx")
    onnx.save(m, tmp)

    opts = SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = InferenceSession(tmp, opts, providers=["CPUExecutionProvider"])
    a = np.random.randn(M, K).astype(np.float32)
    for _ in range(WARMUP):
        sess.run(None, {"A": a})
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {"A": a})
        ts.append((time.perf_counter() - t0) * 1000.0)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return calc_stats(ts)


# ---------- Engine-AI Exe Çalıştır ve Parse Et ----------
def bench_engine_exe():
    if not os.path.exists(ENGINE_EXE):
        print(f"[!] Exe bulunamadi: {ENGINE_EXE}")
        return {}

    try:
        out = subprocess.check_output(
            [ENGINE_EXE], text=True, cwd=os.path.dirname(ENGINE_EXE),
            stderr=subprocess.STDOUT, timeout=600,
        )
    except Exception as e:
        print(f"[!] Exe çalıştırılamadı: {e}")
        return {}

    results = {}
    # C++ çıktısındaki satırları güvenle parse et[cite: 3, 9]
    for line in out.splitlines():
        if "ms" in line and "GFLOPS" in line:
            parts = line.strip().split()
            if len(parts) >= 3:
                # Örn: ["128^3", "0.031", "ms", "135.3", "GFLOPS"] veya benzeri
                # Etiketi ve süreyi esnek bulalım
                try:
                    ms_val = float(parts[1] if parts[1].replace('.','',1).isdigit() else parts[2])
                    # Etiket ilk parça olabilir
                    label = parts[0]
                    results[label] = ms_val
                except ValueError:
                    # Alternatif arama: 'ms' ifadesinden hemen öncesini al
                    idx_ms = line.find("ms")
                    if idx_ms != -1:
                        sub = line[:idx_ms].strip()
                        tokens = sub.split()
                        if tokens:
                            try:
                                ms_val = float(tokens[-1])
                                # Etiket geriye kalan kısım
                                label = " ".join(tokens[:-1])
                                results[label] = ms_val
                            except ValueError:
                                pass

    # Alternatif kesin eşleşme için tüm satırları tarayan esnek sözlük
    # C++ etiketlerini doğrudan anahtar yapalım
    parsed_dict = {}
    for label_name in ["128^3", "256^3", "512^3", "1024^3", "2048^3", "1x4096x4096 (LLM)", "196x768x768 (ViT)", "1024x2048x512"]:
        for line in out.splitlines():
            if label_name in line:
                idx_ms = line.find("ms")
                if idx_ms != -1:
                    sub = line[:idx_ms].strip()
                    tokens = sub.split()
                    for t in tokens:
                        try:
                            val = float(t)
                            parsed_dict[label_name] = val
                            break
                        except ValueError:
                            continue

    return parsed_dict


def main():
    print("=" * 100)
    print(f"   MatMul: PyTorch vs ONNX Runtime vs Engine-AI    "
          f"(threads={THREADS}, warmup={WARMUP}, runs={RUNS})")
    print("=" * 100)

    eng = bench_engine_exe()
    print(f"[debug] Engine'den parse edilen şekiller: {list(eng.keys())}\n")

    header = (f"   {'Shape':<22} "
              f"{'PyTorch':>16} "
              f"{'ONNX':>16} "
              f"{'Engine-AI':>16} "
              f"{'E/P':>7} {'E/O':>7}")
    print(header)
    print("-" * 100)

    for M, K, N, label in SHAPES:
        pt = bench_torch(M, K, N)
        on = bench_onnx(M, K, N)
        eg_ms = eng.get(label, None)

        pt_ms = pt["med"]
        on_ms = on["med"]

        pt_fps = 1000.0 / pt_ms
        on_fps = 1000.0 / on_ms
        eg_fps = 1000.0 / eg_ms if eg_ms else 0.0
        ep = (eg_ms / pt_ms) if eg_ms else 0.0
        eo = (eg_ms / on_ms) if eg_ms else 0.0

        pt_col = f"{pt_ms:6.3f}ms {pt_fps:5.0f}fps"
        on_col = f"{on_ms:6.3f}ms {on_fps:5.0f}fps"
        eg_col = f"{eg_ms:6.3f}ms {eg_fps:5.0f}fps" if eg_ms else "      ---       "
        ep_str = f"{ep:5.2f}x" if ep else "   -  "
        eo_str = f"{eo:5.2f}x" if eo else "   -  "

        print(f"   {label:<22} {pt_col:>16} {on_col:>16} {eg_col:>16} "
              f"{ep_str:>7} {eo_str:>7}")

    print("=" * 100)


if __name__ == "__main__":
    main()