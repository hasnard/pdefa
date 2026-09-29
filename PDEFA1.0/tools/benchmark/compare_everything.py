#!/usr/bin/env python3
"""
Engine-AI vs PyTorch vs ONNX RT — Kapsamlı Karşılaştırma Betiği.

Bölümler:
  0) Referans hazırlığı  : model → ONNX → .engine + PyTorch referansı
  1) MatMul              : 8 şekil (PyTorch vs ONNX vs Engine)
  2) BigDetector 640×640 : uçtan uca
  3) MiniDetector 640×640: uçtan uca
  4) YOLOv8s 640×640     : uçtan uca + doğruluk + kanal analizi
  5) Doğruluk testi      : PyTorch vs Engine çıktı karşılaştırması (BigDetector)
  6) Katman analizi      : PyTorch (op tipi bazında, dinamik peak TFLOPS)
  7) Engine vs PyTorch   : katman katman (CSV otomatik üretilir)

Kullanım:
    python compare_everything.py                  # her şey
    python compare_everything.py --only matmul    # sadece MatMul
    python compare_everything.py --only yolo      # sadece YOLOv8s
    python compare_everything.py --only layers    # PyTorch katman analizi
    python compare_everything.py --only layer_vs  # Engine vs PyTorch katman katman
    python compare_everything.py --skip-prep      # referans hazırlığını atla
    python compare_everything.py --no-color       # renkleri kapat
"""
import os
import sys
import csv
import time
import argparse
import subprocess
import re
import platform
from statistics import median as _median
import numpy as np
import torch
import torch.nn as nn

from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
import onnx
from onnx import helper, TensorProto, numpy_helper

# ============================================================
#  YOLLAR
# ============================================================
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CONVERTER    = os.path.join(PROJECT_ROOT, 'tools', 'converter')
RELEASE_BIN  = os.path.join(PROJECT_ROOT, 'build', 'bin', 'Release')

sys.path.insert(0, CONVERTER)

try:
    from big_model import BigDetector
except ImportError:
    BigDetector = None
    print("[!] big_model.py import edilemedi")

try:
    from onnx_to_engine import convert as onnx_to_engine_convert
except ImportError:
    onnx_to_engine_convert = None
    print("[!] onnx_to_engine.py import edilemedi")

# ============================================================
#  CPU PEAK FLOPS (dinamik)
# ============================================================
def detect_cpu_info():
    """CPU adı, fiziksel çekirdek, boost frekans, AVX2 peak TFLOPS döndür."""
    name = platform.processor() or "Unknown"
    # Fiziksel çekirdek sayısı
    try:
        cores = os.cpu_count() or 4
    except Exception:
        cores = 4

    # AVX2+FMA: 16 FLOP/cycle/core
    # Zen 3+/Intel Skylake+ için boost frekansı (all-core) tahmini
    cpu = name.lower()
    if 'ryzen 7 7' in cpu or 'ryzen 9 7' in cpu:
        phys_cores, boost_ghz = 8, 4.0
    elif 'ryzen 5 7' in cpu:
        phys_cores, boost_ghz = 6, 3.8   # 7535HS: 6C/12T, ~4.55 single, ~3.8 sustained
    elif 'ryzen' in cpu:
        phys_cores, boost_ghz = max(2, cores // 2), 3.5
    elif 'intel' in cpu and 'i9' in cpu:
        phys_cores, boost_ghz = max(2, cores // 2), 3.8
    elif 'intel' in cpu and 'i7' in cpu:
        phys_cores, boost_ghz = max(2, cores // 2), 3.6
    elif 'intel' in cpu:
        phys_cores, boost_ghz = max(2, cores // 2), 3.4
    else:
        phys_cores, boost_ghz = max(2, cores // 2), 3.0

    # Peak: phys × 16 FLOP/cycle × boost_GHz / 1000 = TFLOPS
    peak_tflops = phys_cores * 16 * boost_ghz * 1e-3
    return name, phys_cores, boost_ghz, peak_tflops


CPU_NAME, PHYS_CORES, BOOST_GHZ, PEAK_TFLOPS = detect_cpu_info()


# ============================================================
#  MINIDETECTOR
# ============================================================
class MiniDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3,    16,  3, 2, 1)
        self.conv2 = nn.Conv2d(16,   32,  3, 2, 1)
        self.conv3 = nn.Conv2d(32,   64,  3, 2, 1)
        self.conv4 = nn.Conv2d(64,   128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128,  256, 3, 2, 1)
        self.relu  = nn.ReLU()
        self.head  = nn.Conv2d(256, 3 * (5 + num_classes), 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        return self.head(x)

# ============================================================
#  RENK & FORMAT YARDIMCILARI
# ============================================================
class C:
    RESET  = '\033[0m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'
    RED    = '\033[91m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    BLUE   = '\033[94m'
    MAGENTA= '\033[95m'
    CYAN   = '\033[96m'
    GRAY   = '\033[90m'

USE_COLOR = True

def col(s, color):
    return f"{color}{s}{C.RESET}" if USE_COLOR else s

def visible_len(s):
    return len(re.sub(r'\033\[[0-9;]*m', '', s))

def pad(s, width, align='left'):
    diff = width - visible_len(s)
    if diff <= 0:
        return s
    if align == 'left':
        return s + ' ' * diff
    if align == 'right':
        return ' ' * diff + s
    left = diff // 2
    return ' ' * left + s + ' ' * (diff - left)

# ============================================================
#  GENEL TABLO ÇİZİCİ
# ============================================================
def draw_table(headers, rows, aligns=None, title=None):
    n_cols = len(headers)
    if aligns is None:
        aligns = ['l'] + ['r'] * (n_cols - 1)
    align_map = {'l': 'left', 'r': 'right', 'c': 'center'}

    widths = [visible_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], visible_len(cell))
    widths = [w + 2 for w in widths]

    def line(left, mid, right, fill='─'):
        return left + mid.join(fill * w for w in widths) + right

    lines = []
    if title:
        total_w = sum(widths) + (n_cols - 1)
        title_str = f" {title} "
        lines.append(col("┌" + title_str.center(total_w, '─') + "┐", C.CYAN))

    lines.append(col(line("┌", "┬", "┐"), C.GRAY))

    header_cells = [pad(col(h, C.BOLD + C.CYAN), widths[i], 'center')
                    for i, h in enumerate(headers)]
    lines.append(col("│", C.GRAY) + col("│", C.GRAY).join(header_cells) + col("│", C.GRAY))

    lines.append(col(line("├", "┼", "┤"), C.GRAY))

    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            if i >= n_cols:
                break
            cells.append(pad(cell, widths[i], align_map[aligns[i]]))
        sep = col("│", C.GRAY)
        lines.append(sep + sep.join(cells) + sep)

    lines.append(col(line("└", "┴", "┘"), C.GRAY))
    return "\n".join(lines)

def ratio_cell(value, ref=1.0, lower_is_better=True):
    if value <= 0:
        return col("  -  ", C.GRAY)
    txt = f"{value:5.2f}x"
    if lower_is_better:
        if value < 0.95:
            return col(txt, C.GREEN + C.BOLD)
        elif value > 1.10:
            return col(txt, C.RED + C.BOLD)
        else:
            return col(txt, C.YELLOW)
    else:
        if value > 1.05:
            return col(txt, C.GREEN + C.BOLD)
        elif value < 0.90:
            return col(txt, C.RED + C.BOLD)
        else:
            return col(txt, C.YELLOW)

def fmt_ms(ms):
    return f"{ms:7.3f}"

def fmt_fps(fps):
    return f"{fps:6.0f}"

def fmt_ms_fps(ms):
    fps = 1000.0 / ms if ms > 0 else 0
    return f"{ms:7.3f} ms  {fps:5.0f} fps"

# ============================================================
#  AYARLAR
# ============================================================
THREADS = 12
WARMUP  = 15
RUNS    = 50

MM_WARMUP = 5
MM_RUNS   = 20

MM_SHAPES = [
    (128,  128,  128,  "128^3"),
    (256,  256,  256,  "256^3"),
    (512,  512,  512,  "512^3"),
    (1024, 1024, 1024, "1024^3"),
    (2048, 2048, 2048, "2048^3"),
    (1,    4096, 4096, "1×4096×4096 (LLM)"),
    (196,  768,  768,  "196×768×768 (ViT)"),
    (1024, 2048, 512,  "1024×2048×512"),
]

MM_LABEL_ALIASES = {
    "1×4096×4096 (LLM)": ["1x4096x4096 (LLM)", "1x4096x4096(LLM)", "1×4096×4096 (LLM)"],
    "196×768×768 (ViT)": ["196x768x768 (ViT)", "196x768x768(ViT)", "196×768×768 (ViT)"],
    "1024×2048×512":     ["1024x2048x512", "1024×2048×512"],
}

REF_INPUT     = os.path.join(CONVERTER, 'test_input.f32')
REF_ONNX      = os.path.join(CONVERTER, 'big_model.onnx')
REF_ENGINE    = os.path.join(CONVERTER, 'big_model.engine')
REF_PT_OUT    = os.path.join(CONVERTER, 'test_pytorch_output.f32')
REF_ENG_OUT   = os.path.join(RELEASE_BIN, 'test_engine_output.f32')

# YOLO
YOLO_MODEL_PT   = "yolov8s.pt"
YOLO_ONNX       = os.path.join(CONVERTER, 'yolov8s.onnx')
YOLO_ENGINE     = os.path.join(CONVERTER, 'yolov8s.engine')
YOLO_PT_OUT     = os.path.join(CONVERTER, 'yolov8s_pt.f32')
YOLO_ENG_OUT    = os.path.join(CONVERTER, 'yolov8s_eng.f32')

# ============================================================
#  YARDIMCILAR
# ============================================================
def stats_ms(times):
    t = sorted(times)
    n = len(t)
    idx_p95 = int(np.ceil(0.95 * n)) - 1
    return {
        'min':    t[0],
        'median': t[n // 2],
        'p95':    t[max(0, idx_p95)],
        'max':    t[-1],
    }

def call_engine_exe(exe_name, env=None, capture_stderr=True, cwd=None):
    exe = os.path.join(RELEASE_BIN, exe_name)
    if not os.path.exists(exe):
        print(f"  {col('[!]', C.RED)} {exe_name} bulunamadı")
        return None
    myenv = os.environ.copy()
    if env:
        myenv.update(env)
    try:
        return subprocess.check_output(
            [exe], text=True, cwd=(cwd or RELEASE_BIN), env=myenv,
            stderr=subprocess.STDOUT if capture_stderr else subprocess.DEVNULL,
            timeout=600,
        )
    except Exception as e:
        print(f"  {col('[!]', C.RED)} {exe_name}: {e}")
        return None

def parse_min_med_max(out):
    if not out:
        return None
    times = {}
    for line in out.splitlines():
        m = re.match(r'\s*(Min|Median|Max):\s+([\d.]+)\s+ms', line)
        if m:
            times[m.group(1).lower()] = float(m.group(2))
    if 'median' not in times:
        return None
    return {
        'min':    times.get('min', 0.0),
        'median': times.get('median', 0.0),
        'p95':    times.get('max', 0.0),
        'max':    times.get('max', 0.0),
    }

def parse_matmul_output(out):
    if not out:
        return {}
    results = {}
    for line in out.splitlines():
        m = re.search(r'([\d.]+)\s*ms', line)
        if not m:
            continue
        try:
            ms_val = float(m.group(1))
        except ValueError:
            continue
        for _, _, _, label in MM_SHAPES:
            aliases = MM_LABEL_ALIASES.get(label, [label])
            if label in line or any(a in line for a in aliases):
                results[label] = ms_val
                break
    return results

def section_header(num, title):
    line = "═" * 70
    print(f"\n{col(line, C.CYAN)}")
    print(f"  {col(f'BÖLÜM {num}', C.BOLD + C.YELLOW)}  {col(title, C.BOLD)}")
    print(f"{col(line, C.CYAN)}")

# ============================================================
#  BÖLÜM 0: REFERANS HAZIRLIĞI
# ============================================================
def prepare_reference():
    section_header(0, "Referans Hazırlığı")

    if BigDetector is None or onnx_to_engine_convert is None:
        print(f"  {col('[!]', C.RED)} Gerekli modüller yok, atlanıyor")
        return False

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    model = BigDetector(num_classes=2).eval()
    dummy = torch.randn(1, 3, 640, 640)
    dummy.numpy().astype(np.float32).tofile(REF_INPUT)
    print(f"  {col('✓', C.GREEN)} Input         → {REF_INPUT}")
    print(f"    shape={tuple(dummy.shape)}, sum={dummy.numpy().sum():.6f}")

    torch.onnx.export(
        model, dummy, REF_ONNX,
        input_names=["input"], output_names=["output"],
        opset_version=17, dynamo=False,
    )
    print(f"  {col('✓', C.GREEN)} ONNX          → {REF_ONNX}")

    try:
        onnx_to_engine_convert(REF_ONNX, REF_ENGINE)
        print(f"  {col('✓', C.GREEN)} .engine       → {REF_ENGINE}")
    except Exception as e:
        print(f"  {col('✗', C.RED)} .engine üretilemedi: {e}")

    with torch.no_grad():
        ref = model(dummy).numpy().astype(np.float32)
    ref.tofile(REF_PT_OUT)
    print(f"  {col('✓', C.GREEN)} PyTorch ref   → {REF_PT_OUT}")
    print(f"    shape={ref.shape}, sum={ref.sum():.6f}")
    return True

# ============================================================
#  BÖLÜM 1: MATMUL
# ============================================================
def bench_torch_matmul(M, K, N):
    torch.set_num_threads(THREADS)
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    with torch.no_grad():
        for _ in range(MM_WARMUP):
            _ = A @ B
        ts = []
        for _ in range(MM_RUNS):
            t0 = time.perf_counter()
            _ = A @ B
            ts.append((time.perf_counter() - t0) * 1000.0)
    return stats_ms(ts)

def bench_onnx_matmul(M, K, N):
    A_in   = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    B_init = numpy_helper.from_array(np.random.randn(K, N).astype(np.float32), "B")
    Y_out  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    node   = helper.make_node("MatMul", ["A", "B"], ["Y"])
    g      = helper.make_graph([node], "mm", [A_in], [Y_out], [B_init])
    m      = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    tmp    = os.path.join(HERE, f"_tmp_mm_{M}_{K}_{N}.onnx")
    onnx.save(m, tmp)

    opts = SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = InferenceSession(tmp, opts, providers=["CPUExecutionProvider"])

    a = np.random.randn(M, K).astype(np.float32)
    for _ in range(MM_WARMUP):
        sess.run(None, {"A": a})
    ts = []
    for _ in range(MM_RUNS):
        t0 = time.perf_counter()
        sess.run(None, {"A": a})
        ts.append((time.perf_counter() - t0) * 1000.0)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return stats_ms(ts)

def bench_engine_matmul():
    out = call_engine_exe("benchmark_matmul.exe", capture_stderr=True)
    if not out:
        return {}
    parsed = parse_matmul_output(out)
    if not parsed:
        print(f"  {col('[!]', C.RED)} Engine-AI MatMul çıktısı parse edilemedi.")
    return parsed

def section_matmul():
    section_header(1, f"MatMul Benchmark  (threads={THREADS}, warmup={MM_WARMUP}, runs={MM_RUNS})")

    eng = bench_engine_matmul()

    headers = ["Shape", "PyTorch", "ONNX Runtime", "Engine-AI", "E/P", "E/O"]
    aligns  = ['l', 'r', 'r', 'r', 'r', 'r']
    rows    = []

    for M, K, N, label in MM_SHAPES:
        pt = bench_torch_matmul(M, K, N)
        on = bench_onnx_matmul(M, K, N)
        eg_ms = eng.get(label, None)

        pt_ms = pt['median']
        on_ms = on['median']

        pt_str = fmt_ms_fps(pt_ms)
        on_str = fmt_ms_fps(on_ms)
        eg_str = fmt_ms_fps(eg_ms) if eg_ms else col("   ---   ", C.GRAY)

        ep = (eg_ms / pt_ms) if eg_ms and pt_ms > 0 else 0
        eo = (eg_ms / on_ms) if eg_ms and on_ms > 0 else 0

        rows.append([label, pt_str, on_str, eg_str, ratio_cell(ep), ratio_cell(eo)])

    print(draw_table(headers, rows, aligns))

    print()
    print(f"  {col('Efsane:', C.BOLD)}  "
          f"{col('yeşil', C.GREEN)} = Engine daha hızlı   "
          f"{col('sarı', C.YELLOW)} = ~eşit   "
          f"{col('kırmızı', C.RED)} = Engine daha yavaş")

# ============================================================
#  BÖLÜM 2: BIGDETECTOR
# ============================================================
def section_big_model():
    section_header(2, "BigDetector 640×640  (uçtan uca)")

    if BigDetector is None:
        print(f"  {col('[!]', C.RED)} big_model.py yok, atlanıyor")
        return

    dummy = torch.randn(1, 3, 640, 640)
    results = []

    torch.set_num_threads(THREADS)
    model = BigDetector(num_classes=2).eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(dummy)
        ts = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            _ = model(dummy)
            ts.append((time.perf_counter() - t0) * 1000.0)
    pt = stats_ms(ts)
    results.append(("PyTorch", pt))

    ort = None
    if os.path.exists(REF_ONNX):
        opts = SessionOptions()
        opts.intra_op_num_threads = THREADS
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = InferenceSession(REF_ONNX, opts, providers=["CPUExecutionProvider"])
        dummy_np = dummy.numpy().astype(np.float32)
        for _ in range(WARMUP):
            sess.run(None, {"input": dummy_np})
        ts = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            sess.run(None, {"input": dummy_np})
            ts.append((time.perf_counter() - t0) * 1000.0)
        ort = stats_ms(ts)
        results.append(("ONNX Runtime", ort))

    out = call_engine_exe("benchmark_big_model.exe")
    eng = parse_min_med_max(out)
    if eng:
        results.append(("Engine-AI", eng))

    headers = ["Framework", "Min (ms)", "Median (ms)", "P95 (ms)", "Max (ms)", "FPS"]
    aligns  = ['l', 'r', 'r', 'r', 'r', 'r']
    rows = []
    for name, s in results:
        color = C.CYAN if name == "PyTorch" else (C.MAGENTA if name == "ONNX Runtime" else C.GREEN)
        rows.append([
            col(f"{name:12s}", color + C.BOLD),
            f"{s['min']:8.3f}",
            col(f"{s['median']:8.3f}", C.BOLD),
            f"{s['p95']:8.3f}",
            f"{s['max']:8.3f}",
            fmt_fps(1000.0 / s['median']) if s['median'] > 0 else "  ---",
        ])

    print(draw_table(headers, rows, aligns))

    if eng:
        print()
        print(f"  {col('Karşılaştırma:', C.BOLD)}")
        if ort:
            print(f"    Engine ÷ PyTorch : {ratio_cell(eng['median']/pt['median'])}   "
                  f"({eng['median']:.2f} ms vs {pt['median']:.2f} ms)")
            print(f"    Engine ÷ ONNX    : {ratio_cell(eng['median']/ort['median'])}   "
                  f"({eng['median']:.2f} ms vs {ort['median']:.2f} ms)")
        else:
            print(f"    Engine ÷ PyTorch : {ratio_cell(eng['median']/pt['median'])}   "
                  f"({eng['median']:.2f} ms vs {pt['median']:.2f} ms)")

# ============================================================
#  BÖLÜM 3: MINIDETECTOR
# ============================================================
def section_mini_model():
    section_header(3, "MiniDetector 640×640  (uçtan uca)")

    dummy = torch.randn(1, 3, 640, 640)
    results = []

    torch.set_num_threads(THREADS)
    model = MiniDetector().eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(dummy)
        ts = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            _ = model(dummy)
            ts.append((time.perf_counter() - t0) * 1000.0)
    pt = stats_ms(ts)
    results.append(("PyTorch", pt))

    ort = None
    onnx_path = os.path.join(CONVERTER, "mini_detector.onnx")
    if os.path.exists(onnx_path):
        opts = SessionOptions()
        opts.intra_op_num_threads = THREADS
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        dummy_np = dummy.numpy().astype(np.float32)
        for _ in range(WARMUP):
            sess.run(None, {"input": dummy_np})
        ts = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            sess.run(None, {"input": dummy_np})
            ts.append((time.perf_counter() - t0) * 1000.0)
        ort = stats_ms(ts)
        results.append(("ONNX Runtime", ort))

    out = call_engine_exe("benchmark_detection.exe")
    eng = parse_min_med_max(out)
    if eng:
        results.append(("Engine-AI", eng))

    headers = ["Framework", "Min (ms)", "Median (ms)", "P95 (ms)", "Max (ms)", "FPS"]
    aligns  = ['l', 'r', 'r', 'r', 'r', 'r']
    rows = []
    for name, s in results:
        color = C.CYAN if name == "PyTorch" else (C.MAGENTA if name == "ONNX Runtime" else C.GREEN)
        rows.append([
            col(f"{name:12s}", color + C.BOLD),
            f"{s['min']:8.3f}",
            col(f"{s['median']:8.3f}", C.BOLD),
            f"{s['p95']:8.3f}",
            f"{s['max']:8.3f}",
            fmt_fps(1000.0 / s['median']) if s['median'] > 0 else "  ---",
        ])

    print(draw_table(headers, rows, aligns))

    if eng:
        print()
        print(f"  {col('Karşılaştırma:', C.BOLD)}")
        if ort:
            print(f"    Engine ÷ PyTorch : {ratio_cell(eng['median']/pt['median'])}   "
                  f"({eng['median']:.2f} ms vs {pt['median']:.2f} ms)")
            print(f"    Engine ÷ ONNX    : {ratio_cell(eng['median']/ort['median'])}   "
                  f"({eng['median']:.2f} ms vs {ort['median']:.2f} ms)")

# ============================================================
#  BÖLÜM 4: YOLOv8s  (yeni)
# ============================================================
def section_yolo():
    section_header(4, "YOLOv8s 640×640  (uçtan uca + doğruluk)")

    # ---- Engine benchmark (mevcut exe) ----
    print(f"  {col('→', C.CYAN)} Engine benchmark_yolo.exe çalıştırılıyor...")
    eng_out = call_engine_exe("benchmark_yolo.exe", cwd=PROJECT_ROOT)

    eng = None
    if eng_out:
        m = re.search(r'Engine-AI\s+([\d.]+)\s+ms', eng_out)
        if m:
            med = float(m.group(1))
            eng = {'min': med, 'median': med, 'p95': med, 'max': med}

    # ---- PyTorch ----
    pt = None
    ref_np = None
    dummy = None
    try:
        from ultralytics import YOLO
        print(f"  {col('→', C.CYAN)} PyTorch YOLOv8s yükleniyor...")
        model = YOLO(YOLO_MODEL_PT)
        torch_model = model.model.eval().fuse()

        torch.manual_seed(42)
        dummy = torch.randn(1, 3, 640, 640)
        torch.set_num_threads(THREADS)

        with torch.inference_mode():
            for _ in range(WARMUP):
                _ = torch_model(dummy)
            ts = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                _ = torch_model(dummy)
                ts.append((time.perf_counter() - t0) * 1000.0)
            pt = stats_ms(ts)

            ref = torch_model(dummy)
            if isinstance(ref, (tuple, list)): ref = ref[0]
            ref_np = ref.numpy().astype(np.float32)

        # Input'u dosyaya yaz (engine exe yeniden üretirse tutarlı olsun)
        dummy.numpy().astype(np.float32).tofile(
            os.path.join(CONVERTER, "yolov8s_input.f32"))
        ref_np.tofile(YOLO_PT_OUT)
    except ImportError:
        print(f"  {col('[!]', C.YELLOW)} ultralytics yüklü değil, PyTorch atlanıyor")
    except Exception as e:
        print(f"  {col('[!]', C.RED)} PyTorch hatası: {e}")

    # ---- ONNX Runtime ----
    ort = None
    if os.path.exists(YOLO_ONNX):
        try:
            print(f"  {col('→', C.CYAN)} ONNX Runtime yükleniyor...")
            opts = SessionOptions()
            opts.intra_op_num_threads = THREADS
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = InferenceSession(YOLO_ONNX, opts, providers=["CPUExecutionProvider"])
            dummy_np = (dummy.numpy().astype(np.float32)
                        if dummy is not None
                        else np.random.randn(1, 3, 640, 640).astype(np.float32))
            for _ in range(WARMUP):
                sess.run(None, {"images": dummy_np})
            ts = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                sess.run(None, {"images": dummy_np})
                ts.append((time.perf_counter() - t0) * 1000.0)
            ort = stats_ms(ts)
        except Exception as e:
            print(f"  {col('[!]', C.RED)} ONNX: {e}")

    # ---- Tablo ----
    headers = ["Framework", "Min (ms)", "Median (ms)", "P95 (ms)", "Max (ms)", "FPS", "E/P", "E/O"]
    aligns  = ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
    rows = []
    for name, s, color in [("PyTorch", pt, C.CYAN),
                            ("ONNX Runtime", ort, C.MAGENTA),
                            ("Engine-AI", eng, C.GREEN)]:
        if s is None:
            continue
        ep = (eng['median'] / s['median']) if (eng and s['median'] > 0) else 0
        eo = (eng['median'] / ort['median']) if (eng and ort and ort['median'] > 0) else 0
        rows.append([
            col(f"{name:12s}", color + C.BOLD),
            f"{s['min']:8.3f}",
            col(f"{s['median']:8.3f}", C.BOLD),
            f"{s['p95']:8.3f}",
            f"{s['max']:8.3f}",
            fmt_fps(1000.0 / s['median']) if s['median'] > 0 else "  ---",
            ratio_cell(ep) if ep else "  -  ",
            ratio_cell(eo) if eo else "  -  ",
        ])

    print(draw_table(headers, rows, aligns))

    # ---- Doğruluk ----
    if ref_np is not None and os.path.exists(YOLO_ENG_OUT):
        try:
            eng_np = np.fromfile(YOLO_ENG_OUT, dtype=np.float32)
            if eng_np.size == ref_np.size:
                eng_np = eng_np.reshape(ref_np.shape)
            if eng_np.shape == ref_np.shape:
                diff = np.abs(ref_np - eng_np)
                cos = (np.dot(ref_np.flatten(), eng_np.flatten()) /
                       (np.linalg.norm(ref_np) * np.linalg.norm(eng_np) + 1e-9))
                print(f"\n  {col('Doğruluk (final output):', C.BOLD)}")
                print(f"    max abs diff  = {diff.max():.4e}")
                print(f"    mean abs diff = {diff.mean():.4e}")
                print(f"    cosine sim    = {col(f'{cos:.9f}', C.BOLD + C.CYAN)}")

                # Kanal bazlı cos — bbox / class ayrı
                if ref_np.ndim == 3 and ref_np.shape[0] == 1:
                    r2 = ref_np[0]   # [84, 8400]
                    e2 = eng_np[0]
                    n_ch = r2.shape[0]
                    cos_ch = []
                    for ch in range(n_ch):
                        c = (np.dot(r2[ch], e2[ch]) /
                             (np.linalg.norm(r2[ch]) * np.linalg.norm(e2[ch]) + 1e-9))
                        cos_ch.append(c)
                    cos_ch = np.array(cos_ch)
                    box_cos = cos_ch[:4].mean() if n_ch >= 4 else 0
                    cls_cos = cos_ch[4:].mean() if n_ch > 4 else 0
                    print(f"    bbox cos  (ch 0-3) : {box_cos:.6f}")
                    print(f"    class cos (ch 4+)  : {cls_cos:.6f}")
                    print(f"    min ch cos         : {cos_ch.min():.6f} "
                          f"(ch {cos_ch.argmin()})")
        except Exception as e:
            print(f"  {col('[!]', C.RED)} Doğruluk karşılaştırma: {e}")

# ============================================================
#  BÖLÜM 5: DOĞRULUK TESTİ (BigDetector)
# ============================================================
def section_verify():
    section_header(5, "Doğruluk Testi  (BigDetector, PyTorch vs Engine)")

    if not os.path.exists(REF_PT_OUT):
        print(f"  {col('✗', C.RED)} PyTorch referansı yok: {REF_PT_OUT}")
        return
    if not os.path.exists(REF_ENG_OUT):
        print(f"  {col('✗', C.RED)} Engine çıktısı yok: {REF_ENG_OUT}")
        return

    ref = np.fromfile(REF_PT_OUT, dtype=np.float32)
    eng = np.fromfile(REF_ENG_OUT, dtype=np.float32)

    if ref.shape != eng.shape:
        print(f"  {col('✗', C.RED)} BOYUT UYUŞMUYOR! ref={ref.shape} eng={eng.shape}")
        return

    abs_diff = np.abs(ref - eng)
    max_abs  = abs_diff.max()
    mean_abs = abs_diff.mean()
    denom    = np.maximum(np.abs(ref), 1e-6)
    max_rel  = (abs_diff / denom).max()
    dot      = np.dot(ref.flatten(), eng.flatten())
    norm     = np.linalg.norm(ref) * np.linalg.norm(eng)
    cos_sim  = dot / norm if norm > 0 else 0.0

    if cos_sim > 0.9999 and max_abs < 1e-3:
        verdict, vcolor = "MÜKEMMEL — PyTorch ile aynı çıktı", C.GREEN + C.BOLD
    elif cos_sim > 0.999 and max_abs < 1e-2:
        verdict, vcolor = "İYİ — Küçük sayısal hata", C.GREEN
    elif cos_sim > 0.99:
        verdict, vcolor = "KABUL EDİLEBİLİR", C.YELLOW
    else:
        verdict, vcolor = "BAŞARISIZ — Motor yanlış sonuç veriyor!", C.RED + C.BOLD

    headers = ["Metrik", "Değer"]
    aligns  = ['l', 'r']
    rows = [
        ["Öğe sayısı (ref)",     f"{ref.size:,}"],
        ["Öğe sayısı (engine)",  f"{eng.size:,}"],
        ["Toplam (ref)",         f"{ref.sum():.6f}"],
        ["Toplam (engine)",      f"{eng.sum():.6f}"],
        ["Max abs diff",         f"{max_abs:.3e}"],
        ["Mean abs diff",        f"{mean_abs:.3e}"],
        ["Max rel diff",         f"{max_rel:.3e}"],
        ["Cosine similarity",    col(f"{cos_sim:.9f}", C.BOLD + C.CYAN)],
    ]
    print(draw_table(headers, rows, aligns))
    print()
    print(f"  {col('SONUÇ:', C.BOLD)}  {col(verdict, vcolor)}")

# ============================================================
#  BÖLÜM 6: KATMAN ANALİZİ (PyTorch, doğru peak TFLOPS)
# ============================================================
def _module_type_name(m):
    return type(m).__name__

def _bar(pct, width=28, color=C.GREEN):
    filled = int(round(pct / 100.0 * width))
    filled = max(0, min(width, filled))
    return col("█" * filled, color) + col("░" * (width - filled), C.GRAY)

def _fmt_flops(f):
    if f >= 1e12: return f"{f/1e12:7.2f} T"
    if f >= 1e9:  return f"{f/1e9:7.2f} G"
    if f >= 1e6:  return f"{f/1e6:7.2f} M"
    if f >= 1e3:  return f"{f/1e3:7.2f} K"
    return f"{f:7.0f}"

def profile_pytorch_layers(model, dummy, warmup=10, runs=30):
    timings = {}
    def make_pre():
        def hook(module, inp):
            module._prof_t0 = time.perf_counter()
        return hook
    def make_post(lname):
        def hook(module, inp, out):
            t1 = time.perf_counter()
            t0 = getattr(module, '_prof_t0', t1)
            dt = (t1 - t0) * 1000.0
            if lname not in timings:
                timings[lname] = {'type': _module_type_name(module),
                                  'calls': 0, 'time_ms': 0.0}
            timings[lname]['calls'] += 1
            timings[lname]['time_ms'] += dt
        return hook

    handles = []
    for lname, module in model.named_modules():
        if lname == '' or len(list(module.children())) > 0:
            continue
        handles.append(module.register_forward_pre_hook(make_pre()))
        handles.append(module.register_forward_hook(make_post(lname)))

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        for k in timings:
            timings[k]['time_ms'] = 0.0
            timings[k]['calls'] = 0
        for _ in range(runs):
            _ = model(dummy)

    for h in handles:
        h.remove()

    for k in timings:
        timings[k]['time_ms'] /= max(1, runs)
        timings[k]['calls'] //= max(1, runs)
    return timings

def analyze_flops(model, dummy):
    flops = {}
    def make_hook(lname):
        def hook(module, inp, out):
            f = 0
            if isinstance(module, nn.Conv2d):
                N, Cin, _, _ = inp[0].shape
                _, Cout, Hout, Wout = out.shape
                KH, KW = module.kernel_size
                f = 2 * N * Cout * Hout * Wout * Cin * KH * KW / module.groups
            elif isinstance(module, nn.Linear):
                K = inp[0].shape[-1]
                M = inp[0].numel() // K
                N = out.shape[-1]
                f = 2 * M * K * N
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                f = 2 * out.numel()
            elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh,
                                     nn.Hardswish, nn.SiLU)):
                f = out.numel()
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d,
                                     nn.AdaptiveAvgPool2d)):
                f = out.numel() * 4
            flops[lname] = {'type': type(module).__name__, 'flops': f}
        return hook

    handles = []
    for lname, module in model.named_modules():
        if lname == '' or len(list(module.children())) > 0:
            continue
        handles.append(module.register_forward_hook(make_hook(lname)))

    with torch.no_grad():
        _ = model(dummy)
    for h in handles:
        h.remove()
    return flops

def _print_layer_table(timings, flops, model_name, peak_tflops=None):
    if peak_tflops is None:
        peak_tflops = PEAK_TFLOPS

    agg = {}
    for lname, t in timings.items():
        typ = t['type']
        if typ not in agg:
            agg[typ] = {'time_ms': 0.0, 'flops': 0, 'calls': 0, 'layers': 0}
        agg[typ]['time_ms'] += t['time_ms']
        agg[typ]['calls']   += t['calls']
        agg[typ]['layers']  += 1
        if lname in flops:
            agg[typ]['flops'] += flops[lname]['flops']

    total_time  = sum(v['time_ms'] for v in agg.values())
    total_flops = sum(v['flops']   for v in agg.values())
    if total_time <= 0:
        print(f"  {col('[!]', C.RED)} {model_name}: ölçüm yok")
        return

    items = sorted(agg.items(), key=lambda kv: kv[1]['time_ms'], reverse=True)

    headers = ["Op Tipi", "Katman", "Çağrı", "Süre(ms)", "Zaman %",
               "FLOPs", "FLOP %", "Teorik Min(ms)", "Verim", ""]
    aligns  = ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'l']

    rows = []
    for typ, v in items:
        pct  = v['time_ms'] / total_time * 100
        fpct = v['flops'] / total_flops * 100 if total_flops > 0 else 0
        # Teorik min = FLOPs / peak_TFLOPs  (ms cinsine çevir)
        theo_ms = v['flops'] / (peak_tflops * 1e12) * 1000.0
        # Verim = teorik / gerçek  (yüzde)
        eff = (theo_ms / v['time_ms'] * 100) if v['time_ms'] > 0 else 0
        eff_txt = f"{eff:5.1f}%"
        if eff >= 60:
            eff_txt = col(eff_txt, C.GREEN + C.BOLD)
        elif eff >= 30:
            eff_txt = col(eff_txt, C.YELLOW)
        else:
            eff_txt = col(eff_txt, C.RED)

        rows.append([
            typ,
            f"{v['layers']:3d}",
            f"{v['calls']:3d}",
            f"{v['time_ms']:8.3f}",
            f"{pct:5.1f}%",
            _fmt_flops(v['flops']),
            f"{fpct:5.1f}%",
            f"{theo_ms:8.3f}",
            eff_txt,
            _bar(pct, 24),
        ])

    total_theo = total_flops / (peak_tflops * 1e12) * 1000
    total_eff = (total_theo / total_time * 100) if total_time > 0 else 0

    rows.append([
        col("TOPLAM", C.BOLD), "", "", 
        col(f"{total_time:8.3f}", C.BOLD),
        col("100.0%", C.BOLD),
        col(_fmt_flops(total_flops), C.BOLD),
        col("100.0%", C.BOLD),
        col(f"{total_theo:8.3f}", C.BOLD),
        col(f"{total_eff:5.1f}%", C.BOLD),
        "",
    ])

    print(draw_table(headers, rows, aligns))
    print(f"  {col('Not:', C.DIM)} CPU: {CPU_NAME}")
    print(f"  {col('      ', C.DIM)} Fiziksel çekirdek: {PHYS_CORES}, "
          f"boost: {BOOST_GHZ:.1f} GHz, "
          f"AVX2 peak: {peak_tflops:.2f} TFLOPS "
          f"(= {PHYS_CORES} × 16 FLOP/cyc × {BOOST_GHZ:.1f} GHz)")
    print(f"  {col('      ', C.DIM)} Verim = Teorik Min / Gerçek. "
          f"≥%60 {col('iyi', C.GREEN)}, %30-60 {col('orta', C.YELLOW)}, "
          f"<%30 {col('düşük', C.RED)}")

def _print_top_layers(timings, top=15):
    if not timings:
        return
    items = sorted(timings.items(), key=lambda kv: kv[1]['time_ms'], reverse=True)[:top]
    total = sum(v['time_ms'] for v in timings.values()) or 1.0

    headers = ["#", "Katman", "Tip", "Süre(ms)", "Zaman %", ""]
    aligns  = ['r', 'l', 'l', 'r', 'r', 'l']
    rows = []
    for i, (lname, v) in enumerate(items, 1):
        pct = v['time_ms'] / total * 100
        rows.append([
            f"{i:2d}", lname, v['type'],
            f"{v['time_ms']:8.3f}",
            f"{pct:5.1f}%",
            _bar(pct, 24),
        ])
    print(draw_table(headers, rows, aligns))

def section_layer_analysis():
    section_header(6, "Katman / Kademe Analizi  (PyTorch, dinamik peak)")

    print(f"\n  {col('CPU:', C.BOLD)} {CPU_NAME}")
    print(f"  {col('Peak:', C.BOLD)} {PEAK_TFLOPS:.2f} TFLOPS "
          f"({PHYS_CORES}C × 16 FLOP/cyc × {BOOST_GHZ:.1f} GHz)")

    # --- MiniDetector ---
    print(f"\n  {col('▶ MiniDetector 640×640', C.BOLD + C.MAGENTA)}")
    try:
        torch.manual_seed(42)
        mini = MiniDetector().eval()
        dummy = torch.randn(1, 3, 640, 640)
        t_mini = profile_pytorch_layers(mini, dummy, warmup=10, runs=30)
        f_mini = analyze_flops(mini, dummy)
        _print_layer_table(t_mini, f_mini, "MiniDetector")
        print(f"\n  {col('  En pahalı katmanlar:', C.DIM)}")
        _print_top_layers(t_mini, top=10)
    except Exception as e:
        print(f"  {col('[!]', C.RED)} MiniDetector analiz: {e}")

    # --- BigDetector ---
    if BigDetector is None:
        return
    print(f"\n  {col('▶ BigDetector 640×640', C.BOLD + C.MAGENTA)}")
    try:
        torch.manual_seed(42)
        big = BigDetector(num_classes=2).eval()
        dummy = torch.randn(1, 3, 640, 640)
        t_big = profile_pytorch_layers(big, dummy, warmup=10, runs=30)
        f_big = analyze_flops(big, dummy)
        _print_layer_table(t_big, f_big, "BigDetector")
        print(f"\n  {col('  En pahalı katmanlar:', C.DIM)}")
        _print_top_layers(t_big, top=15)
    except Exception as e:
        print(f"  {col('[!]', C.RED)} BigDetector analiz: {e}")

# ============================================================
#  BÖLÜM 7: ENGINE vs PyTorch — KATMAN KATMAN
# ============================================================
def _profile_pytorch_ordered(model, dummy, warmup=5, runs=20):
    samples = {}
    def make_pre():
        def hook(module, inp):
            module._prof_t0 = time.perf_counter()
        return hook
    def make_post(name):
        def hook(module, inp, out):
            t1 = time.perf_counter()
            t0 = getattr(module, '_prof_t0', t1)
            dt = (t1 - t0) * 1000.0
            samples.setdefault(name, []).append(dt)
        return hook

    handles = []
    for lname, mod in model.named_modules():
        if lname == '' or len(list(mod.children())) > 0:
            continue
        handles.append(mod.register_forward_pre_hook(make_pre()))
        handles.append(mod.register_forward_hook(make_post(lname)))

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        for name in samples:
            samples[name] = []
        for _ in range(runs):
            _ = model(dummy)

    for h in handles:
        h.remove()

    order_hits = []
    def make_order_hook(name):
        def hook(module, inp, out):
            order_hits.append(name)
        return hook
    handles = []
    for lname, mod in model.named_modules():
        if lname == '' or len(list(mod.children())) > 0:
            continue
        handles.append(mod.register_forward_hook(make_order_hook(lname)))
    with torch.no_grad():
        _ = model(dummy)
    for h in handles:
        h.remove()

    name_to_mod = dict(model.named_modules())
    result = []
    for i, name in enumerate(order_hits):
        mod = name_to_mod[name]
        times = sorted(samples.get(name, [0.0]))
        med = times[len(times) // 2] if times else 0.0
        result.append({
            'order': i, 'name': name,
            'type': type(mod).__name__,
            'median_ms': med,
        })
    return result

def _aggregate_engine_csv(path):
    buckets = {}
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            key = (int(row['node_idx']), row['op_type'], row['dims'])
            buckets.setdefault(key, []).append(float(row['time_ms']))
    agg = []
    for (idx, typ, dims), times in sorted(buckets.items()):
        agg.append({
            'order': idx, 'type': typ, 'dims': dims,
            'median_ms': _median(times),
        })
    return agg

def section_engine_vs_pytorch():
    section_header(7, "Engine vs PyTorch — Katman Katman")

    if BigDetector is None:
        print(f"  {col('[!]', C.YELLOW)} BigDetector yok, atlanıyor")
        return

    csv_path = os.path.join(RELEASE_BIN, "engine_layer_times.csv")
    if os.path.exists(csv_path):
        try: os.remove(csv_path)
        except OSError: pass

    print(f"  {col('→', C.CYAN)} CSV üretiliyor: benchmark_big_model.exe")
    env = {"ENGINE_PROFILE_CSV": csv_path}
    out = call_engine_exe("benchmark_big_model.exe", env=env)
    if out is None or not os.path.exists(csv_path):
        print(f"  {col('[!]', C.RED)} CSV oluşmadı")
        return

    try:
        torch.manual_seed(42)
        big = BigDetector(num_classes=2).eval()
        dummy = torch.randn(1, 3, 640, 640)
        pt_layers = _profile_pytorch_ordered(big, dummy, warmup=5, runs=20)
        eng_layers = _aggregate_engine_csv(csv_path)

        headers = ["#", "PyTorch Katman", "PyTorch(ms)",
                   "Engine Op", "Engine(ms)", "E/P", "Δ(ms)"]
        aligns = ['r', 'l', 'r', 'l', 'r', 'r', 'r']
        rows = []
        total_pt = 0.0
        total_eng = 0.0

        pt_convs = [l for l in pt_layers if l['type'] == 'Conv2d']
        eng_convs = [l for l in eng_layers if l['type'] == 'Conv2d']

        n = min(len(pt_convs), len(eng_convs))
        for i in range(n):
            pl = pt_convs[i]
            el = eng_convs[i]
            ratio = el['median_ms'] / pl['median_ms'] if pl['median_ms'] > 0 else 0
            delta = el['median_ms'] - pl['median_ms']
            total_pt  += pl['median_ms']
            total_eng += el['median_ms']
            rows.append([
                f"{i:2d}", pl['name'],
                f"{pl['median_ms']:8.3f}",
                f"{el['type']} {el['dims']}",
                f"{el['median_ms']:8.3f}",
                ratio_cell(ratio), f"{delta:+7.3f}",
            ])
        rows.append([
            col("TOPLAM", C.BOLD), "",
            col(f"{total_pt:8.3f}", C.BOLD), "",
            col(f"{total_eng:8.3f}", C.BOLD),
            ratio_cell(total_eng / total_pt) if total_pt > 0 else "-",
            f"{total_eng - total_pt:+7.3f}",
        ])
        print(draw_table(headers, rows, aligns))
    except Exception as e:
        print(f"  {col('[!]', C.RED)} Hata: {e}")

# ============================================================
#  ANA BLOK
# ============================================================
def main():
    global USE_COLOR
    parser = argparse.ArgumentParser(description="Engine-AI vs PyTorch vs ONNX RT")
    parser.add_argument("--only",
                        choices=["prep", "matmul", "big", "mini", "yolo",
                                 "verify", "layers", "layer_vs", "all"],
                        default="all")
    parser.add_argument("--skip-prep", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        USE_COLOR = False

    t_start = time.perf_counter()

    w = 74
    print(col("╔" + "═" * w + "╗", C.CYAN))
    print(col("║", C.CYAN) + col("  ENGINE-AI  vs  PyTorch  vs  ONNX Runtime".center(w), C.BOLD) + col("║", C.CYAN))
    print(col("║", C.CYAN) + f"  Threads: {THREADS}   Warmup: {WARMUP}   Runs: {RUNS}".center(w) + col("║", C.CYAN))
    print(col("║", C.CYAN) + f"  CPU: {CPU_NAME[:40]}".ljust(w)[:w] + col("║", C.CYAN))
    print(col("║", C.CYAN) + f"  Peak (AVX2): {PEAK_TFLOPS:.2f} TFLOPS  ({PHYS_CORES}C × {BOOST_GHZ:.1f}GHz)".ljust(w)[:w] + col("║", C.CYAN))
    print(col("╚" + "═" * w + "╝", C.CYAN))

    if args.only in ("all", "prep") and not args.skip_prep:
        try: prepare_reference()
        except Exception as e: print(f"\n{col('[!]', C.RED)} {e}")

    if args.only in ("all", "matmul"):
        try: section_matmul()
        except Exception as e: print(f"\n{col('[!]', C.RED)} {e}")

    if args.only in ("all", "big"):
        try: section_big_model()
        except Exception as e: print(f"\n{col('[!]', C.RED)} {e}")

    if args.only in ("all", "mini"):
        try: section_mini_model()
        except Exception as e: print(f"\n{col('[!]', C.RED)} {e}")

    if args.only in ("all", "yolo"):
        try: section_yolo()
        except Exception as e: print(f"\n{col('[!]', C.RED)} {e}")

    if args.only in ("all", "verify"):
        try: section_verify()
        except Exception as e: print(f"\n{col('[!]', C.RED)} {e}")

    if args.only in ("all", "layers"):
        try: section_layer_analysis()
        except Exception as e: print(f"\n{col('[!]', C.RED)} {e}")

    if args.only in ("all", "layer_vs"):
        try: section_engine_vs_pytorch()
        except Exception as e: print(f"\n{col('[!]', C.RED)} {e}")

    elapsed = time.perf_counter() - t_start
    print()
    print(col("╔" + "═" * w + "╗", C.CYAN))
    print(col("║", C.CYAN) + f"  TAMAMLANDI  —  Toplam süre: {elapsed:.1f} sn".ljust(w) + col("║", C.CYAN))
    print(col("╚" + "═" * w + "╝", C.CYAN))


if __name__ == "__main__":
    main()