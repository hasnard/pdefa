# pdefa - AGPL-3.0-or-later - see LICENSE
"""
Model compatibility & performance benchmark.

Compares three backends across a wide range of models:
    1. PyTorch (reference)
    2. ONNX Runtime (CPU)
    3. pdefa engine (.engine)

Stages:
    1 - YOLO family        (v8/v9/v10/v11/v12, n/s/m/l/x)
    2 - Classic models     (ResNet, VGG, MobileNet, DenseNet, ...)
    3 - Modern backbones   (ConvNeXt, Swin, ViT, RegNet, MaxViT)
    4 - Segmentation       (FCN, DeepLabV3, LR-ASPP)
    C - Custom             (comma-separated model names)
    A - All stages
    Q - Quit

Dependencies:
    pip install -e ".[bench]"
    # or: pip install torch torchvision ultralytics onnxruntime numpy

Usage:
    python benchmarks/bench_models.py
    # or with custom model cache:
    set PDEFA_BENCH_MODELS=D:\\my\\models
    python benchmarks/bench_models.py
"""
from __future__ import annotations

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torchvision.models as tvm
import onnxruntime as ort

import pdefa


# ============================================================
# Configuration
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]   # pdefa/
DEFAULT_MODELS_DIR = Path(
    os.environ.get("PDEFA_BENCH_MODELS", str(REPO_ROOT / "bench_models"))
)
WORK_DIR = DEFAULT_MODELS_DIR
RESULT_DIR = WORK_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CONVERTER = REPO_ROOT / "PDEFA1.0" / "tools" / "converter" / "onnx_to_engine.py"

WARMUP = 5
RUNS = 30
THREADS = int(os.environ.get("PDEFA_BENCH_THREADS", "12"))

torch.set_num_threads(THREADS)


# ============================================================
# Stage 1 — YOLO family
# ============================================================
YOLO_MODELS = {
    # name: (size_mb_hint, input_shape)
    "yolov8n": (6,   (1, 3, 640, 640)),
    "yolov8s": (22,  (1, 3, 640, 640)),
    "yolov8m": (50,  (1, 3, 640, 640)),
    "yolov8l": (87,  (1, 3, 640, 640)),
    "yolov8x": (131, (1, 3, 640, 640)),
    "yolov9t": (8,   (1, 3, 640, 640)),
    "yolov9s": (26,  (1, 3, 640, 640)),
    "yolov9m": (46,  (1, 3, 640, 640)),
    "yolov9c": (50,  (1, 3, 640, 640)),
    "yolov9e": (113, (1, 3, 640, 640)),
    # NOTE: YOLOv10 uses TopK + GatherElements (slow path)
    "yolov10n": (6,  (1, 3, 640, 640)),
    "yolov10s": (16, (1, 3, 640, 640)),
    "yolov10m": (32, (1, 3, 640, 640)),
    "yolov10b": (32, (1, 3, 640, 640)),
    "yolov10l": (32, (1, 3, 640, 640)),
    "yolov10x": (48, (1, 3, 640, 640)),
    "yolo11n": (6,   (1, 3, 640, 640)),
    "yolo11s": (19,  (1, 3, 640, 640)),
    "yolo11m": (40,  (1, 3, 640, 640)),
    "yolo11l": (50,  (1, 3, 640, 640)),
    "yolo11x": (110, (1, 3, 640, 640)),
    # NOTE: YOLO12 includes attention ops
    "yolo12n": (6,   (1, 3, 640, 640)),
    "yolo12s": (18,  (1, 3, 640, 640)),
    "yolo12m": (30,  (1, 3, 640, 640)),
    "yolo12l": (30,  (1, 3, 640, 640)),
    "yolo12x": (42,  (1, 3, 640, 640)),
}


# ============================================================
# Stage 2 — Classic models
# ============================================================
CLASSIC_MODELS = {
    "resnet18":        ("tv", lambda: tvm.resnet18(weights="DEFAULT"),           (1, 3, 224, 224)),
    "resnet34":        ("tv", lambda: tvm.resnet34(weights="DEFAULT"),           (1, 3, 224, 224)),
    "resnet50":        ("tv", lambda: tvm.resnet50(weights="DEFAULT"),           (1, 3, 224, 224)),
    "mobilenet_v2":    ("tv", lambda: tvm.mobilenet_v2(weights="DEFAULT"),       (1, 3, 224, 224)),
    "mobilenet_v3_s":  ("tv", lambda: tvm.mobilenet_v3_small(weights="DEFAULT"), (1, 3, 224, 224)),
    "mobilenet_v3_l":  ("tv", lambda: tvm.mobilenet_v3_large(weights="DEFAULT"), (1, 3, 224, 224)),
    "vgg16":           ("tv", lambda: tvm.vgg16(weights="DEFAULT"),              (1, 3, 224, 224)),
    "vgg19":           ("tv", lambda: tvm.vgg19(weights="DEFAULT"),              (1, 3, 224, 224)),
    "densenet121":     ("tv", lambda: tvm.densenet121(weights="DEFAULT"),        (1, 3, 224, 224)),
    "densenet169":     ("tv", lambda: tvm.densenet169(weights="DEFAULT"),        (1, 3, 224, 224)),
    "squeezenet1_0":   ("tv", lambda: tvm.squeezenet1_0(weights="DEFAULT"),      (1, 3, 224, 224)),
    "squeezenet1_1":   ("tv", lambda: tvm.squeezenet1_1(weights="DEFAULT"),      (1, 3, 224, 224)),
    "efficientnet_b0": ("tv", lambda: tvm.efficientnet_b0(weights="DEFAULT"),    (1, 3, 224, 224)),
    "efficientnet_b1": ("tv", lambda: tvm.efficientnet_b1(weights="DEFAULT"),    (1, 3, 224, 224)),
    "shufflenet_v2":   ("tv", lambda: tvm.shufflenet_v2_x1_0(weights="DEFAULT"), (1, 3, 224, 224)),
}


# ============================================================
# Stage 3 — Modern backbones
# ============================================================
MODERN_MODELS = {
    "convnext_t":     ("tv", lambda: tvm.convnext_tiny(weights="DEFAULT"),       (1, 3, 224, 224)),
    "convnext_s":     ("tv", lambda: tvm.convnext_small(weights="DEFAULT"),      (1, 3, 224, 224)),
    "convnext_b":     ("tv", lambda: tvm.convnext_base(weights="DEFAULT"),       (1, 3, 224, 224)),
    "swin_t":         ("tv", lambda: tvm.swin_t(weights="DEFAULT"),              (1, 3, 224, 224)),
    "swin_s":         ("tv", lambda: tvm.swin_s(weights="DEFAULT"),              (1, 3, 224, 224)),
    "swin_b":         ("tv", lambda: tvm.swin_b(weights="DEFAULT"),              (1, 3, 224, 224)),
    "swin_v2_t":      ("tv", lambda: tvm.swin_v2_t(weights="DEFAULT"),           (1, 3, 256, 256)),
    "vit_b_16":       ("tv", lambda: tvm.vit_b_16(weights="DEFAULT"),            (1, 3, 224, 224)),
    "vit_b_32":       ("tv", lambda: tvm.vit_b_32(weights="DEFAULT"),            (1, 3, 224, 224)),
    "vit_l_16":       ("tv", lambda: tvm.vit_l_16(weights="DEFAULT"),            (1, 3, 224, 224)),
    "maxvit_t":       ("tv", lambda: tvm.maxvit_t(weights="DEFAULT"),            (1, 3, 224, 224)),
    "regnet_y_400mf": ("tv", lambda: tvm.regnet_y_400mf(weights="DEFAULT"),      (1, 3, 224, 224)),
    "regnet_y_800mf": ("tv", lambda: tvm.regnet_y_800mf(weights="DEFAULT"),      (1, 3, 224, 224)),
    "regnet_y_1_6gf": ("tv", lambda: tvm.regnet_y_1_6gf(weights="DEFAULT"),      (1, 3, 224, 224)),
}


# ============================================================
# Stage 4 — Segmentation
# ============================================================
SEG_MODELS = {
    "fcn_resnet50":   ("tv", lambda: tvm.segmentation.fcn_resnet50(weights="DEFAULT"),           (1, 3, 520, 520)),
    "fcn_resnet101":  ("tv", lambda: tvm.segmentation.fcn_resnet101(weights="DEFAULT"),          (1, 3, 520, 520)),
    "deeplabv3_r50":  ("tv", lambda: tvm.segmentation.deeplabv3_resnet50(weights="DEFAULT"),     (1, 3, 520, 520)),
    "deeplabv3_r101": ("tv", lambda: tvm.segmentation.deeplabv3_resnet101(weights="DEFAULT"),    (1, 3, 520, 520)),
    "lraspp_mnv3":    ("tv", lambda: tvm.segmentation.lraspp_mobilenet_v3_large(weights="DEFAULT"), (1, 3, 520, 520)),
}


# ============================================================
# Stage registry
# ============================================================
STAGES = {
    "1": ("YOLO family",      "yolo", YOLO_MODELS),
    "2": ("Classic models",   "tv",   CLASSIC_MODELS),
    "3": ("Modern backbones", "tv",   MODERN_MODELS),
    "4": ("Segmentation",     "tv",   SEG_MODELS),
}


# ============================================================
# Export + convert
# ============================================================
def ensure_onnx(name: str, kind: str, model_fn, input_shape) -> Path:
    """Export or reuse an ONNX file for the given model."""
    onnx_path = WORK_DIR / f"{name}.onnx"
    if onnx_path.exists() and onnx_path.stat().st_size > 1024:
        return onnx_path

    print(f"    [->] ONNX export: {name}")
    if kind == "yolo":
        from ultralytics import YOLO
        cwd = os.getcwd()
        os.chdir(WORK_DIR)
        try:
            y = YOLO(f"{name}.pt")
            y.export(format="onnx", imgsz=640, opset=18,
                     dynamic=False, simplify=True)
        finally:
            os.chdir(cwd)
    else:
        model = model_fn().eval()
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            torch.onnx.export(
                model, dummy, str(onnx_path),
                input_names=["input"], output_names=["output"],
                opset_version=18, dynamo=False,
                do_constant_folding=True,
            )
    return onnx_path


def ensure_engine(onnx_path: Path) -> Path:
    """Convert ONNX -> .engine (or reuse existing)."""
    engine_path = onnx_path.with_suffix(".engine")
    if engine_path.exists() and engine_path.stat().st_size > 4096:
        return engine_path

    if not CONVERTER.exists():
        raise FileNotFoundError(f"converter not found: {CONVERTER}")

    print(f"    [->] Convert: {onnx_path.name}")
    r = subprocess.run(
        [sys.executable, str(CONVERTER), str(onnx_path), str(engine_path)],
        capture_output=True, text=True, timeout=900,
    )
    if r.returncode != 0 or not engine_path.exists():
        tail = (r.stdout or "")[-300:]
        raise RuntimeError(f"convert failed: {tail}")
    return engine_path


# ============================================================
# Benchmark helpers
# ============================================================
def bench(fn, warmup: int = WARMUP, runs: int = RUNS) -> dict:
    fn()  # warm once
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        "median": ts[len(ts) // 2],
        "min":    ts[0],
        "max":    ts[-1],
        "p95":    ts[int(0.95 * len(ts))],
    }


def bench_pytorch(kind: str, model_fn, input_shape) -> dict:
    if kind == "yolo":
        from ultralytics import YOLO
        m = YOLO(str(model_fn)).model.eval()
    else:
        m = model_fn().eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        return bench(lambda: m(x))


def bench_ort(onnx_path: Path) -> dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path), opts, providers=["CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0]
    shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in inp.shape)
    x = np.random.randn(*shape).astype(np.float32)
    return bench(lambda: sess.run(None, {inp.name: x}))


def bench_engine(engine_path: Path) -> dict:
    ctx = pdefa.Context()
    ctx.set_num_threads(THREADS)
    model = pdefa.Model(ctx, str(engine_path))
    session = pdefa.Session(ctx, model)
    info = model.input_info(0)
    t = pdefa.randn(*info.shape, seed=0)  # numpy-free
    session.set_input(0, t)
    return bench(lambda: session.run())


# ============================================================
# Stage runner
# ============================================================
def run_stage(stage_name: str, models_dict: dict, kind: str) -> list:
    print(f"\n{'#' * 80}")
    print(f"#  STAGE: {stage_name}  ({len(models_dict)} models)")
    print(f"{'#' * 80}")

    results = []
    for name, spec in models_dict.items():
        if kind == "yolo":
            mfn, shape = spec
            pt_model_ref = WORK_DIR / f"{name}.pt"
        else:
            _, mfn, shape = spec
            pt_model_ref = mfn

        print(f"\n--- {name} ---")
        r = {"name": name, "pt": None, "ort": None, "eng": None, "err": ""}

        # PyTorch
        try:
            print("  [1/3] PyTorch...")
            r["pt"] = bench_pytorch(kind, pt_model_ref, shape)
            print(f"        {r['pt']['median']:.2f} ms")
        except Exception as e:
            r["err"] += f"PT:{type(e).__name__} "
            print(f"        X {e}")

        # ONNX export
        try:
            onnx_path = ensure_onnx(
                name, kind,
                pt_model_ref if kind != "yolo" else None,
                shape,
            )
        except Exception as e:
            r["err"] += "export "
            print(f"  X export: {e}")
            results.append(r)
            continue

        # ONNX Runtime
        try:
            print("  [2/3] ONNX RT...")
            r["ort"] = bench_ort(onnx_path)
            print(f"        {r['ort']['median']:.2f} ms")
        except Exception as e:
            r["err"] += f"ORT:{type(e).__name__} "
            print(f"        X {e}")

        # pdefa engine
        try:
            engine_path = ensure_engine(onnx_path)
            print("  [3/3] pdefa engine...")
            r["eng"] = bench_engine(engine_path)
            print(f"        {r['eng']['median']:.2f} ms")
        except Exception as e:
            r["err"] += f"ENG:{type(e).__name__} "
            print(f"        X {e}")

        results.append(r)

    return results


# ============================================================
# Summary
# ============================================================
def print_summary(results: list) -> None:
    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")
    header = (
        f"  {'Model':<18} {'PyTorch':>10} {'ONNX RT':>10} {'Engine':>10} "
        f"{'E/PT':>7} {'E/ORT':>7} {'Spread':>8}"
    )
    print(header)
    print(
        f"  {'-' * 18} {'-' * 10} {'-' * 10} {'-' * 10} "
        f"{'-' * 7} {'-' * 7} {'-' * 8}"
    )

    wins = {"pt": 0, "ort": 0, "eng": 0}
    for r in results:
        pt   = r["pt"]["median"]  if r["pt"]  else None
        ort_ = r["ort"]["median"] if r["ort"] else None
        eng  = r["eng"]["median"] if r["eng"] else None

        def fmt(x):
            return f"{x:7.2f}ms" if x else "    N/A"

        def ratio(a, b):
            return f"{a / b:5.2f}x" if (a and b) else "  -  "

        spread = ""
        if r["eng"]:
            d = (r["eng"]["max"] - r["eng"]["min"]) / r["eng"]["min"] * 100
            spread = f"{d:6.1f}%"

        row = f"  {r['name']:<18} {fmt(pt):>10} {fmt(ort_):>10} {fmt(eng):>10} "
        row += f"{ratio(eng, pt):>7} {ratio(eng, ort_):>7} {spread:>8}"
        if r["err"]:
            row += f"  [{r['err'].strip()}]"
        print(row)

        times = [(x, k) for x, k in [(pt, "pt"), (ort_, "ort"), (eng, "eng")] if x]
        if times:
            wins[min(times)[1]] += 1

    print(f"\n  Wins: PyTorch={wins['pt']}  ONNX RT={wins['ort']}  Engine={wins['eng']}")
    print(f"{'=' * 90}\n")


def save_results(results: list, stage_name: str) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULT_DIR / f"{stage_name}_{ts}.json"
    path.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  [i] Saved: {path}")


# ============================================================
# Menu
# ============================================================
def menu() -> str:
    print(f"\n{'=' * 70}")
    print("  PDEFA BENCHMARK MENU")
    print(f"{'=' * 70}")
    print("  [1] YOLO family       — yolov8 n/s/m/l/x, v9, v10, v11, v12")
    print("  [2] Classic models    — ResNet, VGG, MobileNet, DenseNet, ...")
    print("  [3] Modern backbones  — ConvNeXt, Swin, ViT, RegNet, MaxViT")
    print("  [4] Segmentation      — FCN, DeepLabV3, LR-ASPP")
    print("  [A] All stages")
    print("  [C] Custom            — type your own model names")
    print("  [Q] Quit")
    print(f"{'=' * 70}")
    return input("  Choice: ").strip().upper()


def custom_menu() -> dict:
    print("\n  Enter model names separated by commas.")
    print("  Example: yolov8x,convnext_b,vit_l_16")
    print("  Note: both torchvision and ultralytics models are supported.")
    raw = input("  Models: ").strip()
    names = [n.strip() for n in raw.split(",") if n.strip()]

    custom = {}
    for name in names:
        for src in (YOLO_MODELS, CLASSIC_MODELS, MODERN_MODELS, SEG_MODELS):
            if name in src:
                custom[name] = src[name]
                break
        else:
            # Unknown name -> assume YOLO if it looks like one
            if name.startswith("yolo"):
                custom[name] = ((1, 3, 640, 640), (1, 3, 640, 640))
            else:
                print(f"  [warn] unknown model, skipped: {name}")

    return custom


# ============================================================
# Main
# ============================================================
def main() -> int:
    if sys.platform == "win32":
        os.environ["PYTHONIOENCODING"] = "utf-8"

    while True:
        choice = menu()

        if choice == "Q":
            print("  Bye!")
            return 0

        try:
            if choice == "1":
                results = run_stage(*STAGES["1"])
                print_summary(results)
                save_results(results, "yolo")
            elif choice == "2":
                results = run_stage(*STAGES["2"])
                print_summary(results)
                save_results(results, "classic")
            elif choice == "3":
                results = run_stage(*STAGES["3"])
                print_summary(results)
                save_results(results, "modern")
            elif choice == "4":
                results = run_stage(*STAGES["4"])
                print_summary(results)
                save_results(results, "segmentation")
            elif choice == "A":
                for key in ("1", "2", "3", "4"):
                    name, kind, models = STAGES[key]
                    results = run_stage(name, models, kind)
                    print_summary(results)
                    save_results(results, key)
            elif choice == "C":
                custom = custom_menu()
                if not custom:
                    print("  Empty list.")
                    continue
                kind = "yolo" if any(k.startswith("yolo") for k in custom) else "tv"
                results = run_stage("Custom", custom, kind)
                print_summary(results)
                save_results(results, "custom")
            else:
                print("  Invalid choice!")
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            return 130


if __name__ == "__main__":
    sys.exit(main())