# Benchmarks

Scripts here require heavy dependencies and are **not run in CI**.
Run them manually.

## Dependencies

```bash
pip install -e ".[bench]"
# or manually:
pip install numpy torch torchvision ultralytics onnxruntime
```

## Scripts

| File | Purpose |
|---|---|
| `numerical_vs_pytorch.py` | Single-model numerical comparison (cosine, max diff) |
| `diagnose_channel.py` | Per-channel error diagnosis |
| `bench_ops.py` | Ops micro-benchmark (matmul, softmax, ...) |
| `bench_models.py` | Full model matrix benchmark (50+ models, 3 backends) |

---

## `bench_models.py`

Interactive benchmark that compares **three backends** across a wide
range of models:

1. **PyTorch** — reference timings
2. **ONNX Runtime** — CPU provider
3. **pdefa engine** — our `.engine` format

### Stages

| Stage | Content |
|---|---|
| 1 | **YOLO family** — yolov8/v9/v10/v11/v12 (n/s/m/l/x) |
| 2 | **Classic models** — ResNet, VGG, MobileNet, DenseNet, SqueezeNet, EfficientNet, ShuffleNet |
| 3 | **Modern backbones** — ConvNeXt, Swin, ViT, RegNet, MaxViT |
| 4 | **Segmentation** — FCN, DeepLabV3, LR-ASPP |
| C | **Custom** — comma-separated model names |
| A | **All stages** |

### Usage

```bash
pip install -e ".[bench]"
python benchmarks/bench_models.py
```

You'll get an interactive menu:

```
======================================================================
  PDEFA BENCHMARK MENU
======================================================================
  [1] YOLO family       — yolov8 n/s/m/l/x, v9, v10, v11, v12
  [2] Classic models    — ResNet, VGG, MobileNet, DenseNet, ...
  [3] Modern backbones  — ConvNeXt, Swin, ViT, RegNet, MaxViT
  [4] Segmentation      — FCN, DeepLabV3, LR-ASPP
  [A] All stages
  [C] Custom            — type your own model names
  [Q] Quit
======================================================================
  Choice:
```

### Configuration

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PDEFA_BENCH_MODELS` | `<repo>/bench_models` | Directory for ONNX/`.engine` cache |
| `PDEFA_BENCH_THREADS` | `12` | Thread count for all backends |

Example:

```powershell
$env:PDEFA_BENCH_MODELS = "D:\my\models"
$env:PDEFA_BENCH_THREADS = "8"
python benchmarks/bench_models.py
```

```bash
PDEFA_BENCH_MODELS=/data/models PDEFA_BENCH_THREADS=8 \
    python benchmarks/bench_models.py
```

### Output

Results are printed to stdout and also saved as JSON:

```
bench_models/results/yolo_20260929_103200.json
```

Each entry contains `median`, `min`, `max`, and `p95` timings (ms)
for each backend, plus any error encountered during export or inference.

---

## `numerical_vs_pytorch.py`

Single-model numerical comparison.

```bash
python numerical_vs_pytorch.py model.engine yolov8n.pt image.jpg
```

Expected output:

- `Cosine similarity > 0.9999` → excellent
- `Max abs diff < 1e-2` → acceptable
- `Cosine similarity < 0.99` → investigate

---

## `diagnose_channel.py`

Per-channel error diagnosis. Useful when a single channel of the
output diverges from PyTorch.

```bash
python diagnose_channel.py model.engine yolov8n.pt image.jpg
```

Prints the top-10 largest per-element differences with their
channel and anchor position.

---

## `bench_ops.py`

Micro-benchmark for individual ops (`matmul`, `relu`, `sigmoid`,
`softmax`, `reduce_sum`).

```bash
python bench_ops.py
```

No external dependencies beyond `pdefa` itself.