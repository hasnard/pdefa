# pdefa tools

Development and model-conversion utilities.

## Files

| File | Purpose |
|---|---|
| `onnx_to_engine.py` | ONNX → `.engine` converter (main) |
| `pytorch_to_onnx.py` | PyTorch → ONNX wrapper (one-liner around `torch.onnx.export`) |
| `manual_test.py` | Manual smoke test during development |

## Convert a model

### Option A — From PyTorch (recommended, one command)

```bash
pip install torch onnx numpy

# 1) PyTorch -> ONNX
python pytorch_to_onnx.py model.pt model.onnx --input 1,3,640,640

# 2) ONNX -> .engine
python onnx_to_engine.py model.onnx model.engine
```

### Option B — From an existing ONNX file

```bash
pip install onnx numpy
python onnx_to_engine.py model.onnx model.engine
```

### Option C — From Ultralytics YOLO

```bash
pip install ultralytics onnx numpy

python -c "from ultralytics import YOLO; \
  YOLO('yolov8n.pt').export(format='onnx', opset=12, imgsz=640, dynamic=False, simplify=False)"

python onnx_to_engine.py yolov8n.onnx yolov8n.engine
```

## Requirements

| Tool | Dependencies |
|---|---|
| `onnx_to_engine.py` | `onnx`, `numpy` |
| `pytorch_to_onnx.py` | `torch` |
| `manual_test.py` | `pdefa` (installed) |

> **Note:** These are **build-time** tools. The pdefa **runtime** has zero dependencies.

## Tested with

- PyTorch 2.5 – 2.11
- Ultralytics 8.1 – 8.4
- ONNX 1.14 – 1.21
- Opset 11, 12, 17