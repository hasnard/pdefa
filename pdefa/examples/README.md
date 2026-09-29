# Examples

Runnable examples. Each is self-contained.

## Install

```bash
pip install pdefa
```

## List

| File | Description |
|---|---|
| `01_simple_detection.py` | 3-line YOLO detection |
| `02_full_pipeline.py` | Manual pipeline (preprocess → run → decode) |
| `03_classification.py` | ResNet ImageNet classification |
| `04_segmentation.py` | U-Net segmentation |
| `05_batch_inference.py` | Parallel batch processing |
| `06_custom_model.py` | Run your own ONNX model |

## Preparing a model

Convert ONNX to `.engine`:

```bash
python ../tools/converter/onnx_to_engine.py model.onnx model.engine
```