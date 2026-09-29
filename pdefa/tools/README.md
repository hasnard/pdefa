# Tools

Development and conversion utilities.

## Files

| File | Purpose |
|---|---|
| `manual_test.py` | Manual smoke test during development |
| `converter/onnx_to_engine.py` | ONNX → `.engine` converter |

## Usage

```bash
# Manual smoke test
python manual_test.py image.png

# Convert ONNX model
python converter/onnx_to_engine.py model.onnx model.engine
```