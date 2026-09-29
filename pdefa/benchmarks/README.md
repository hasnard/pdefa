# Benchmarks

Scripts here require heavy dependencies and are **not run in CI**.
Run them manually.

## Dependencies

```bash
pip install -e ".[bench]"
# or:
pip install numpy torch ultralytics
```

## Scripts

| File | Purpose |
|---|---|
| `numerical_vs_pytorch.py` | Compare against PyTorch |
| `diagnose_channel.py` | Per-channel error diagnosis |
| `bench_ops.py` | Ops micro-benchmark |

## Usage

```bash
python numerical_vs_pytorch.py model.engine yolov8n.pt image.jpg
python diagnose_channel.py    model.engine yolov8n.pt image.jpg
python bench_ops.py
```

## Expected output

- `Cosine similarity > 0.9999` → excellent
- `Max abs diff < 1e-2` → acceptable
- `Cosine similarity < 0.99` → investigate