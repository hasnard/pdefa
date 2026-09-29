# Test Data

Some tests require small test files. If missing, tests auto-skip.

## Required files

| File | Size | Description |
|---|---|---|
| `data/cat.jpg` | < 20 KB | 50×50 JPEG, RGB |
| `data/bus.png` | < 20 KB | 50×50 PNG, RGB |
| `data/tiny.engine` | < 200 KB | Model with 64×64 input |

## Generating `tiny.engine`

```bash
python ../tools/converter/onnx_to_engine.py tiny.onnx data/tiny.engine
```

Minimal `tiny.onnx`:

```python
import torch, torch.nn as nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(8, 4)
    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1))

torch.onnx.export(
    Tiny().eval(),
    torch.zeros(1, 3, 64, 64),
    "tiny.onnx",
    opset_version=17,
)
```

## Generating test images

```bash
convert -size 50x50 xc:red data/cat.jpg
convert -size 50x50 xc:blue data/bus.png
```

Or with Pillow:

```python
from PIL import Image
Image.new("RGB", (50, 50), (200, 30, 30)).save("data/cat.jpg")
Image.new("RGB", (50, 50), (30, 30, 200)).save("data/bus.png")
```