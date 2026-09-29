# pdefa

**CPU inference engine** — with built-in PNG/JPEG loader, ONNX→engine converter, and YOLO post-processing. **Zero Python dependencies.**

Written in C++20 with AVX2 SIMD. The Python layer uses only `ctypes` — no external packages required.

---

## Installation

```bash
pip install pdefa
```

**Requirements:**
- Python 3.8+
- Windows x64 / Linux x64 / macOS (x86_64, arm64)

No external Python dependencies.

---

## Getting a `.engine` model

pdefa runs `.engine` files. Two ways to get one:

**Option A — Convert your own ONNX model:**

```bash
pip install onnx numpy
python pdefa/tools/onnx_to_engine.py your_model.onnx your_model.engine
```

> **Note:** The pdefa **runtime** is dependency-free. The **converter** is a
> build-time tool and requires `onnx` + `numpy`.

**Option B — Use a pre-converted model** (see the `examples/` folder).

---

## Quick Start in 60 Seconds

### YOLO detection (one-liner)

```python
import pdefa

model = pdefa.load("yolov8n.engine")
boxes = model("bus.jpg", conf_thres=0.25)

for b in boxes:
    print(f"cls={b['cls']}  conf={b['conf']:.3f}  "
          f"box=({b['x1']:.0f},{b['y1']:.0f},{b['x2']:.0f},{b['y2']:.0f})")
```

### Manual pipeline (full control)

```python
import pdefa

ctx     = pdefa.Context()
model   = pdefa.Model(ctx, "yolov8n.engine")
session = pdefa.Session(ctx, model)

tensor, meta = pdefa.prepare_for_model(model, "bus.jpg")

session.set_input(0, tensor)
session.run()

out = session.output(0)

boxes = pdefa.decode_yolo(out, meta, conf_thres=0.25)
```

> **Note:** `decode_yolo` returns `list[dict]`. Access fields with brackets:
> `d['cls']`, `d['conf']`, `d['x1']`, `d['y1']`, `d['x2']`, `d['y2']`.

---

## Examples

Runnable examples in [`examples/`](examples/):

```bash
python examples/yolo_inference.py yolov8n.engine bus.jpg
```

The `examples/yolo_inference.py` script prints detections with class names,
confidence scores, and original-image coordinates.

---

## API Reference

### Meta

```python
pdefa.version()       # -> "1.0.3"
pdefa.simd_level()    # -> "AVX2"
pdefa.cpu_brand()     # -> "AMD/Intel ..."
```

### Convenience API

```python
model = pdefa.load("yolov8n.engine")
model.set_num_threads(8)
boxes = model("bus.jpg", conf_thres=0.25, iou_thres=0.45)
```

### Context

CPU / thread configuration. Create **once**, share across all models.

```python
ctx = pdefa.Context()
ctx.set_num_threads(8)
ctx.get_num_threads()          # -> 8
```

### Model

Loads a `.engine` file.

```python
model = pdefa.Model(ctx, "yolov8n.engine")

model.name                        # property, returns path to .engine file

info = model.input_info(0)
print(info.name, info.shape)      # "input_0" (1, 3, 640, 640)

info = model.output_info(0)       # "output_0" (1, 84, 8400)
```

> **Note:** `model.name` is a **property** (no parentheses).
> `model.input_info(0)` is a **method** (parentheses required).

### Session

An **inference session** tied to a Model + Context.

```python
session = pdefa.Session(ctx, model)

session.set_input(0, tensor)
session.run()

out = session.output(0)                # Tensor (non-owning)
arr = out.tolist()                     # nested Python list

s = session.stats()
print(f"last: {s['last_run_ms']:.3f} ms, {s['runs']} runs")

boxes = session.run_and_decode(meta, conf_thres=0.25)

t = session.debug_intermediate(5)
print(t.shape)
```

### Tensor

Works with plain Python lists.

```python
x = pdefa.randn(3, 4, seed=42)
y = pdefa.rand(3, 4, seed=0)
z = pdefa.zeros(3, 4)
o = pdefa.ones(3, 4)
e = pdefa.empty(3, 4)
w = pdefa.from_list([[1.0, 2.0], [3.0, 4.0]])

a = x + y
b = x - y
c = x * 2.0
d = x / y
m = A @ B

arr = x.tolist()
x.fill(0.5)
x.copy_from_list([[1, 2, 3], [4, 5, 6]])

x.shape      # (3, 4)
x.numel      # 12
x.ndim       # 2
```

### Ops (direct math)

```python
c = pdefa.ops_add(a, b)
c = pdefa.ops_sub(a, b)
c = pdefa.ops_mul(a, b)
c = pdefa.ops_div(a, b)

pdefa.ops_relu(t)
pdefa.ops_silu(t)
pdefa.ops_sigmoid(t)
pdefa.ops_tanh(t)
pdefa.ops_gelu(t)
pdefa.ops_clip(t, -1.0, 1.0)
pdefa.ops_softmax(t, axis=-1)

pdefa.ops_reduce_max(a, axis=1, keepdims=False)
pdefa.ops_reduce_sum(a, axis=0, keepdims=True)
pdefa.ops_reduce_mean(a, axis=-1, keepdims=False)

pdefa.ops_argmax(a, axis=1, keepdims=False)
pdefa.ops_gather(a, indices, axis=0)
pdefa.ops_reshape(a, [3, 2])
pdefa.ops_transpose(a, [1, 0])
pdefa.ops_concat([a, b], axis=0)
pdefa.ops_stack([a, b], axis=0)

out = pdefa.tensor_empty([512, 512])
pdefa.ops_matmul(A, B, out)
```

### Image / Preprocess

```python
img = pdefa.load_image("photo.jpg")
img.width, img.height, img.channels     # 1920, 1080, 3
img.format                               # 1 (RGB8) | 2 (RGBA8) | 0 (GRAY8)

arr = img.tolist()
raw = img.tobytes()

tensor, meta = pdefa.preprocess_letterbox(
    img,
    target_w=640, target_h=640,
    scale=1.0/255.0,
    pad_value=0.0,
)

tensor, meta = pdefa.prepare_for_engine(
    "photo.jpg", 640, 640,
    scale=1.0/255.0, pad_value=0.0,
)

tensor, meta = pdefa.prepare_for_model(model, "photo.jpg")
```

`meta` contents:

```python
meta.orig_w    # 1920
meta.orig_h    # 1080
meta.pad_x
meta.pad_y
meta.scaled_w
meta.scaled_h
meta.ratio     # 0.333
```

### Post-process

```python
boxes = pdefa.decode_yolo(
    out, meta,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=300,
)
# [{'x1': ..., 'y1': ..., 'x2': ..., 'y2': ..., 'conf': ..., 'cls': ...}, ...]

orig = pdefa.map_box_to_original(pdefa.Box(100, 200, 300, 400), meta)

x, y = pdefa.map_point_to_original(320.0, 320.0, meta)

boxes = session.run_and_decode(meta, conf_thres=0.25)
```

### Batch Inference

```python
ctx   = pdefa.Context()
model = pdefa.Model(ctx, "yolov8n.engine")

with pdefa.BatchInference(ctx, model, batch_size=4) as batch:
    results = batch.run(["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
    for out_tensor, meta in results:
        arr = out_tensor.tolist()
        print(len(arr[0]))
```

---

## Examples

### 1. Simplest detection

```python
import pdefa

model = pdefa.load("yolov8n.engine")
boxes = model("bus.jpg", conf_thres=0.3)

for b in boxes:
    print(f"{b['cls']}  {b['conf']:.2f}")
```

### 2. Batch processing

```python
import pdefa

ctx   = pdefa.Context()
model = pdefa.Model(ctx, "yolov8n.engine")

with pdefa.BatchInference(ctx, model, batch_size=4) as batch:
    results = batch.run(["a.jpg", "b.jpg", "c.jpg"])
    for out_tensor, meta in results:
        print(out_tensor.shape)
```

### 3. Classification (ImageNet)

```python
import pdefa

ctx     = pdefa.Context()
model   = pdefa.Model(ctx, "resnet18.engine")
session = pdefa.Session(ctx, model)

from PIL import Image
img = Image.open("cat.jpg").convert("RGB").resize((224, 224))
arr = [[[[0.0]*224 for _ in range(224)] for _ in range(3)]]
px = img.load()
for y in range(224):
    for x in range(224):
        r, g, b = px[x, y]
        arr[0][0][y][x] = (r / 255.0 - 0.485) / 0.229
        arr[0][1][y][x] = (g / 255.0 - 0.456) / 0.224
        arr[0][2][y][x] = (b / 255.0 - 0.406) / 0.225

t = pdefa.from_list(arr)
session.set_input(0, t)
session.run()

out = session.output(0).tolist()
flat = out[0]
top5 = sorted(range(len(flat)), key=lambda i: flat[i], reverse=True)[:5]
for i, idx in enumerate(top5, 1):
    print(f"{i}. class={idx}  prob={flat[idx]:.4f}")
```

### 4. Segmentation

```python
ctx     = pdefa.Context()
model   = pdefa.Model(ctx, "unet.engine")
session = pdefa.Session(ctx, model)

tensor, meta = pdefa.prepare_for_model(model, "image.jpg")
session.set_input(0, tensor)
session.run()

mask = session.output(0).tolist()
```

### 5. Custom model

```python
ctx     = pdefa.Context()
model   = pdefa.Model(ctx, "my_model.engine")
session = pdefa.Session(ctx, model)

input_list = [[[[0.0]*640 for _ in range(640)] for _ in range(3)]]
t = pdefa.from_list(input_list)

session.set_input(0, t)
session.run()

raw = session.output(0).tolist()
```

---

## Supported Formats

### Images

| Format | Support | Details |
|---|---|---|
| **PNG** | yes | 8/16-bit, gray / RGB / RGBA / palette, Adam7 interlace |
| **JPEG** | yes | Baseline 8-bit, gray / YCbCr (4:4:4 / 4:2:2 / 4:2:0), restart markers |

### Model Ops

| Category | Ops |
|---|---|
| **Conv** | Conv2d (fused activation: ReLU/SiLU/Leaky/Sigmoid/Tanh/GELU) |
| **Linear** | MatMul / Gemm |
| **Elementwise** | Add / Mul / Sub / Div (broadcast) |
| **Activation** | ReLU / LeakyReLU / Sigmoid / Tanh / SiLU / GELU / HardSwish / HardSigmoid / Clip |
| **Pooling** | MaxPool / AvgPool / GlobalAveragePool |
| **Norm** | BatchNorm2d / LayerNorm |
| **Attention** | Softmax / MatMul (4D batch) |
| **Shape** | Reshape / Flatten / Transpose / Concat / Slice / Gather / Squeeze / Unsqueeze / Expand / Tile / Stack |
| **Reduce** | ReduceMax / ReduceSum / ReduceMean |
| **Index** | ArgMax / TopK / GatherElements |
| **Resize** | Upsample / Resize (nearest + bilinear) |

**Model conversion:** use `pdefa/tools/onnx_to_engine.py` to convert ONNX -> `.engine`:

```bash
pip install onnx numpy
python pdefa/tools/onnx_to_engine.py model.onnx model.engine
```

> **Note:** The pdefa **runtime** is dependency-free. The **converter**
> is a build-time tool and requires `onnx` + `numpy`.

---

## Limitations

| Limitation | Notes |
|---|---|
| **Model format** | Only `.engine` (our format). ONNX/PyTorch require the converter. |
| **INT8 quantization** | `ConvInteger`, `DynamicQuantizeLinear` not supported. |
| **AVX-512** | Not supported. A prototype exists in source (`matmul_avx512.cpp`, ~80 LOC) but is **not integrated, not tested, and not shipped**. Engine uses AVX2 only. |
| **macOS** | Not build-tested (should work in theory). |
| **Swin / MaxViT** | Some window attention ops still missing. |
| **ViT export** | `torchvision` -> ONNX export fails (upstream issue). |

---

## Known Issues

### Runtime failures (planned for next release)

| Model | Node | Op | Cause |
|---|---|---|---|
| yolov12l | `/model.6/...` | Mul | Rank-5 broadcast edge case |
| yolov12x | `/model.6/...` | Mul | Same as above |
| convnext_t | `/features/...` | MatMul | 4D x 2D linear path regression |
| convnext_s | `/features/...` | MatMul | Same as above |
| convnext_b | `/features/...` | MatMul | Same as above |
| swin_t | `/features/...` | - | Window attention `Roll` op missing |
| swin_s | `/features/...` | - | Same as above |
| swin_b | `/features/...` | - | Same as above |
| swin_v2_t | `/features/...` | MatMul | Same as above |
| maxvit_t | `/stem/...` | Mul | Missing input (converter edge case) |

### Slow paths (planned optimization)

| Model | Speed vs PyTorch | Cause |
|---|---|---|
| yolov10n | 3.23x | `TopK` / `GatherElements` scalar path |
| yolov10s | 2.39x | Same as above |
| yolov10m/b/l/x | 1.4-1.8x | Same as above |
| regnet_y_400mf | 1.97x | Grouped conv memcpy overhead |
| regnet_y_800mf | 1.56x | Same as above |
| regnet_y_1_6gf | 1.27x | Same as above |

### Export failures (upstream, not pdefa)

| Model | Error | Notes |
|---|---|---|
| vit_b_16 | `UnsupportedOperatorError` | `torchvision` -> ONNX export issue |
| vit_b_32 | Same | Same |
| vit_l_16 | Same | Same |

These models fail at the **PyTorch -> ONNX** stage, before pdefa is involved.

---

## Changelog

### v1.0.3 - 2026-09-24

**NumPy-independent release**

This release removes **all external Python dependencies**. Prior versions required NumPy for `decode_yolo`, `from_numpy`, and tensor data access. **v1.0.3 runs without installing any additional packages** - `pip install pdefa` is now fully self-contained.

**Breaking change:** `pyproject.toml` - `dependencies = []` (was `["numpy>=1.20"]`).

**New features**
- `pdefa.load("model.engine")("image.jpg")` - one-liner YOLO API
- `pdefa.EngineModel` class with `.set_num_threads()`
- `Context.set_num_threads(n)` / `.get_num_threads()` exposed to Python
- 20+ ops exposed via C API:
  - Binary: `ops_add`, `ops_sub`, `ops_mul`, `ops_div` (broadcast supported)
  - Unary: `ops_sigmoid`, `ops_tanh`, `ops_gelu`, `ops_clip`, `ops_softmax`
  - Reduce: `ops_reduce_max`, `ops_reduce_sum`, `ops_reduce_mean`
  - Index: `ops_argmax`, `ops_gather`
  - Shape: `ops_reshape`, `ops_transpose`, `ops_concat`, `ops_stack`
  - Random: `random_randn`, `random_rand`
- Tensor creation without NumPy: `pdefa.randn`, `pdefa.rand`, `pdefa.zeros`, `pdefa.ones`, `pdefa.empty`, `pdefa.from_list`
- `Tensor.tolist()` - nested Python list access
- `Tensor` operator overloading: `+`, `-`, `*`, `/`, `@`, unary `-`
- `Tensor.fill()`, `Tensor.copy_from_list()` - in-place operations
- `Image.tolist()` / `Image.tobytes()` - pixel access
- `decode_yolo()` rewritten without NumPy (pure Python)
- `_nms()` rewritten without NumPy (pure Python, class-based)
- `engine_ops_argmax`, `engine_ops_stack` - new graph ops
- `engine_random_randn`, `engine_random_rand` - random tensor generation
- **Zero external Python dependencies** - `dependencies = []` in `pyproject.toml`

**Fixes**
- `Slice` NHWC axis remap (axis=2, 3)
- `Reshape` 4D to 5D channel shuffle support
- `Transpose` 5D NHWC remap
- Channel-broadcast `Mul`/`Add` fast path (SE blocks)
- Depthwise conv parallelization + border peeling
- `Clip` op support (MobileNetV2 fix)
- `ReduceSum` keepdims heuristic for rank >= 5

**Models supported**
- YOLOv8 (n/s/m/l/x)
- YOLOv9 (t/s/m/c/e)
- YOLO10 (n/s/m/b/l/x) - works but slow (TopK scalar path)
- YOLO11 (n/s/m/l/x)
- YOLO12 (n/s/m)
- ResNet (18/34/50)
- MobileNet (v2, v3_s, v3_l)
- VGG (16/19)
- DenseNet (121/169)
- SqueezeNet (1_0, 1_1)
- EfficientNet (b0, b1)
- ShuffleNetV2
- RegNet (400mf, 800mf, 1.6gf)
- FCN (resnet50, resnet101)
- DeepLabV3 (resnet50, resnet101)
- LR-ASPP (MobileNetV3)

**Performance (vs PyTorch, AVX2)**
- Classification: 1.2x - 2.2x faster
- YOLO detection: 0.8x - 1.3x (comparable)

**Migration notes (from v1.0.1 to v1.0.3)**
- `pdefa.from_numpy(arr)` -> use `pdefa.from_list(arr.tolist())`
- `tensor.numpy()` -> use `tensor.tolist()`
- `image.numpy()` -> use `image.tolist()` or `image.tobytes()`
- Test scripts using `np.random.randn(...)` -> use `pdefa.randn(..., seed=0)`

---

### v1.0.1 - 2026-09-21

- Session profiling callback
- `debug_intermediate()` for layer inspection
- Input/output tensor info API
- `engine_run_simple()` convenience function
- Custom `.engine` v2 format with input/output metadata
- Improved conv dispatcher (Winograd, Direct NHWC, Gemm fallbacks)
- Zero-copy 1x1 convolution path

---

### v1.0.0.1 - 2026-09-17
- PNG + JPEG loader (baseline, grayscale, RGB, RGBA, Adam7)

### v1.0.0 - 2026-09-15

- Initial release
- C++20 engine with AVX2 SIMD kernels
- ONNX -> `.engine` converter (`pdefa/tools/onnx_to_engine.py`)
- YOLOv8 detection pipeline
- Python bindings via ctypes
- CMake build system
- Cross-platform support (Windows / Linux / macOS)

---

## Roadmap

### v1.4 (next)
- **Vectorized `TopK` + `GatherElements`** - YOLOv10 speed fix (~3x faster)
### v1.5
- **ConvNeXt MatMul 4D x 2D** fix
### v1.5.5
- **Swin window attention** `Roll` op
### v1.5.6
- **MaxViT** converter fix
### v1.8
- **Grouped convolution** optimization (RegNet, MobileNet variants)
### v1.8.1
- **Bilinear Resize** for segmentation models
### v1.9
- **INT8 quantization** support
### v2.0
- **AVX-512 kernels** (prototype exists)
### v2.1
- **CUDA backend** (optional)
### v2.2
- **Multi-input / multi-output** models
### v2.2.1
- **ONNX Runtime API compatibility layer**

---

## Model Support Matrix

Legend: yes = works, slow = works but slow, no = not supported

| Family | Models | Status |
|---|---|---|
| **YOLOv8** | n / s / m / l / x | yes all |
| **YOLOv9** | t / s / m / c / e | yes all |
| **YOLOv10** | n / s / m / b / l / x | slow all |
| **YOLO11** | n / s / m / l / x | yes all |
| **YOLO12** | n / s / m | yes |
| **YOLO12** | l / x | no - Mul fail |
| **ResNet** | 18 / 34 / 50 | yes all |
| **MobileNet** | v2 / v3_s / v3_l | yes all |
| **VGG** | 16 / 19 | yes all |
| **DenseNet** | 121 / 169 | yes all |
| **SqueezeNet** | 1_0 / 1_1 | yes all |
| **EfficientNet** | b0 / b1 | yes all |
| **ShuffleNet** | v2_x1_0 | yes |
| **ConvNeXt** | t / s / b | no - MatMul fail |
| **Swin** | t / s / b / v2_t | no - Roll op missing |
| **ViT** | b_16 / b_32 / l_16 | no - export fail (upstream) |
| **MaxViT** | t | no - Mul missing input |
| **RegNet** | 400mf / 800mf / 1.6gf | slow all |
| **FCN** | resnet50 / resnet101 | yes all |
| **DeepLabV3** | resnet50 / resnet101 | yes all |
| **LR-ASPP** | mobilenet_v3_large | yes |

---

## Performance

> **CPU Backend Note:** pdefa ships an **AVX2-only** engine. **No AVX-512 kernels are active** in this release. Benchmarks below are AVX2 on an AMD Zen 3+ CPU. AVX-512 support is planned for a future release.

Tested on AMD Ryzen 5 7535HS (12 threads, AVX2). All times median of 20 runs, 3 warmups.

### YOLO - Detection

| Model    | PyTorch | ONNX RT | pdefa   | Speed vs PyTorch |
|----------|---------|---------|---------|------------------|
| yolov8n  | 47 ms   | 24 ms   | 37 ms   | 1.27x faster |
| yolov8s  | 86 ms   | 59 ms   | 91 ms   | 1.06x slower |
| yolov8m  | 187 ms  | 155 ms  | 229 ms  | 1.23x slower |
| yolov9t  | 76 ms   | 32 ms   | 45 ms   | 1.69x faster |
| yolov9c  | 310 ms  | 230 ms  | 339 ms  | 1.09x slower |
| yolov9e  | 746 ms  | 503 ms  | 660 ms  | 1.13x faster |
| yolo11n  | 44 ms   | 25 ms   | 55 ms   | 1.25x slower |
| yolo11m  | 207 ms  | 170 ms  | 248 ms  | 1.20x slower |
| yolo12n  | 60 ms   | 26 ms   | 46 ms   | 1.30x faster |
| yolo12s  | 111 ms  | 62 ms   | 100 ms  | 1.11x faster |

### Classification - Classic

| Model            | PyTorch | ONNX RT | pdefa   | Speed vs PyTorch |
|------------------|---------|---------|---------|------------------|
| resnet18         | 17 ms   | 7 ms    | 14 ms   | 1.23x faster |
| resnet50         | 40 ms   | 15 ms   | 31 ms   | 1.27x faster |
| mobilenet_v2     | 13 ms   | 5 ms    | 8.5 ms  | 1.56x faster |
| mobilenet_v3_s   | 8 ms    | 2 ms    | 4.1 ms  | 1.88x faster |
| densenet121      | 44 ms   | 20 ms   | 32 ms   | 1.39x faster |
| densenet169      | 60 ms   | 25 ms   | 40 ms   | 1.49x faster |
| squeezenet1_1    | 9 ms    | 3 ms    | 4.0 ms  | 2.22x faster |
| efficientnet_b0  | 18 ms   | 7 ms    | 9.4 ms  | 1.96x faster |
| shufflenet_v2    | 13 ms   | 4 ms    | 7.1 ms  | 1.85x faster |

### Numerical Accuracy

**YOLOv8n vs PyTorch:**
- Cosine similarity: 0.99998
- Max absolute diff: < 1e-4

**EfficientNet-B0 vs PyTorch:**
- Cosine similarity: 1.000000
- Max absolute diff: 3.2e-06

---

## Development

Source: https://github.com/hasnard/pdefa

The engine is C++20, built with CMake. The Python package uses `setuptools`.

### Quick install (prebuilt DLLs are in the repo)

```bash
git clone https://github.com/hasnard/pdefa
cd pdefa
pip install -e ".[test]"

cd pdefa
python -m pytest tests -v
```

### Rebuilding the engine from source (Windows)

```powershell
git clone https://github.com/hasnard/pdefa
cd pdefa

cd PDEFA1.0
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release `
      -DENGINE_BUILD_TESTS=OFF `
      -DENGINE_BUILD_BENCHMARKS=OFF `
      -DENGINE_BUILD_EXAMPLES=OFF `
      -DENGINE_BUILD_TOOLS=OFF
cmake --build build --config Release --parallel
cd ..

Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

Copy-Item build\bin\pdefa_native.dll pdefa\python\src\pdefa\lib\ -Force
Copy-Item build\bin\engine.dll       pdefa\python\src\pdefa\lib\ -Force

pip install -e ".[test]"

cd pdefa
python -m pytest tests -v
```

Expected: `86 passed`

### Building a wheel

```powershell
cd pdefa
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
python -m build
```

Publish to TestPyPI (test) or PyPI (production):

```powershell
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

Username: `__token__` - Password: your API token (`pypi-...`).

---

## License

pdefa is released under a **dual license**.

### 1. AGPL-3.0-or-later (open source)

Free to use under the terms of the GNU Affero General Public License v3.0
or later. Under this license you must:

- Release the complete source code of any derivative work under AGPL-3.0.
- **Disclose source code even when the software is only offered over a
  network** (SaaS, API, web service) - this is the key difference from GPLv3.
- Not embed pdefa in a closed-source product and distribute it.

Full text: [LICENSE](LICENSE)

### 2. Commercial License

For companies that cannot or do not want to comply with AGPL-3.0.

Contact: **info@pdfduty.com**
Details: [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)

### Contributing

By submitting a pull request, patch, or any code contribution, you agree
to the terms of the [CLA](CLA.md).

### Third-party notices

See [NOTICE](NOTICE) for bundled third-party components.