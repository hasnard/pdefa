# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.3] — 2026-09-24

### Added
- `pdefa.load("model.engine")("image.jpg")` — one-liner YOLO API
- `EngineModel` class with `.set_num_threads()`
- `Context.set_num_threads(n)` / `.get_num_threads()`
- 20+ ops exposed via C API
- `random_randn` / `random_rand`
- Tensor creation without NumPy: `randn`, `rand`, `zeros`, `ones`,
  `empty`, `from_list`
- `Tensor.tolist()` — nested Python list access
- `Tensor` operator overloading: `+`, `-`, `*`, `/`, `@`, unary `-`
- `Tensor.fill()`, `Tensor.copy_from_list()`
- `Image.tolist()`, `Image.tobytes()`
- `decode_yolo()`, `_nms()` rewritten without NumPy
- **Zero external Python dependencies** (`dependencies = []`)
- Dual license: AGPL-3.0-or-later + Commercial

### Fixed
- `Slice` NHWC axis remap (axis=2, 3)
- `Reshape` 4D↔5D channel shuffle
- `Transpose` 5D NHWC remap
- Channel-broadcast `Mul`/`Add` fast path (SE blocks)
- Depthwise conv parallelization + border peeling
- `Clip` op support (MobileNetV2 fix)
- `ReduceSum` keepdims heuristic for rank ≥ 5

### Changed
- **License: MIT → AGPL-3.0-or-later + Commercial**
- `pyproject.toml`: `dependencies = []` (was `["numpy>=1.20"]`)

## [1.0.1] — 2026-09-21

### Added
- Session profiling callback
- `debug_intermediate()` for layer inspection
- Input/output tensor info API
- Custom `.engine` v2 format with metadata

## [1.0.0] — 2026-09-15

### Added
- Initial release
- C++20 engine with AVX2 SIMD kernels
- ONNX → `.engine` converter
- YOLOv8 detection pipeline
- Python bindings via `ctypes`
- CMake build system
- Cross-platform support