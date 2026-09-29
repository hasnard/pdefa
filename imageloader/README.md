# imageloader

A lightweight, high-performance C++ image loading and preprocessing library designed for inference engines. It provides self-contained decoders for JPEG and PNG, along with optimized letterbox preprocessing and helper APIs for preparing model inputs and mapping results back to original image coordinates.

## Features

- **Zero external dependencies** – pure C++ implementation of PNG and JPEG decoders.
- **JPEG decoder**
  - Baseline DCT (progressive not supported).
  - Grayscale and YCbCr color spaces.
  - Restart markers (RSTn).
  - Optimized Huffman decoding and AAN IDCT.
  - DC‑only shortcut for blocks without AC coefficients.
- **PNG decoder**
  - Supports all standard color types: grayscale, RGB, palette, grayscale+alpha, RGBA.
  - Bit depths: 1, 2, 4, 8, 16 (16‑bit is downconverted to 8‑bit except grayscale without transparency, which is kept as `GRAY16`).
  - Interlacing (Adam7).
  - Transparency (`tRNS`) for grayscale, RGB and palette.
  - Fast zlib inflate with 64‑bit bit reader and two‑level Huffman LUT.
  - AVX2‑accelerated PNG unfilter (Up filter) when available.
- **Preprocessing**
  - Letterbox resize with bilinear interpolation.
  - Padding with configurable value.
  - Normalization via `scale` factor.
  - Zero‑allocation core for direct tensor filling (`preprocess_letterbox_direct`).
- **Engine API**
  - Prepare input tensors from images or file paths.
  - Batch preparation.
  - Map bounding boxes and points from model coordinates back to original image coordinates.
- **Batch processing** – load and preprocess multiple images.

## File Structure

| File | Description |
|------|-------------|
| `engine_api.cpp` / `engine_api.hpp` | Engine integration helpers (input preparation, coordinate mapping). |
| `loader.cpp` / `loader.hpp` | Image loading dispatcher (detects PNG/JPEG). |
| `preprocess.cpp` / `preprocess.hpp` | Letterbox preprocessing and direct tensor filling. |
| `jpeg_*.cpp` / `jpeg_internal.hpp` | JPEG decoder implementation. |
| `png_*.cpp` / `png_internal.hpp` | PNG decoder implementation. |

## Usage

### Loading an image

```cpp
#include "imageloader/loader.hpp"
using namespace imageloader;

Image img = load_image("path/to/image.jpg");
// img.width, img.height, img.format, img.pixels