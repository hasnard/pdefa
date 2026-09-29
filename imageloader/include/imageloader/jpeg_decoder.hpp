#pragma once
#include "imageloader/image.hpp"
#include <cstddef>
#include <cstdint>

namespace imageloader {
namespace jpeg {

// Baseline sequential JPEG decoder.
// Desteklenen:  8-bit, grayscale, YCbCr (4:4:4 / 4:2:0 / 4:2:2), restart.
// Desteklenmeyen: progressive, arithmetic, 12-bit, CMYK.
Image decode_jpeg(const uint8_t* data, size_t size);

} // namespace jpeg
} // namespace imageloader