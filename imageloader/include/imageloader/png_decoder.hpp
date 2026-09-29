#pragma once
#include "imageloader/image.hpp"
#include <cstddef>
#include <cstdint>

namespace imageloader {
namespace png {

// Ham PNG byte'larını çözer. Hata durumunda std::runtime_error fırlatır.
Image decode_png(const uint8_t* data, size_t size);

} // namespace png
} // namespace imageloader