#pragma once
#include "imageloader/image.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace imageloader {
namespace png {

struct IHDR {
    uint32_t width       = 0;
    uint32_t height      = 0;
    uint8_t  bit_depth   = 0;
    uint8_t  color_type  = 0;
    uint8_t  compression = 0;
    uint8_t  filter      = 0;
    uint8_t  interlace   = 0;
};

struct PNGInfo {
    IHDR ihdr;
    std::vector<uint8_t> idat;
    std::vector<uint8_t> plte;       // 3*N
    std::vector<uint8_t> trns;       // ham bytes
    bool has_plte = false;
    bool has_trns = false;
};

// png_chunks.cpp
PNGInfo read_chunks(const uint8_t* data, size_t size);

// png_inflate.cpp
std::vector<uint8_t> zlib_inflate(const uint8_t* data, size_t size);

// png_filters.cpp
void unfilter(std::vector<uint8_t>& raw,
              uint32_t pass_w, uint32_t pass_h,
              size_t bpp, size_t row_bytes);

// png_unpack.cpp
size_t png_channels(uint8_t color_type);
size_t png_bpp_bytes(uint8_t color_type, uint8_t bit_depth);
size_t png_row_bytes(uint32_t width, uint8_t color_type, uint8_t bit_depth);
uint16_t png_read_sample(const uint8_t* row, size_t sample_index, int bit_depth);

void png_write_pass_to_canvas(
    const std::vector<uint8_t>& packed,
    uint32_t pass_w, uint32_t pass_h,
    int x_start, int y_start, int x_step, int y_step,
    uint32_t canvas_w, uint32_t canvas_h,
    int bit_depth, int channels,
    std::vector<uint16_t>& canvas);

Image png_canvas_to_image(const PNGInfo& info, const std::vector<uint16_t>& canvas);

} // namespace png
} // namespace imageloader