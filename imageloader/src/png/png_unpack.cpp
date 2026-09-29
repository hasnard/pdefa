#include "png_internal.hpp"
#include <cstring>
#include <stdexcept>

namespace imageloader {
namespace png {

// ------------------------------------------------------------------
// Boyut/kanal yardımcıları
// ------------------------------------------------------------------
size_t png_channels(uint8_t color_type) {
    switch (color_type) {
        case 0: return 1;   // Gray
        case 2: return 3;   // RGB
        case 3: return 1;   // Palette (tek index)
        case 4: return 2;   // Gray+Alpha
        case 6: return 4;   // RGBA
    }
    throw std::runtime_error("png: bilinmeyen color_type");
}

size_t png_bpp_bytes(uint8_t color_type, uint8_t bit_depth) {
    // Filtering icin "bytes per pixel"
    size_t bits = (size_t)bit_depth * png_channels(color_type);
    size_t b = (bits + 7) / 8;
    return b < 1 ? 1 : b;
}

size_t png_row_bytes(uint32_t width, uint8_t color_type, uint8_t bit_depth) {
    size_t bits = (size_t)width * bit_depth * png_channels(color_type);
    return (bits + 7) / 8;
}

// ------------------------------------------------------------------
// Bit-packed satirdan tek sample oku (1/2/4/8/16-bit)
// ------------------------------------------------------------------
uint16_t png_read_sample(const uint8_t* row, size_t sample_index, int bit_depth) {
    if (bit_depth == 8)  return row[sample_index];
    if (bit_depth == 16) {
        size_t i = sample_index * 2;
        return (uint16_t)(((uint16_t)row[i] << 8) | row[i + 1]);
    }
    // 1, 2, 4-bit
    int per_byte = 8 / bit_depth;
    size_t byte_i = sample_index / (size_t)per_byte;
    int off = (int)(sample_index % (size_t)per_byte);
    int shift = 8 - bit_depth * (off + 1);
    uint8_t mask = (uint8_t)((1u << bit_depth) - 1);
    return (uint16_t)((row[byte_i] >> shift) & mask);
}

// ------------------------------------------------------------------
// Bir pass'in piksellerini canvas'a yerlestir
// (non-interlaced: x_start=0,y_start=0,x_step=1,y_step=1)
// ------------------------------------------------------------------
void png_write_pass_to_canvas(
    const std::vector<uint8_t>& packed,
    uint32_t pass_w, uint32_t pass_h,
    int x_start, int y_start, int x_step, int y_step,
    uint32_t canvas_w, uint32_t canvas_h,
    int bit_depth, int channels,
    std::vector<uint16_t>& canvas)
{
    size_t row_bytes = ((size_t)pass_w * bit_depth * channels + 7) / 8;

    for (uint32_t py = 0; py < pass_h; ++py) {
        const uint8_t* row = packed.data() + (size_t)py * row_bytes;
        int cy = y_start + (int)py * y_step;
        if (cy < 0 || (uint32_t)cy >= canvas_h) continue;

        for (uint32_t px = 0; px < pass_w; ++px) {
            int cx = x_start + (int)px * x_step;
            if (cx < 0 || (uint32_t)cx >= canvas_w) continue;

            size_t dst_base = ((size_t)cy * canvas_w + cx) * (size_t)channels;
            for (int c = 0; c < channels; ++c) {
                size_t sample_idx = (size_t)px * channels + c;
                canvas[dst_base + c] = png_read_sample(row, sample_idx, bit_depth);
            }
        }
    }
}

// ------------------------------------------------------------------
// Sample -> 8-bit donusumu
// ------------------------------------------------------------------
static uint8_t to_8bit(uint16_t v, int bit_depth) {
    if (bit_depth == 8)  return (uint8_t)v;
    if (bit_depth == 16) return (uint8_t)(v >> 8);
    int max = (1 << bit_depth) - 1;
    return (uint8_t)((v * 255 + max / 2) / max);
}

// ------------------------------------------------------------------
// Canvas -> Image
// ------------------------------------------------------------------
Image png_canvas_to_image(const PNGInfo& info, const std::vector<uint16_t>& canvas) {
    const IHDR& h = info.ihdr;
    const int bd  = h.bit_depth;
    const int ct  = h.color_type;

    Image img;
    img.width  = h.width;
    img.height = h.height;

    size_t total = (size_t)h.width * h.height;

    PixelFormat fmt;
    size_t och;

    if (ct == 0) {
        if (info.has_trns) { fmt = PixelFormat::RGBA8; och = 4; }
        else if (bd == 16) { fmt = PixelFormat::GRAY16; och = 2; }
        else               { fmt = PixelFormat::GRAY8;  och = 1; }
    } else if (ct == 2) {
        if (info.has_trns) { fmt = PixelFormat::RGBA8; och = 4; }
        else               { fmt = PixelFormat::RGB8;  och = 3; }
    } else if (ct == 3) {
        if (info.has_trns) { fmt = PixelFormat::RGBA8; och = 4; }
        else               { fmt = PixelFormat::RGB8;  och = 3; }
    } else if (ct == 4) {
        fmt = PixelFormat::RGBA8; och = 4;
    } else if (ct == 6) {
        fmt = PixelFormat::RGBA8; och = 4;
    } else {
        throw std::runtime_error("png: unpack bilinmeyen color_type");
    }

    img.format = fmt;
    img.pixels.assign(total * och, 0);

    // tRNS: transparent color (ct==0 veya ct==2 icin)
    uint16_t trns_gray = 0, tr_g = 0, tr_r = 0, tr_b = 0;
    if (info.has_trns) {
        if (ct == 0) {
            if (info.trns.size() < 2)
                throw std::runtime_error("png: tRNS(gray) boyu");
            trns_gray = (uint16_t)(((uint16_t)info.trns[0] << 8) | info.trns[1]);
        } else if (ct == 2) {
            if (info.trns.size() < 6)
                throw std::runtime_error("png: tRNS(rgb) boyu");
            tr_r = (uint16_t)(((uint16_t)info.trns[0] << 8) | info.trns[1]);
            tr_g = (uint16_t)(((uint16_t)info.trns[2] << 8) | info.trns[3]);
            tr_b = (uint16_t)(((uint16_t)info.trns[4] << 8) | info.trns[5]);
        }
    }

    if (ct == 0) {
        if (info.has_trns) {
            for (size_t i = 0; i < total; ++i) {
                uint16_t s = canvas[i];
                uint8_t v = to_8bit(s, bd);
                uint8_t a = (s == trns_gray) ? 0 : 255;
                img.pixels[i*4+0] = v;
                img.pixels[i*4+1] = v;
                img.pixels[i*4+2] = v;
                img.pixels[i*4+3] = a;
            }
        } else if (bd == 16) {
            for (size_t i = 0; i < total; ++i) {
                uint16_t s = canvas[i];
                img.pixels[i*2+0] = (uint8_t)(s >> 8);
                img.pixels[i*2+1] = (uint8_t)(s & 0xFF);
            }
        } else {
            for (size_t i = 0; i < total; ++i)
                img.pixels[i] = to_8bit(canvas[i], bd);
        }
    }
    else if (ct == 2) {
        if (info.has_trns) {
            for (size_t i = 0; i < total; ++i) {
                uint16_t r = canvas[i*3+0], g = canvas[i*3+1], b = canvas[i*3+2];
                uint8_t a = (r == tr_r && g == tr_g && b == tr_b) ? 0 : 255;
                img.pixels[i*4+0] = to_8bit(r, bd);
                img.pixels[i*4+1] = to_8bit(g, bd);
                img.pixels[i*4+2] = to_8bit(b, bd);
                img.pixels[i*4+3] = a;
            }
        } else {
            for (size_t i = 0; i < total; ++i) {
                img.pixels[i*3+0] = to_8bit(canvas[i*3+0], bd);
                img.pixels[i*3+1] = to_8bit(canvas[i*3+1], bd);
                img.pixels[i*3+2] = to_8bit(canvas[i*3+2], bd);
            }
        }
    }
    else if (ct == 3) {
        size_t n_pal = info.plte.size() / 3;
        bool alpha = info.has_trns;
        for (size_t i = 0; i < total; ++i) {
            uint16_t idx = canvas[i];
            if (idx >= n_pal)
                throw std::runtime_error("png: palette index tasmasi");
            uint8_t r = info.plte[idx*3+0];
            uint8_t g = info.plte[idx*3+1];
            uint8_t b = info.plte[idx*3+2];
            if (alpha) {
                uint8_t a = (idx < info.trns.size()) ? info.trns[idx] : 255;
                img.pixels[i*4+0] = r;
                img.pixels[i*4+1] = g;
                img.pixels[i*4+2] = b;
                img.pixels[i*4+3] = a;
            } else {
                img.pixels[i*3+0] = r;
                img.pixels[i*3+1] = g;
                img.pixels[i*3+2] = b;
            }
        }
    }
    else if (ct == 4) {
        for (size_t i = 0; i < total; ++i) {
            uint8_t v = to_8bit(canvas[i*2+0], bd);
            uint8_t a = to_8bit(canvas[i*2+1], bd);
            img.pixels[i*4+0] = v;
            img.pixels[i*4+1] = v;
            img.pixels[i*4+2] = v;
            img.pixels[i*4+3] = a;
        }
    }
    else if (ct == 6) {
        for (size_t i = 0; i < total; ++i) {
            img.pixels[i*4+0] = to_8bit(canvas[i*4+0], bd);
            img.pixels[i*4+1] = to_8bit(canvas[i*4+1], bd);
            img.pixels[i*4+2] = to_8bit(canvas[i*4+2], bd);
            img.pixels[i*4+3] = to_8bit(canvas[i*4+3], bd);
        }
    }

    return img;
}

} // namespace png
} // namespace imageloader