#include "imageloader/png_decoder.hpp"
#include "png_internal.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace imageloader {
namespace png {

static const uint8_t SIGNATURE[8] = {0x89,'P','N','G',0x0D,0x0A,0x1A,0x0A};

struct Pass {
    int x_start, y_start, x_step, y_step;
};

static const Pass ADAM7[7] = {
    {0, 0, 8, 8}, {4, 0, 8, 8}, {0, 4, 4, 8},
    {2, 0, 4, 4}, {0, 2, 2, 4}, {1, 0, 2, 2}, {0, 1, 1, 2},
};

static inline uint32_t div_ceil(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

Image decode_png(const uint8_t* data, size_t size) {
    if (size < 8 || std::memcmp(data, SIGNATURE, 8) != 0)
        throw std::runtime_error("png: gecersiz imza");

    PNGInfo info = read_chunks(data, size);
    const IHDR& h = info.ihdr;

    std::vector<uint8_t> inflated = zlib_inflate(info.idat.data(), info.idat.size());

    const size_t channels  = png_channels(h.color_type);
    const size_t bpp       = png_bpp_bytes(h.color_type, h.bit_depth);
    const size_t row_bytes = png_row_bytes(h.width, h.color_type, h.bit_depth);

    // =========================================================================
    // FAST PATH: 8-bit Standart PNG (Non-Interlaced RGB/RGBA)
    // SIFIR KOPYA: 'unfilter' satırları yerinde birleştirir.
    // 'inflated' doğrudan 'Image.pixels'e taşınır (0 ms bellek maliyeti).
    // =========================================================================
    if (h.interlace == 0 && h.bit_depth == 8 && (h.color_type == 2 || h.color_type == 6)) {
        const size_t expected = (row_bytes + 1) * h.height;
        if (inflated.size() < expected)
            throw std::runtime_error("png: inflate ciktisi eksik");

        unfilter(inflated, h.width, h.height, bpp, row_bytes);

        Image img;
        img.width  = h.width;
        img.height = h.height;
        img.format = (h.color_type == 2) ? PixelFormat::RGB8 : PixelFormat::RGBA8;
        img.pixels = std::move(inflated); // Bellek kopyalaması YOK, işaretçi takası
        return img;
    }

    // =========================================================================
    // SLOW PATH: Adam7 veya Paletli / 16-bit PNG'ler için genel akış
    // =========================================================================
    std::vector<uint16_t> canvas((size_t)h.width * h.height * channels, 0);

    if (h.interlace == 0) {
        size_t expected = (row_bytes + 1) * h.height;
        if (inflated.size() < expected)
            throw std::runtime_error("png: inflate ciktisi eksik");

        unfilter(inflated, h.width, h.height, bpp, row_bytes);
        png_write_pass_to_canvas(inflated,
            h.width, h.height,
            0, 0, 1, 1,
            h.width, h.height,
            h.bit_depth, (int)channels, canvas);
    } else {
        size_t offset = 0;
        for (int p = 0; p < 7; ++p) {
            const Pass& ps = ADAM7[p];
            if ((uint32_t)ps.x_start >= h.width || (uint32_t)ps.y_start >= h.height)
                continue;

            uint32_t pw = div_ceil(h.width  - (uint32_t)ps.x_start, (uint32_t)ps.x_step);
            uint32_t ph = div_ceil(h.height - (uint32_t)ps.y_start, (uint32_t)ps.y_step);
            if (pw == 0 || ph == 0) continue;

            size_t prb   = png_row_bytes(pw, h.color_type, h.bit_depth);
            size_t psize = (prb + 1) * ph;

            if (offset + psize > inflated.size())
                throw std::runtime_error("png: interlaced inflate eksik");

            std::vector<uint8_t> praw(inflated.begin() + offset,
                                      inflated.begin() + offset + psize);
            unfilter(praw, pw, ph, bpp, prb);

            png_write_pass_to_canvas(praw,
                pw, ph,
                ps.x_start, ps.y_start, ps.x_step, ps.y_step,
                h.width, h.height,
                h.bit_depth, (int)channels, canvas);

            offset += psize;
        }
    }

    return png_canvas_to_image(info, canvas);
}

} // namespace png
} // namespace imageloader