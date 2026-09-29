#include "png_internal.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>

namespace imageloader {
namespace png {

static inline uint8_t paeth_fast(int a, int b, int c) {
    int p  = a + b - c;
    int pa = std::abs(p - a);
    int pb = std::abs(p - b);
    int pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return static_cast<uint8_t>(a);
    if (pb <= pc)             return static_cast<uint8_t>(b);
    return static_cast<uint8_t>(c);
}

void unfilter(std::vector<uint8_t>& raw,
              uint32_t width, uint32_t height,
              size_t bpp, size_t row_bytes)
{
    (void)width;
    const size_t src_stride = row_bytes + 1; // Her satır başında 1 filtre baytı
    uint8_t* base = raw.data();

    // SIFIR KOPYA: 'raw' vektörü üzerinde yerinde (in-place) çözülür.
    // Her satırda dst_index < src_index olduğundan üzerine yazma çakışması yaşanmaz.
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* row_src = base + (size_t)y * src_stride;
        uint8_t filter = row_src[0];
        const uint8_t* src = row_src + 1;
        uint8_t* dst = base + (size_t)y * row_bytes;
        const uint8_t* prev = (y == 0) ? nullptr : (base + (size_t)(y - 1) * row_bytes);

        switch (filter) {
            case 0: { // None
                std::memmove(dst, src, row_bytes);
                break;
            }
            case 1: { // Sub
                for (size_t x = 0; x < bpp; ++x) {
                    dst[x] = src[x];
                }
                for (size_t x = bpp; x < row_bytes; ++x) {
                    dst[x] = static_cast<uint8_t>(src[x] + dst[x - bpp]);
                }
                break;
            }
            case 2: { // Up (AVX2 Vektörize)
                if (!prev) {
                    std::memmove(dst, src, row_bytes);
                } else {
                    size_t x = 0;
#if defined(__AVX2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
                    for (; x + 32 <= row_bytes; x += 32) {
                        __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + x));
                        __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prev + x));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + x), _mm256_add_epi8(s, p));
                    }
#endif
                    for (; x < row_bytes; ++x) {
                        dst[x] = static_cast<uint8_t>(src[x] + prev[x]);
                    }
                }
                break;
            }
            case 3: { // Average
                if (!prev) {
                    for (size_t x = 0; x < bpp; ++x) {
                        dst[x] = src[x];
                    }
                    for (size_t x = bpp; x < row_bytes; ++x) {
                        dst[x] = static_cast<uint8_t>(src[x] + (dst[x - bpp] >> 1));
                    }
                } else {
                    for (size_t x = 0; x < bpp; ++x) {
                        dst[x] = static_cast<uint8_t>(src[x] + (prev[x] >> 1));
                    }
                    for (size_t x = bpp; x < row_bytes; ++x) {
                        dst[x] = static_cast<uint8_t>(src[x] + ((dst[x - bpp] + prev[x]) >> 1));
                    }
                }
                break;
            }
            case 4: { // Paeth
                if (!prev) {
                    for (size_t x = 0; x < bpp; ++x) {
                        dst[x] = src[x];
                    }
                    for (size_t x = bpp; x < row_bytes; ++x) {
                        dst[x] = static_cast<uint8_t>(src[x] + dst[x - bpp]);
                    }
                } else {
                    for (size_t x = 0; x < bpp; ++x) {
                        dst[x] = static_cast<uint8_t>(src[x] + prev[x]);
                    }
                    for (size_t x = bpp; x < row_bytes; ++x) {
                        int a = dst[x - bpp];
                        int b = prev[x];
                        int c = prev[x - bpp];
                        dst[x] = static_cast<uint8_t>(src[x] + paeth_fast(a, b, c));
                    }
                }
                break;
            }
            default:
                throw std::runtime_error("png: geçersiz filtre");
        }
    }

    // Filtre baytları ayıklandığı için boyutu tam görüntü boyutuna çek
    raw.resize((size_t)height * row_bytes);
}

} // namespace png
} // namespace imageloader