#include "jpeg_internal.hpp"
#include <cstring>

namespace imageloader {
namespace jpeg {

// ------------------------------------------------------------------
// Dequant: zigzag sirada gelen katsayilari dogal siraya cevir + quant
// ------------------------------------------------------------------
void dequant_zigzag(const int16_t* in, const uint16_t* q, int16_t* out) {
    for (int i = 0; i < 64; ++i) {
        int v = (int)in[i] * (int)q[i];
        out[ZIGZAG[i]] = (int16_t)v;
    }
}

// ------------------------------------------------------------------
// AAN IDCT (nanojpeg tabanli)
// ------------------------------------------------------------------
#define W1 2841
#define W2 2676
#define W3 2408
#define W5 1609
#define W6 1108
#define W7 565

static inline uint8_t clip_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

// ------------------------------------------------------------------
static void row_idct(int* blk) {
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;
    if (!((x1 = blk[4] << 11) | (x2 = blk[6]) | (x3 = blk[2]) |
          (x4 = blk[1]) | (x5 = blk[7]) | (x6 = blk[5]) | (x7 = blk[3])))
    {
        int v = blk[0] << 3;
        blk[0]=blk[1]=blk[2]=blk[3]=blk[4]=blk[5]=blk[6]=blk[7]=v;
        return;
    }
    x0 = (blk[0] << 11) + 128;
    x8 = W7 * (x4 + x5);
    x4 = x8 + (W1 - W7) * x4;
    x5 = x8 - (W1 + W7) * x5;
    x8 = W3 * (x6 + x7);
    x6 = x8 - (W3 - W5) * x6;
    x7 = x8 - (W3 + W5) * x7;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2);
    x2 = x1 - (W2 + W6) * x2;
    x3 = x1 + (W2 - W6) * x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    blk[0] = (x7 + x1) >> 8;
    blk[1] = (x3 + x2) >> 8;
    blk[2] = (x0 + x4) >> 8;
    blk[3] = (x8 + x6) >> 8;
    blk[4] = (x8 - x6) >> 8;
    blk[5] = (x0 - x4) >> 8;
    blk[6] = (x3 - x2) >> 8;
    blk[7] = (x7 - x1) >> 8;
}

// ------------------------------------------------------------------
static void col_idct(const int* blk, uint8_t* out, int stride) {
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;
    if (!((x1 = blk[8*4] << 8) | (x2 = blk[8*6]) | (x3 = blk[8*2]) |
          (x4 = blk[8*1]) | (x5 = blk[8*7]) | (x6 = blk[8*5]) | (x7 = blk[8*3])))
    {
        int v = clip_u8(((blk[0] + 32) >> 6) + 128);
        for (int i = 0; i < 8; ++i) { out[i * stride] = (uint8_t)v; }
        return;
    }
    x0 = (blk[0] << 8) + 8192;
    x8 = W7 * (x4 + x5) + 4;
    x4 = (x8 + (W1 - W7) * x4) >> 3;
    x5 = (x8 - (W1 + W7) * x5) >> 3;
    x8 = W3 * (x6 + x7) + 4;
    x6 = (x8 - (W3 - W5) * x6) >> 3;
    x7 = (x8 - (W3 + W5) * x7) >> 3;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2) + 4;
    x2 = (x1 - (W2 + W6) * x2) >> 3;
    x3 = (x1 + (W2 - W6) * x3) >> 3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    out[0 * stride] = clip_u8(((x7 + x1) >> 14) + 128);
    out[1 * stride] = clip_u8(((x3 + x2) >> 14) + 128);
    out[2 * stride] = clip_u8(((x0 + x4) >> 14) + 128);
    out[3 * stride] = clip_u8(((x8 + x6) >> 14) + 128);
    out[4 * stride] = clip_u8(((x8 - x6) >> 14) + 128);
    out[5 * stride] = clip_u8(((x0 - x4) >> 14) + 128);
    out[6 * stride] = clip_u8(((x3 - x2) >> 14) + 128);
    out[7 * stride] = clip_u8(((x7 - x1) >> 14) + 128);
}

// ------------------------------------------------------------------
void idct_8x8(const int16_t* in, uint8_t* out, int out_stride) {
    int blk[64];
    for (int i = 0; i < 64; ++i) blk[i] = in[i];

    for (int i = 0; i < 8; ++i) row_idct(blk + i * 8);
    for (int i = 0; i < 8; ++i) col_idct(blk + i, out + i, out_stride);
}

} // namespace jpeg
} // namespace imageloader