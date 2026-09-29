#include "jpeg_internal.hpp"
#include <stdexcept>

namespace imageloader {
namespace jpeg {

static inline int clamp255(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

void ycbcr_to_rgb(const JPEGData& j, uint8_t* out_rgb) {
    const int W = j.width;
    const int H = j.height;

    if (j.num_components == 1) {
        const Component& Y = j.comps[0];
        for (int y = 0; y < H; ++y) {
            int sy = (int)((int64_t)y * Y.comp_height / H);
            if (sy >= Y.comp_height) sy = Y.comp_height - 1;
            const uint8_t* yrow = Y.samples.data() + (size_t)sy * Y.comp_width;
            uint8_t* outrow = out_rgb + (size_t)y * W * 3;
            for (int x = 0; x < W; ++x) {
                int sx = (int)((int64_t)x * Y.comp_width / W);
                if (sx >= Y.comp_width) sx = Y.comp_width - 1;
                uint8_t v = yrow[sx];
                outrow[3*x+0] = v;
                outrow[3*x+1] = v;
                outrow[3*x+2] = v;
            }
        }
        return;
    }

    if (j.num_components < 3)
        throw std::runtime_error("jpeg: renkli icin 3+ kanal gerekli");

    const Component& Y  = j.comps[0];
    const Component& Cb = j.comps[1];
    const Component& Cr = j.comps[2];

    for (int y = 0; y < H; ++y) {
        int syY  = (int)((int64_t)y * Y.comp_height  / H);  if (syY  >= Y.comp_height)  syY  = Y.comp_height  - 1;
        int syCb = (int)((int64_t)y * Cb.comp_height / H);  if (syCb >= Cb.comp_height) syCb = Cb.comp_height - 1;
        int syCr = (int)((int64_t)y * Cr.comp_height / H);  if (syCr >= Cr.comp_height) syCr = Cr.comp_height - 1;

        const uint8_t* yrow  = Y.samples.data()  + (size_t)syY  * Y.comp_width;
        const uint8_t* cbrow = Cb.samples.data() + (size_t)syCb * Cb.comp_width;
        const uint8_t* crrow = Cr.samples.data() + (size_t)syCr * Cr.comp_width;
        uint8_t* outrow = out_rgb + (size_t)y * W * 3;

        for (int x = 0; x < W; ++x) {
            int sxY  = (int)((int64_t)x * Y.comp_width  / W);  if (sxY  >= Y.comp_width)  sxY  = Y.comp_width  - 1;
            int sxCb = (int)((int64_t)x * Cb.comp_width / W);  if (sxCb >= Cb.comp_width) sxCb = Cb.comp_width - 1;
            int sxCr = (int)((int64_t)x * Cr.comp_width / W);  if (sxCr >= Cr.comp_width) sxCr = Cr.comp_width - 1;

            int Yv = yrow [sxY];
            int cb = (int)cbrow[sxCb] - 128;
            int cr = (int)crrow[sxCr] - 128;

            int R = Yv + ((91881  * cr) >> 16);
            int G = Yv - ((22554  * cb + 46802 * cr) >> 16);
            int B = Yv + ((116130 * cb) >> 16);

            outrow[3*x+0] = (uint8_t)clamp255(R);
            outrow[3*x+1] = (uint8_t)clamp255(G);
            outrow[3*x+2] = (uint8_t)clamp255(B);
        }
    }
}

} // namespace jpeg
} // namespace imageloader