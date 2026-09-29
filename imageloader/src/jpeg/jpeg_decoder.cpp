#include "imageloader/jpeg_decoder.hpp"
#include "jpeg_internal.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace imageloader {
namespace jpeg {

// Standart bağımsız hızlı clamp
static inline uint8_t clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<uint8_t>(v);
}

// MCU döngüsündeki tablo aramalarını önbelleğe alan yapı
struct DecCompContext {
    Component* comp;
    const HuffTable* dc_t;
    const HuffTable* ac_t;
    const uint16_t* quant;
    int h_samp;
    int v_samp;
    int comp_w;
    int comp_h;
};

// ------------------------------------------------------------------
// Blok Çözme: memset ile sıfırlama + AC varlık tespiti
// ------------------------------------------------------------------
static inline bool decode_block_fast(BitReader& br,
                                     const HuffTable& dc_tab,
                                     const HuffTable& ac_tab,
                                     int& dc_pred,
                                     int16_t* zz_out)
{
    std::memset(zz_out, 0, 64 * sizeof(int16_t));

    // DC
    int t = br.decode_huff(dc_tab);
    if (t > 15) throw std::runtime_error("jpeg: DC kategori");
    int diff = br.receive_extend(t);
    dc_pred += diff;
    zz_out[0] = static_cast<int16_t>(dc_pred);

    // AC
    int k = 1;
    bool has_ac = false;
    while (k < 64) {
        int rs = br.decode_huff(ac_tab);
        int r = rs >> 4;
        int s = rs & 0x0F;
        if (s == 0) {
            if (r == 15) { k += 16; continue; } // ZRL
            break;                               // EOB
        }
        k += r;
        if (k >= 64) throw std::runtime_error("jpeg: AC run tasmasi");
        zz_out[k++] = static_cast<int16_t>(br.receive_extend(s));
        has_ac = true;
    }

    return has_ac;
}

// ------------------------------------------------------------------
// Optimize Scan Çözücü
// ------------------------------------------------------------------
static void decode_scan(JPEGData& j, const uint8_t* data, size_t size) {
    BitReader br(data, size, j.scan_start);

    DecCompContext ctx[4];
    for (int s = 0; s < j.scan_ncomp; ++s) {
        int ci = j.scan_comp[s];
        Component& cp = j.comps[ci];

        const HuffTable& dc_t = j.huff_dc[cp.dc_table];
        const HuffTable& ac_t = j.huff_ac[cp.ac_table];

        if (!dc_t.valid || !ac_t.valid)
            throw std::runtime_error("jpeg: eksik Huffman tablosu");
        if (!j.quant_set[cp.quant_id])
            throw std::runtime_error("jpeg: eksik quant tablosu");

        ctx[s] = {
            &cp,
            &dc_t,
            &ac_t,
            j.quant[cp.quant_id],
            cp.h_samp,
            cp.v_samp,
            cp.comp_width,
            cp.comp_height
        };
    }

    alignas(16) int16_t zz[64];
    alignas(16) int16_t nat[64];
    alignas(16) uint8_t block[64];

    int mcus_since_restart = 0;
    const int total_mcus = j.mcus_per_line * j.mcus_per_col;

    int mx = 0;
    int my = 0;

    for (int mcu = 0; mcu < total_mcus; ++mcu) {
        for (int s = 0; s < j.scan_ncomp; ++s) {
            DecCompContext& c = ctx[s];
            Component& cp = *c.comp;

            for (int vy = 0; vy < c.v_samp; ++vy) {
                for (int hx = 0; hx < c.h_samp; ++hx) {
                    bool has_ac = decode_block_fast(br, *c.dc_t, *c.ac_t, cp.dc_pred, zz);

                    // DC-Only: AC yoksa IDCT çalıştırma
                    if (!has_ac) {
                        int dc_val = (zz[0] * c.quant[0] + 4) >> 3;
                        uint8_t clamped = clamp_u8(dc_val + 128);
                        std::memset(block, clamped, 64);
                    } else {
                        dequant_zigzag(zz, c.quant, nat);
                        idct_8x8(nat, block, 8);
                    }

                    const int blk_x = (mx * c.h_samp + hx) * 8;
                    const int blk_y = (my * c.v_samp + vy) * 8;

                    // Hızlı yol: Resim içi blokları 8 baytlık satırlar halinde yaz
                    if (blk_x + 8 <= c.comp_w && blk_y + 8 <= c.comp_h) {
                        uint8_t* dst = cp.samples.data() + (size_t)blk_y * c.comp_w + blk_x;
                        for (int by = 0; by < 8; ++by) {
                            std::memcpy(dst, block + by * 8, 8);
                            dst += c.comp_w;
                        }
                    } else {
                        // Sınır blokları için kırpma (clipping)
                        for (int by = 0; by < 8; ++by) {
                            int row = blk_y + by;
                            if (row >= c.comp_h) break;
                            int n = 8;
                            if (blk_x + n > c.comp_w)
                                n = c.comp_w - blk_x;
                            if (n <= 0) continue;
                            std::memcpy(cp.samples.data() + (size_t)row * c.comp_w + blk_x,
                                        block + by * 8,
                                        (size_t)n);
                        }
                    }
                }
            }
        }

        // idiv yerine sayaç
        if (++mx == j.mcus_per_line) {
            mx = 0;
            ++my;
        }

        // Restart Marker Kontrolü
        if (j.restart_interval > 0) {
            if (++mcus_since_restart == j.restart_interval && mcu + 1 < total_mcus) {
                if (!br.expect_restart())
                    throw std::runtime_error("jpeg: RST marker bulunamadi");
                for (int c_idx = 0; c_idx < j.num_components; ++c_idx)
                    j.comps[c_idx].dc_pred = 0;
                mcus_since_restart = 0;
            }
        }
    }
}

// ------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------
Image decode_jpeg(const uint8_t* data, size_t size) {
    if (size < 4 || data[0] != 0xFF || data[1] != 0xD8)
        throw std::runtime_error("jpeg: gecersiz imza (SOI yok)");

    JPEGData j;
    parse_jpeg(data, size, j);

    if (j.scan_start == 0 || j.scan_start >= size)
        throw std::runtime_error("jpeg: scan verisi bulunamadi");

    decode_scan(j, data, size);

    Image img;
    img.width  = (uint32_t)j.width;
    img.height = (uint32_t)j.height;
    img.format = PixelFormat::RGB8;
    img.pixels.resize((size_t)j.width * j.height * 3);

    ycbcr_to_rgb(j, img.pixels.data());
    return img;
}

} // namespace jpeg
} // namespace imageloader