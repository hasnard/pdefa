#pragma once
#include "imageloader/image.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace imageloader {
namespace jpeg {

constexpr int MAX_COMPONENTS = 4;

// Zigzag: dogal sira -> hangi zigzag pozisyonundan okunur
extern const int ZIGZAG[64];

struct HuffTable {
    uint8_t bits[17]  = {0};      // bits[i] = uzunluk i kod sayisi (i=1..16)
    uint8_t vals[256] = {0};      // semboller (kod sirasi)
    int     mincode[17] = {0};
    int     maxcode[17] = {0};    // -1 = bu uzunlukta kod yok
    int     valptr[17]  = {0};
    int     num_vals    = 0;
    bool    valid       = false;
};

struct Component {
    int id        = 0;
    int h_samp    = 1;
    int v_samp    = 1;
    int quant_id  = 0;
    int dc_table  = 0;
    int ac_table  = 0;
    int dc_pred   = 0;

    // decode sirasinda hesaplanir
    int comp_width     = 0;   // ornek cinsinden
    int comp_height    = 0;
    int blocks_per_line = 0;
    int blocks_per_col  = 0;
    std::vector<uint8_t> samples;   // comp_width * comp_height
};

struct JPEGData {
    int width       = 0;
    int height      = 0;
    int precision   = 8;
    int num_components = 0;
    Component comps[MAX_COMPONENTS];
    int h_max = 1;
    int v_max = 1;
    int mcus_per_line = 0;
    int mcus_per_col  = 0;

    uint16_t quant[4][64] = {{0}};
    bool     quant_set[4] = {false, false, false, false};

    HuffTable huff_dc[4];
    HuffTable huff_ac[4];

    int restart_interval = 0;

    // Scan bilgisi
    int    scan_ncomp = 0;
    int    scan_comp[MAX_COMPONENTS] = {0, 0, 0, 0};
    size_t scan_start = 0;   // entropy data baslangic offset
    size_t scan_end   = 0;   // (kullanilmiyor simdilik)
};

// ------------------------------------------------------------------
// Bit okuyucu (entropy-coded segment icin)
// ------------------------------------------------------------------
class BitReader {
public:
    BitReader(const uint8_t* data, size_t size, size_t start)
        : data_(data), size_(size), pos_(start),
          bit_buf_(0), bit_count_(0), marker_hit_(false) {}

    uint32_t get_bits(int n);
    int      decode_huff(const HuffTable& t);
    int      receive_extend(int s);
    void     align();
    bool     expect_restart();
    void     reset();

    size_t byte_pos() const { return pos_; }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
    uint32_t bit_buf_;
    int bit_count_;
    bool marker_hit_;
};

// ------------------------------------------------------------------
// Modul fonksiyonlari
// ------------------------------------------------------------------
void parse_jpeg(const uint8_t* data, size_t size, JPEGData& j);

// jpeg_idct.cpp
void dequant_zigzag(const int16_t* in, const uint16_t* q, int16_t* out);
void idct_8x8(const int16_t* in, uint8_t* out, int out_stride);

// jpeg_color.cpp
void ycbcr_to_rgb(const JPEGData& j, uint8_t* out_rgb);

} // namespace jpeg
} // namespace imageloader