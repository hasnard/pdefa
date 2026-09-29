#include "jpeg_internal.hpp"
#include <stdexcept>

namespace imageloader {
namespace jpeg {

// ------------------------------------------------------------------
// BitReader
// ------------------------------------------------------------------
uint32_t BitReader::get_bits(int n) {
    while (bit_count_ < n) {
        if (marker_hit_ || pos_ >= size_) {
            // Marker'a carptik ya da veri bitti: sifir ile doldur
            bit_buf_ = (bit_buf_ << 8);
            bit_count_ += 8;
            continue;
        }
        uint8_t b = data_[pos_];
        if (b == 0xFF) {
            if (pos_ + 1 >= size_) {
                marker_hit_ = true;
                continue;
            }
            uint8_t nx = data_[pos_ + 1];
            if (nx == 0x00) {
                // Stuffed byte
                pos_ += 2;
                bit_buf_ = (bit_buf_ << 8) | 0xFFu;
                bit_count_ += 8;
            } else {
                // Gercek marker (RSTn, EOI vs.) -> dur
                marker_hit_ = true;
            }
        } else {
            pos_ += 1;
            bit_buf_ = (bit_buf_ << 8) | b;
            bit_count_ += 8;
        }
    }
    bit_count_ -= n;
    return (bit_buf_ >> bit_count_) & ((1u << n) - 1u);
}

void BitReader::align() {
    bit_buf_ = 0;
    bit_count_ = 0;
}

void BitReader::reset() {
    bit_buf_ = 0;
    bit_count_ = 0;
    marker_hit_ = false;
}

bool BitReader::expect_restart() {
    reset();
    // pos_ bir RSTn marker'inda olmali
    while (pos_ + 1 < size_) {
        if (data_[pos_] == 0xFF) {
            uint8_t m = data_[pos_ + 1];
            if (m >= 0xD0 && m <= 0xD7) {
                pos_ += 2;
                return true;
            }
            if (m != 0x00) return false;  // baska bir marker
            pos_ += 2;  // stuffed, atla
        } else {
            pos_ += 1;
        }
    }
    return false;
}

// ------------------------------------------------------------------
// Huffman decoding (canonical, bit-by-bit)
// ------------------------------------------------------------------
int BitReader::decode_huff(const HuffTable& t) {
    int code = 0;
    for (int len = 1; len <= 16; ++len) {
        code = (code << 1) | (int)get_bits(1); 
        if (t.maxcode[len] >= 0 && code <= t.maxcode[len]) {
            int idx = t.valptr[len] + code - t.mincode[len];
            if (idx < 0 || idx >= t.num_vals)
                throw std::runtime_error("jpeg: Huffman index tasmasi");
            return t.vals[idx];
        }
    }
    throw std::runtime_error("jpeg: gecersiz Huffman kodu");
}

// ------------------------------------------------------------------
// EXTEND: s-bit degeri isaretli tamsayiya cevir
// ------------------------------------------------------------------
static int extend_val(int v, int t) {
    if (t == 0) return 0;
    if (v < (1 << (t - 1)))
        v += (-1 << t) + 1;
    return v;
}

int BitReader::receive_extend(int s) {
    if (s == 0) return 0;
    int v = (int)get_bits(s);
    return extend_val(v, s);
}

} // namespace jpeg
} // namespace imageloader