#include "png_internal.hpp"
#include <stdexcept>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdint>

namespace imageloader {
namespace png {

namespace {

// ------------------------------------------------------------------
// 64-bit Akümülatörlü Hızlı Bit Okuyucu
// ------------------------------------------------------------------
struct FastBitReader {
    const uint8_t* in_ptr;
    const uint8_t* in_end;
    uint64_t bitbuf = 0;
    int bits = 0;

    FastBitReader(const uint8_t* src, size_t len)
        : in_ptr(src), in_end(src + len) {}

    inline void refill() {
        while (bits <= 56 && in_ptr < in_end) {
            bitbuf |= static_cast<uint64_t>(*in_ptr++) << bits;
            bits += 8;
        }
    }

    inline uint32_t peek(int n) {
        if (bits < n) refill();
        return static_cast<uint32_t>(bitbuf & ((1ULL << n) - 1));
    }

    inline void drop(int n) {
        bitbuf >>= n;
        bits -= n;
    }

    inline uint32_t read(int n) {
        if (bits < n) refill();
        if (bits < n) throw std::runtime_error("inflate: veri bitti");
        uint32_t v = static_cast<uint32_t>(bitbuf & ((1ULL << n) - 1));
        bitbuf >>= n;
        bits -= n;
        return v;
    }

    inline void align() {
        int rem = bits & 7;
        if (rem) {
            bitbuf >>= rem;
            bits -= rem;
        }
    }
};

// ------------------------------------------------------------------
// Çift Kademeli Hızlı Huffman Lookup Table (LUT)
// ------------------------------------------------------------------
struct TableEntry {
    uint16_t sym = 0xFFFF;
    uint8_t  len = 255;
};

struct FastHuffman {
    std::vector<TableEntry> primary;
    std::vector<TableEntry> secondary;
    int primary_bits = 10;
    uint32_t primary_mask = 0x3FF;

    void build(const uint8_t* lengths, int num_symbols, int p_bits = 10) {
        primary_bits = p_bits;
        primary_mask = (1u << primary_bits) - 1;
        primary.assign(1u << primary_bits, {0xFFFF, 255});
        secondary.clear();

        int counts[16] = {0};
        for (int i = 0; i < num_symbols; ++i) {
            if (lengths[i] > 0 && lengths[i] <= 15) {
                counts[lengths[i]]++;
            }
        }

        int next_code[16] = {0};
        int code = 0;
        for (int len = 1; len <= 15; ++len) {
            code = (code + counts[len - 1]) << 1;
            next_code[len] = code;
        }

        uint16_t secondary_used = 0;

        for (int i = 0; i < num_symbols; ++i) {
            int len = lengths[i];
            if (len == 0) continue;

            int c = next_code[len]++;
            uint32_t rev = 0;
            for (int b = 0; b < len; ++b) {
                rev = (rev << 1) | ((c >> b) & 1);
            }

            if (len <= primary_bits) {
                uint32_t step = 1u << len;
                for (uint32_t idx = rev; idx < (1u << primary_bits); idx += step) {
                    primary[idx] = {static_cast<uint16_t>(i), static_cast<uint8_t>(len)};
                }
            } else {
                uint32_t low = rev & primary_mask;
                uint32_t high = rev >> primary_bits;
                int rem_len = len - primary_bits;

                if (primary[low].len != 99) {
                    primary[low].sym = secondary_used;
                    primary[low].len = 99; // İkincil tablo belirteci
                    secondary_used += 32;
                    secondary.resize(secondary_used, {0xFFFF, 255});
                }

                uint16_t base = primary[low].sym;
                uint32_t step = 1u << rem_len;
                for (uint32_t idx = high; idx < 32; idx += step) {
                    secondary[base + idx] = {static_cast<uint16_t>(i), static_cast<uint8_t>(rem_len)};
                }
            }
        }
    }

    inline int decode(FastBitReader& br) const {
        if (br.bits < 15) br.refill();
        uint32_t b = static_cast<uint32_t>(br.bitbuf & primary_mask);
        TableEntry e = primary[b];

        if (e.len <= primary_bits) {
            br.bitbuf >>= e.len;
            br.bits   -= e.len;
            return e.sym;
        }
        if (e.len == 99) {
            br.bitbuf >>= primary_bits;
            br.bits   -= primary_bits;
            uint32_t sub_b = static_cast<uint32_t>(br.bitbuf & 31);
            TableEntry sub_e = secondary[e.sym + sub_b];
            br.bitbuf >>= sub_e.len;
            br.bits   -= sub_e.len;
            return sub_e.sym;
        }
        throw std::runtime_error("inflate: geçersiz Huffman kodu");
    }
};

const int LEN_BASE[29]  = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,
                           51,59,67,83,99,115,131,163,195,227,258};
const int LEN_EXTRA[29] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,
                           3,3,4,4,4,4,5,5,5,5,0};
const int DIST_BASE[30]  = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,
                            257,385,513,769,1025,1537,2049,3073,4097,6145,
                            8193,12289,16385,24577};
const int DIST_EXTRA[30] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,
                            9,9,10,10,11,11,12,12,13,13};

void build_fixed(FastHuffman& lit, FastHuffman& dist) {
    uint8_t lit_len[288];
    for (int i = 0;   i < 144; ++i) lit_len[i] = 8;
    for (int i = 144; i < 256; ++i) lit_len[i] = 9;
    for (int i = 256; i < 280; ++i) lit_len[i] = 7;
    for (int i = 280; i < 288; ++i) lit_len[i] = 8;
    lit.build(lit_len, 288, 10);

    uint8_t dist_len[32];
    for (int i = 0; i < 32; ++i) dist_len[i] = 5;
    dist.build(dist_len, 32, 8);
}

void build_dynamic(FastBitReader& br, FastHuffman& lit, FastHuffman& dist) {
    int hlit  = (int)br.read(5) + 257;
    int hdist = (int)br.read(5) + 1;
    int hclen = (int)br.read(4) + 4;

    static const int ORDER[19] =
        {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};

    uint8_t clen[19] = {0};
    for (int i = 0; i < hclen; ++i) clen[ORDER[i]] = (uint8_t)br.read(3);

    FastHuffman cl_huff;
    cl_huff.build(clen, 19, 7);

    std::vector<uint8_t> lengths(hlit + hdist, 0);
    int i = 0;
    while (i < hlit + hdist) {
        int sym = cl_huff.decode(br);
        if (sym < 16) {
            lengths[i++] = (uint8_t)sym;
        } else if (sym == 16) {
            if (i == 0) throw std::runtime_error("inflate: prev length yok");
            int rep = 3 + (int)br.read(2);
            uint8_t prev = lengths[i - 1];
            while (rep-- && i < hlit + hdist) lengths[i++] = prev;
        } else if (sym == 17) {
            int rep = 3 + (int)br.read(3);
            while (rep-- && i < hlit + hdist) lengths[i++] = 0;
        } else { // 18
            int rep = 11 + (int)br.read(7);
            while (rep-- && i < hlit + hdist) lengths[i++] = 0;
        }
    }
    lit.build(lengths.data(), hlit, 10);
    dist.build(lengths.data() + hlit, hdist, 8);
}

void inflate_compressed(FastBitReader& br,
                        const FastHuffman& lit,
                        const FastHuffman& dist,
                        std::vector<uint8_t>& out,
                        size_t& out_len)
{
    while (true) {
        int sym = lit.decode(br);
        if (sym < 256) {
            if (out_len >= out.size()) {
                out.resize(out.size() * 2 + 32768);
            }
            out[out_len++] = static_cast<uint8_t>(sym);
        } else if (sym == 256) {
            break; // EOB
        } else {
            int li = sym - 257;
            if (li < 0 || li >= 29) throw std::runtime_error("inflate: len sembolü");
            int length = LEN_BASE[li];
            int extra = LEN_EXTRA[li];
            if (extra) length += (int)br.read(extra);

            int ds = dist.decode(br);
            if (ds < 0 || ds >= 30) throw std::runtime_error("inflate: dist sembolü");
            int distance = DIST_BASE[ds];
            int d_extra = DIST_EXTRA[ds];
            if (d_extra) distance += (int)br.read(d_extra);

            if ((size_t)distance > out_len)
                throw std::runtime_error("inflate: distance out of range");

            if (out_len + length > out.size()) {
                out.resize(std::max(out.size() * 2, out_len + length + 32768));
            }

            uint8_t* dst = out.data() + out_len;
            const uint8_t* src = dst - distance;
            out_len += length;

            // Hızlı LZ77 Kopyalama
            if (distance == 1) {
                std::memset(dst, *src, length);
            } else if (distance >= 8 && length >= 8) {
                int rem = length;
                while (rem >= 8) {
                    std::memcpy(dst, src, 8);
                    dst += 8;
                    src += 8;
                    rem -= 8;
                }
                while (rem > 0) {
                    *dst++ = *src++;
                    rem--;
                }
            } else {
                for (int k = 0; k < length; ++k) {
                    *dst++ = *src++;
                }
            }
        }
    }
}

} // namespace

std::vector<uint8_t> zlib_inflate(const uint8_t* data, size_t size) {
    if (size < 2) throw std::runtime_error("zlib: veri çok küçük");
    uint8_t cmf = data[0];
    uint8_t flg = data[1];
    if ((cmf & 0x0F) != 8) throw std::runtime_error("zlib: CM != 8");
    if (((cmf << 8 | flg) % 31) != 0) throw std::runtime_error("zlib: FCHECK");
    if (flg & 0x20) throw std::runtime_error("zlib: preset dict desteklenmiyor");

    FastBitReader br{data + 2, size - 2};

    // İlk boyutu tahmin ederek tahsis et (böylece reallocation döngüsü neredeyse hiç çalışmaz)
    std::vector<uint8_t> out;
    out.resize(size * 4 + 65536);
    size_t out_len = 0;

    FastHuffman fixed_lit, fixed_dist;
    bool fixed_ready = false;

    bool final = false;
    while (!final) {
        final = br.read(1) != 0;
        int type = (int)br.read(2);

        if (type == 0) {
            br.align();
            uint32_t len  = br.read(16);
            uint32_t nlen = br.read(16);
            if ((len ^ 0xFFFFu) != nlen)
                throw std::runtime_error("inflate: stored LEN/NLEN uyuşmuyor");
            if (out_len + len > out.size()) {
                out.resize(out_len + len + 32768);
            }
            for (uint32_t i = 0; i < len; ++i) {
                out[out_len++] = static_cast<uint8_t>(br.read(8));
            }
        } else if (type == 1) {
            if (!fixed_ready) {
                build_fixed(fixed_lit, fixed_dist);
                fixed_ready = true;
            }
            inflate_compressed(br, fixed_lit, fixed_dist, out, out_len);
        } else if (type == 2) {
            FastHuffman lit, dist;
            build_dynamic(br, lit, dist);
            inflate_compressed(br, lit, dist, out, out_len);
        } else {
            throw std::runtime_error("inflate: geçersiz blok tipi");
        }
    }

    out.resize(out_len);
    return out;
}

} // namespace png
} // namespace imageloader