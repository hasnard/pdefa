#include "jpeg_internal.hpp"
#include <cstring>
#include <stdexcept>

namespace imageloader {
namespace jpeg {

// Zigzag: index = zigzag pozisyonu -> dogal sira pozisyonu
const int ZIGZAG[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// ------------------------------------------------------------------
// Huffman tablosu kur (canonical)
// ------------------------------------------------------------------
static void build_huff(HuffTable& t) {
    int code = 0;
    int k    = 0;
    for (int len = 1; len <= 16; ++len) {
        if (t.bits[len] > 0) {
            t.valptr[len]  = k;
            t.mincode[len] = code;
            code += t.bits[len];
            k    += t.bits[len];
            t.maxcode[len] = code - 1;
        } else {
            t.maxcode[len] = -1;
        }
        code <<= 1;
    }
    t.num_vals = k;
    t.valid    = true;
}

// ------------------------------------------------------------------
static uint16_t rd_be16(const uint8_t* p) {
    return (uint16_t)(((uint16_t)p[0] << 8) | p[1]);
}

// ------------------------------------------------------------------
// Tum marker'lari tara, JPEGData'yi doldur, scan_start'i kaydet
// ------------------------------------------------------------------
void parse_jpeg(const uint8_t* data, size_t size, JPEGData& j) {
    if (size < 4 || data[0] != 0xFF || data[1] != 0xD8)
        throw std::runtime_error("jpeg: SOI bulunamadi");

    size_t p = 2;
    bool saw_sof = false;
    bool saw_sos = false;

    while (p + 1 < size) {
        if (data[p] != 0xFF) { p++; continue; }
        uint8_t marker = data[p + 1];
        p += 2;

        // Dolgu 0xFF'leri atla
        while (marker == 0xFF && p < size) { marker = data[p++]; }

        if (marker == 0xD8) continue;               // SOI
        if (marker == 0xD9) break;                  // EOI
        if (marker == 0x01) continue;               // TEM
        if (marker >= 0xD0 && marker <= 0xD7) continue;  // RSTn (scan oncesi)

        if (p + 1 >= size)
            throw std::runtime_error("jpeg: segment boyu eksik");
        uint16_t seglen = rd_be16(data + p);
        if (seglen < 2 || p + seglen > size)
            throw std::runtime_error("jpeg: segment tasmasi");
        const uint8_t* seg  = data + p + 2;
        size_t         segn = seglen - 2;

        // ---- DQT ----
        if (marker == 0xDB) {
            size_t o = 0;
            while (o < segn) {
                uint8_t pq_tq = seg[o++];
                int pq = pq_tq >> 4;
                int tq = pq_tq & 0x0F;
                if (pq != 0)
                    throw std::runtime_error("jpeg: 16-bit quant yok");
                if (tq > 3)
                    throw std::runtime_error("jpeg: quant tablo id");
                if (o + 64 > segn)
                    throw std::runtime_error("jpeg: DQT eksik");
                for (int i = 0; i < 64; ++i)
                    j.quant[tq][ZIGZAG[i]] = (uint16_t)seg[o + i];
                j.quant_set[tq] = true;
                o += 64;
            }
        }
        // ---- SOF0 (baseline) ----
        else if (marker == 0xC0) {
            if (segn < 6)
                throw std::runtime_error("jpeg: SOF0 kisa");
            j.precision = seg[0];
            if (j.precision != 8)
                throw std::runtime_error("jpeg: sadece 8-bit desteklenir");
            j.height = (seg[1] << 8) | seg[2];
            j.width  = (seg[3] << 8) | seg[4];
            j.num_components = seg[5];
            if (j.num_components < 1 || j.num_components > MAX_COMPONENTS)
                throw std::runtime_error("jpeg: kanal sayisi");
            size_t o = 6;
            for (int c = 0; c < j.num_components; ++c) {
                if (o + 3 > segn)
                    throw std::runtime_error("jpeg: SOF0 eksik");
                Component& comp = j.comps[c];
                comp.id       = seg[o];
                comp.h_samp   = seg[o + 1] >> 4;
                comp.v_samp   = seg[o + 1] & 0x0F;
                comp.quant_id = seg[o + 2];
                if (comp.h_samp < 1 || comp.h_samp > 4 ||
                    comp.v_samp < 1 || comp.v_samp > 4)
                    throw std::runtime_error("jpeg: sampling factor");
                o += 3;
            }
            // h_max, v_max
            j.h_max = 1; j.v_max = 1;
            for (int c = 0; c < j.num_components; ++c) {
                if (j.comps[c].h_samp > j.h_max) j.h_max = j.comps[c].h_samp;
                if (j.comps[c].v_samp > j.v_max) j.v_max = j.comps[c].v_samp;
            }
            j.mcus_per_line = (j.width  + 8 * j.h_max - 1) / (8 * j.h_max);
            j.mcus_per_col  = (j.height + 8 * j.v_max - 1) / (8 * j.v_max);

            // Component boyutlari
            for (int c = 0; c < j.num_components; ++c) {
                Component& cp = j.comps[c];
                cp.blocks_per_line = j.mcus_per_line * cp.h_samp;
                cp.blocks_per_col  = j.mcus_per_col  * cp.v_samp;
                cp.comp_width      = cp.blocks_per_line * 8;
                cp.comp_height     = cp.blocks_per_col  * 8;
                cp.samples.assign((size_t)cp.comp_width * cp.comp_height, 0);
            }
            saw_sof = true;
        }
        // ---- SOF2 (progressive) - desteklenmiyor ----
        else if (marker == 0xC2) {
            throw std::runtime_error(
                "jpeg: progressive JPEG desteklenmiyor (baseline gerekli)");
        }
        // ---- DHT ----
        else if (marker == 0xC4) {
            size_t o = 0;
            while (o < segn) {
                if (o + 1 > segn)
                    throw std::runtime_error("jpeg: DHT eksik");
                uint8_t tc_th = seg[o++];
                int tc = tc_th >> 4;
                int th = tc_th & 0x0F;
                if (tc > 1 || th > 3)
                    throw std::runtime_error("jpeg: DHT tip");
                if (o + 16 > segn)
                    throw std::runtime_error("jpeg: DHT kisa");
                HuffTable& t = (tc == 0) ? j.huff_dc[th] : j.huff_ac[th];
                int total = 0;
                for (int i = 1; i <= 16; ++i) {
                    t.bits[i] = seg[o + i - 1];
                    total += t.bits[i];
                }
                o += 16;
                if (total > 256 || o + (size_t)total > segn)
                    throw std::runtime_error("jpeg: DHT vals");
                for (int i = 0; i < total; ++i)
                    t.vals[i] = seg[o + i];
                o += (size_t)total;
                build_huff(t);
            }
        }
        // ---- DRI ----
        else if (marker == 0xDD) {
            if (segn < 2) throw std::runtime_error("jpeg: DRI kisa");
            j.restart_interval = (seg[0] << 8) | seg[1];
        }
        // ---- SOS ----
        else if (marker == 0xDA) {
            if (!saw_sof)
                throw std::runtime_error("jpeg: SOS'tan once SOF yok");
            if (segn < 4)
                throw std::runtime_error("jpeg: SOS kisa");
            j.scan_ncomp = seg[0];
            if (j.scan_ncomp < 1 || j.scan_ncomp > j.num_components)
                throw std::runtime_error("jpeg: SOS ncomp");
            size_t o = 1;
            for (int i = 0; i < j.scan_ncomp; ++i) {
                if (o + 2 > segn)
                    throw std::runtime_error("jpeg: SOS eksik");
                int cid = seg[o];
                int tbl = seg[o + 1];
                int ci  = -1;
                for (int c = 0; c < j.num_components; ++c) {
                    if (j.comps[c].id == cid) { ci = c; break; }
                }
                if (ci < 0)
                    throw std::runtime_error("jpeg: SOS bilinmeyen kanal");
                j.scan_comp[i] = ci;
                j.comps[ci].dc_table = tbl >> 4;
                j.comps[ci].ac_table = tbl & 0x0F;
                o += 2;
            }
            if (o + 3 > segn)
                throw std::runtime_error("jpeg: SOS sonu");
            // Baseline icin: Ss=0, Se=63, Ah=0, Al=0
            if (seg[o] != 0 || seg[o + 1] != 63 || seg[o + 2] != 0)
                throw std::runtime_error("jpeg: baseline disi scan (progressive?)");

            // Entropy data baslangici = SOS segmentinden sonra
            j.scan_start = p + seglen;
            saw_sos = true;
            break;
        }
        // digerleri (APPn, COM) atlanir

        p += seglen;
    }

    if (!saw_sof) throw std::runtime_error("jpeg: SOF bulunamadi");
    if (!saw_sos) throw std::runtime_error("jpeg: SOS bulunamadi");
}

} // namespace jpeg
} // namespace imageloader