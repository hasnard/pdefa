#include "png_internal.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace imageloader {
namespace png {

static uint32_t g_crc_tab[256];
static bool     g_crc_init = false;

static void crc_init() {
    if (g_crc_init) return;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        g_crc_tab[i] = c;
    }
    g_crc_init = true;
}

static uint32_t crc32_bytes(const uint8_t* d, size_t n) {
    crc_init();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i)
        c = g_crc_tab[(c ^ d[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static uint32_t rd_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
         | ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
}

static void parse_ihdr(const uint8_t* d, size_t n, IHDR& out) {
    if (n != 13) throw std::runtime_error("png: IHDR uzunlugu 13 olmali");
    out.width       = rd_be32(d + 0);
    out.height      = rd_be32(d + 4);
    out.bit_depth   = d[8];
    out.color_type  = d[9];
    out.compression = d[10];
    out.filter      = d[11];
    out.interlace   = d[12];
}

PNGInfo read_chunks(const uint8_t* data, size_t size) {
    PNGInfo info;
    size_t pos = 8;
    bool seen_ihdr = false;
    bool seen_iend = false;

    while (pos + 12 <= size) {
        uint32_t len = rd_be32(data + pos);
        if (len > 0x7FFFFFFFu)
            throw std::runtime_error("png: chunk uzunlugu gecersiz");

        size_t type_off = pos + 4;
        size_t data_off = pos + 8;
        size_t data_end = data_off + len;
        size_t crc_off  = data_end;

        if (crc_off + 4 > size)
            throw std::runtime_error("png: chunk tasmasi");

        char type[5] = {0};
        std::memcpy(type, data + type_off, 4);

        uint32_t stored = rd_be32(data + crc_off);
        uint32_t calc   = crc32_bytes(data + type_off, 4 + len);
        if (stored != calc) {
            char msg[128];
            std::snprintf(msg, sizeof(msg),
                "png: CRC hatasi (chunk='%c%c%c%c')",
                type[0], type[1], type[2], type[3]);
            throw std::runtime_error(msg);
        }

        std::string t(type, 4);

        if (t == "IHDR") {
            if (!seen_ihdr) {
                parse_ihdr(data + data_off, len, info.ihdr);
                seen_ihdr = true;
            }
        } else if (t == "IDAT") {
            info.idat.insert(info.idat.end(),
                             data + data_off, data + data_end);
        } else if (t == "PLTE") {
            if (len == 0 || len % 3 != 0 || len > 3 * 256)
                throw std::runtime_error("png: PLTE boyu gecersiz");
            info.plte.assign(data + data_off, data + data_end);
            info.has_plte = true;
        } else if (t == "tRNS") {
            info.trns.assign(data + data_off, data + data_end);
            info.has_trns = true;
        } else if (t == "IEND") {
            if (len != 0)
                throw std::runtime_error("png: IEND uzunlugu 0 olmali");
            seen_iend = true;
            break;
        }
        // gAMA, sRGB, tEXt, iCCP, pHYs vs. atla

        pos = crc_off + 4;
    }

    if (!seen_ihdr)       throw std::runtime_error("png: IHDR bulunamadi");
    if (info.idat.empty()) throw std::runtime_error("png: IDAT bos");
    if (!seen_iend)       throw std::runtime_error("png: IEND bulunamadi");

    if (info.ihdr.compression != 0)
        throw std::runtime_error("png: compression != 0");
    if (info.ihdr.filter != 0)
        throw std::runtime_error("png: filter != 0");
    if (info.ihdr.interlace > 1)
        throw std::runtime_error("png: interlace gecersiz");
    if (info.ihdr.width == 0 || info.ihdr.height == 0)
        throw std::runtime_error("png: 0 boyut");

    uint8_t ct = info.ihdr.color_type;
    uint8_t bd = info.ihdr.bit_depth;

    if (ct == 3) {
        if (!info.has_plte)
            throw std::runtime_error("png: palette eksik (PLTE)");
        if (bd != 1 && bd != 2 && bd != 4 && bd != 8)
            throw std::runtime_error("png: palette bit_depth gecersiz");
    } else if (ct == 0) {
        if (bd != 1 && bd != 2 && bd != 4 && bd != 8 && bd != 16)
            throw std::runtime_error("png: gray bit_depth gecersiz");
    } else if (ct == 2 || ct == 4 || ct == 6) {
        if (bd != 8 && bd != 16)
            throw std::runtime_error("png: bit_depth 8 veya 16 olmali");
    } else {
        throw std::runtime_error("png: bilinmeyen color_type");
    }

    return info;
}

} // namespace png
} // namespace imageloader