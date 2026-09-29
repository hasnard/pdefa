#include "imageloader/loader.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

static uint32_t rd32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
         | ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "kullanim: %s <file.png>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];

    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "acilamadi: %s\n", path); return 1; }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());

    std::printf("Dosya : %s\n", path);
    std::printf("Boyut : %zu byte\n", buf.size());

    static const uint8_t SIG[8] = {0x89,'P','N','G',0x0D,0x0A,0x1A,0x0A};
    if (buf.size() < 8 || std::memcmp(buf.data(), SIG, 8) != 0) {
        std::printf("Imza  : HAYIR (PNG degil)\n");
        return 2;
    }
    std::printf("Imza  : EVET\n\n");

    std::printf("%-10s %-8s %10s  %s\n",
                "Offset", "Tip", "Uzunluk", "Not");
    std::printf("---------- -------- ----------  --------------------\n");

    size_t pos = 8;
    while (pos + 12 <= buf.size()) {
        uint32_t len = rd32(buf.data() + pos);
        char type[5] = {0};
        std::memcpy(type, buf.data() + pos + 4, 4);

        const char* note = "";
        if (!std::memcmp(type, "IHDR", 4)) note = "header";
        else if (!std::memcmp(type, "IDAT", 4)) note = "image data";
        else if (!std::memcmp(type, "IEND", 4)) note = "son";
        else if (!std::memcmp(type, "PLTE", 4)) note = "palette";
        else if (!std::memcmp(type, "tRNS", 4)) note = "transparency";
        else if (!std::memcmp(type, "gAMA", 4)) note = "gamma";
        else if (!std::memcmp(type, "sRGB", 4)) note = "sRGB profili";
        else if (!std::memcmp(type, "pHYs", 4)) note = "fiziksel boyut";
        else if (!std::memcmp(type, "tEXt", 4)) note = "metin";
        else if (!std::memcmp(type, "iCCP", 4)) note = "ICC profil";

        std::printf("0x%08zX %-8s %10u  %s\n", pos, type, len, note);
        if (!std::memcmp(type, "IEND", 4)) break;
        if (len > 0x7FFFFFFFu) { std::printf("  (bozuk uzunluk)\n"); break; }
        pos += 12 + len;
    }

    if (buf.size() >= 33 && !std::memcmp(buf.data() + 12, "IHDR", 4)) {
        uint32_t w  = rd32(buf.data() + 16);
        uint32_t hh = rd32(buf.data() + 20);
        uint8_t  bd = buf[24];
        uint8_t  ct = buf[25];
        uint8_t  cm = buf[26];
        uint8_t  fm = buf[27];
        uint8_t  il = buf[28];

        const char* ct_name =
            ct == 0 ? "Gray" :
            ct == 2 ? "RGB" :
            ct == 3 ? "Palette" :
            ct == 4 ? "Gray+Alpha" :
            ct == 6 ? "RGBA" : "?";

        std::printf("\n-- IHDR --\n");
        std::printf("Genislik     : %u\n", w);
        std::printf("Yukseklik    : %u\n", hh);
        std::printf("Bit derinligi: %u\n", bd);
        std::printf("Renk tipi    : %u (%s)\n", ct, ct_name);
        std::printf("Compression  : %u\n", cm);
        std::printf("Filter       : %u\n", fm);
        std::printf("Interlace    : %u (%s)\n", il, il ? "Adam7" : "yok");
    }

    std::printf("\n-- Decoder --\n");
    try {
        imageloader::Image img = imageloader::load_image(path);
        const char* fmt =
            img.format == imageloader::PixelFormat::GRAY8  ? "GRAY8"  :
            img.format == imageloader::PixelFormat::RGB8   ? "RGB8"   :
            img.format == imageloader::PixelFormat::RGBA8  ? "RGBA8"  :
            img.format == imageloader::PixelFormat::GRAY16 ? "GRAY16" : "?";
        std::printf("Durum   : BASARILI\n");
        std::printf("Boyut   : %ux%u\n", img.width, img.height);
        std::printf("Format  : %s (%zu kanal)\n", fmt, img.channels());
        std::printf("Bellek  : %zu byte\n", img.pixels.size());
        return 0;
    } catch (const std::exception& e) {
        std::printf("Durum   : HATA\n");
        std::printf("Mesaj   : %s\n", e.what());
        return 3;
    }
}