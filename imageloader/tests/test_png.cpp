// Motor yok — saf assert + printf.
#include "imageloader/loader.hpp"
#include "imageloader/png_decoder.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

// -------------------- yardımcı: PNG üretici --------------------
static void put_u32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((x >> 24) & 0xFF);
    v.push_back((x >> 16) & 0xFF);
    v.push_back((x >>  8) & 0xFF);
    v.push_back( x        & 0xFF);
}

static uint32_t crc32_bytes(const uint8_t* d, size_t n) {
    static uint32_t tab[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            tab[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i)
        c = tab[(c ^ d[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static void put_chunk(std::vector<uint8_t>& out,
                      const char* type,
                      const std::vector<uint8_t>& data) {
    put_u32(out, (uint32_t)data.size());
    size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    uint32_t c = crc32_bytes(out.data() + start, 4 + data.size());
    put_u32(out, c);
}

static uint32_t adler32_bytes(const std::vector<uint8_t>& d) {
    uint32_t s1 = 1, s2 = 0;
    for (uint8_t b : d) {
        s1 = (s1 + b) % 65521u;
        s2 = (s2 + s1) % 65521u;
    }
    return (s2 << 16) | s1;
}

// raw_rows = her satır için [filter_byte, row_bytes...]
static std::vector<uint8_t> make_png(uint32_t w, uint32_t h,
                                     uint8_t color_type,
                                     const std::vector<uint8_t>& raw_rows) {
    std::vector<uint8_t> png = {0x89,'P','N','G',0x0D,0x0A,0x1A,0x0A};

    std::vector<uint8_t> ihdr;
    put_u32(ihdr, w);
    put_u32(ihdr, h);
    ihdr.push_back(8);            // bit depth
    ihdr.push_back(color_type);
    ihdr.push_back(0);            // compression
    ihdr.push_back(0);            // filter method
    ihdr.push_back(0);            // interlace
    put_chunk(png, "IHDR", ihdr);

    // zlib: header + stored deflate bloklar + adler32
    std::vector<uint8_t> z;
    z.push_back(0x78);
    z.push_back(0x01);

    size_t pos = 0;
    while (pos < raw_rows.size()) {
        size_t n     = std::min<size_t>(65535, raw_rows.size() - pos);
        bool   final = (pos + n == raw_rows.size());
        z.push_back(final ? 0x01 : 0x00);
        z.push_back(n & 0xFF);
        z.push_back((n >> 8) & 0xFF);
        uint16_t nlen = (uint16_t)~(uint16_t)n;
        z.push_back(nlen & 0xFF);
        z.push_back((nlen >> 8) & 0xFF);
        z.insert(z.end(), raw_rows.begin() + pos, raw_rows.begin() + pos + n);
        pos += n;
    }

    uint32_t ad = adler32_bytes(raw_rows);
    z.push_back((ad >> 24) & 0xFF);
    z.push_back((ad >> 16) & 0xFF);
    z.push_back((ad >>  8) & 0xFF);
    z.push_back( ad        & 0xFF);

    put_chunk(png, "IDAT", z);
    put_chunk(png, "IEND", {});
    return png;
}

// -------------------- yardımcı: filtre uygula (test için) --------------------
static uint8_t paeth(uint8_t a, uint8_t b, uint8_t c) {
    int p = (int)a + (int)b - (int)c;
    int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

static void filter_row(uint8_t filter,
                       const uint8_t* cur, const uint8_t* prev,
                       size_t bpp, size_t row_bytes,
                       std::vector<uint8_t>& out) {
    out.resize(row_bytes);
    for (size_t x = 0; x < row_bytes; ++x) {
        uint8_t a = (x >= bpp) ? cur[x - bpp] : 0;
        uint8_t b = prev ? prev[x] : 0;
        uint8_t c = (prev && x >= bpp) ? prev[x - bpp] : 0;
        uint8_t v = cur[x];
        uint8_t f = 0;
        switch (filter) {
            case 0: f = v; break;
            case 1: f = (uint8_t)(v - a); break;
            case 2: f = (uint8_t)(v - b); break;
            case 3: f = (uint8_t)(v - ((a + b) >> 1)); break;
            case 4: f = (uint8_t)(v - paeth(a, b, c)); break;
        }
        out[x] = f;
    }
}

// -------------------- test sonuç raporu --------------------
static int  g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("  [FAIL] %s\n", msg); ++g_fail; } \
    else         { std::printf("  [ OK ] %s\n", msg); } \
} while (0)

// -------------------- TEST 1: RGB 8x8 --------------------
static void test_rgb_8x8() {
    std::printf("[TEST] RGB 8x8\n");
    const uint32_t W = 8, H = 8, BPP = 3;
    const size_t row_bytes = W * BPP;

    // İstenen pikseller: (x*32, y*32, (x+y)*16)
    std::vector<uint8_t> pixels(W * H * BPP);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            size_t i = ((size_t)y * W + x) * BPP;
            pixels[i + 0] = (uint8_t)(x * 32);
            pixels[i + 1] = (uint8_t)(y * 32);
            pixels[i + 2] = (uint8_t)((x + y) * 16);
        }

    // Her satıra farklı filtre uygula
    std::vector<uint8_t> raw;
    for (uint32_t y = 0; y < H; ++y) {
        uint8_t filter = (uint8_t)(y % 5);
        std::vector<uint8_t> fr;
        filter_row(filter,
                   pixels.data() + (size_t)y * row_bytes,
                   (y == 0) ? nullptr : pixels.data() + (size_t)(y - 1) * row_bytes,
                   BPP, row_bytes, fr);
        raw.push_back(filter);
        raw.insert(raw.end(), fr.begin(), fr.end());
    }

    std::vector<uint8_t> png = make_png(W, H, 2, raw);

    // Diske yaz
    fs::create_directories("test_images");
    {
        std::ofstream f("test_images/test_rgb_8x8.png", std::ios::binary);
        f.write(reinterpret_cast<const char*>(png.data()), (std::streamsize)png.size());
    }

    // Diskten oku
    imageloader::Image img = imageloader::load_image("test_images/test_rgb_8x8.png");
    CHECK(img.width == W && img.height == H, "boyut 8x8");
    CHECK(img.format == imageloader::PixelFormat::RGB8, "format RGB8");
    CHECK(img.pixels.size() == W * H * BPP, "piksel boyutu");

    bool eq = std::equal(img.pixels.begin(), img.pixels.end(), pixels.begin());
    CHECK(eq, "piksel değerleri birebir");
}

// -------------------- TEST 2: RGBA 4x4 (düz kırmızı) --------------------
static void test_rgba_4x4() {
    std::printf("[TEST] RGBA 4x4\n");
    const uint32_t W = 4, H = 4, BPP = 4;

    std::vector<uint8_t> pixels(W * H * BPP);
    for (size_t i = 0; i < W * H; ++i) {
        pixels[i*4 + 0] = 255;
        pixels[i*4 + 1] = 0;
        pixels[i*4 + 2] = 0;
        pixels[i*4 + 3] = 255;
    }

    std::vector<uint8_t> raw;
    for (uint32_t y = 0; y < H; ++y) {
        // filter 0 = None
        raw.push_back(0);
        raw.insert(raw.end(),
                   pixels.begin() + (size_t)y * W * BPP,
                   pixels.begin() + (size_t)(y + 1) * W * BPP);
    }

    std::vector<uint8_t> png = make_png(W, H, 6, raw);

    fs::create_directories("test_images");
    {
        std::ofstream f("test_images/test_rgba_4x4.png", std::ios::binary);
        f.write(reinterpret_cast<const char*>(png.data()), (std::streamsize)png.size());
    }

    imageloader::Image img = imageloader::load_image("test_images/test_rgba_4x4.png");
    CHECK(img.width == W && img.height == H, "boyut 4x4");
    CHECK(img.format == imageloader::PixelFormat::RGBA8, "format RGBA8");
    CHECK(std::equal(img.pixels.begin(), img.pixels.end(), pixels.begin()), "pikseller");
}

// -------------------- TEST 3: GRAY 16x16 --------------------
static void test_gray_16x16() {
    std::printf("[TEST] GRAY 16x16\n");
    const uint32_t W = 16, H = 16, BPP = 1;

    std::vector<uint8_t> pixels(W * H);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x)
            pixels[(size_t)y * W + x] = (uint8_t)(x * 16 + y);

    std::vector<uint8_t> raw;
    for (uint32_t y = 0; y < H; ++y) {
        raw.push_back((uint8_t)(y % 5));  // filtre çeşitliliği
        std::vector<uint8_t> fr;
        filter_row(raw.back(),
                   pixels.data() + (size_t)y * W,
                   (y == 0) ? nullptr : pixels.data() + (size_t)(y - 1) * W,
                   BPP, W, fr);
        raw.insert(raw.end(), fr.begin(), fr.end());
    }

    std::vector<uint8_t> png = make_png(W, H, 0, raw);

    fs::create_directories("test_images");
    {
        std::ofstream f("test_images/test_gray_16x16.png", std::ios::binary);
        f.write(reinterpret_cast<const char*>(png.data()), (std::streamsize)png.size());
    }

    imageloader::Image img = imageloader::load_image("test_images/test_gray_16x16.png");
    CHECK(img.width == W && img.height == H, "boyut 16x16");
    CHECK(img.format == imageloader::PixelFormat::GRAY8, "format GRAY8");
    CHECK(std::equal(img.pixels.begin(), img.pixels.end(), pixels.begin()), "pikseller");
}

// -------------------- TEST 4: Bozuk imza --------------------
static void test_bad_signature() {
    std::printf("[TEST] Bozuk imza\n");
    const uint8_t junk[8] = {0,0,0,0,0,0,0,0};
    bool threw = false;
    try {
        imageloader::png::decode_png(junk, 8);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "hatalı imza reddedildi");
}

// -------------------- main --------------------
int main() {
    std::printf("=== imageloader test ===\n");
    test_rgb_8x8();
    test_rgba_4x4();
    test_gray_16x16();
    test_bad_signature();

    if (g_fail == 0) {
        std::printf("\nTÜM TESTLER GEÇTİ ✔\n");
        return 0;
    }
    std::printf("\n%d test başarısız ✘\n", g_fail);
    return 1;
}