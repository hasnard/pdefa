// JPEG decoder testleri
// On kosul: tools/gen_test_jpeg.py calistirilarak test_images/*.jpg uretilmis olmali.
#include "imageloader/loader.hpp"
#include "imageloader/jpeg_decoder.hpp"

#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>

namespace fs = std::filesystem;

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("  [FAIL] %s\n", msg); ++g_fail; } \
    else         { std::printf("  [ OK ] %s\n", msg); } \
} while (0)

// ------------------------------------------------------------------
static void test_invalid() {
    std::printf("[TEST] Gecersiz JPEG\n");
    const uint8_t bad[4] = {0xFF, 0xD8, 0xFF, 0x00};
    bool threw = false;
    try {
        imageloader::jpeg::decode_jpeg(bad, 4);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "eksik veri reddedildi");

    bool threw2 = false;
    try {
        imageloader::jpeg::decode_jpeg(nullptr, 0);
    } catch (const std::exception&) {
        threw2 = true;
    }
    CHECK(threw2, "bos veri reddedildi");
}

// ------------------------------------------------------------------
static void test_decode_file(const char* path, int expect_w, int expect_h,
                              const char* label)
{
    std::printf("[TEST] JPEG decode: %s\n", label);

    if (!fs::exists(path)) {
        std::printf("  [SKIP] dosya yok: %s\n", path);
        return;
    }

    try {
        imageloader::Image img = imageloader::load_image(path);
        std::printf("  boyut = %ux%u, format = %d\n",
                    img.width, img.height, (int)img.format);
        CHECK(img.width  == (uint32_t)expect_w, "genislik");
        CHECK(img.height == (uint32_t)expect_h, "yukseklik");
        CHECK(img.format == imageloader::PixelFormat::RGB8, "format RGB8");
        CHECK(img.pixels.size() ==
              (size_t)expect_w * (size_t)expect_h * 3, "piksel boyutu");
    } catch (const std::exception& e) {
        std::printf("  [FAIL] exception: %s\n", e.what());
        ++g_fail;
    }
}

// ------------------------------------------------------------------
int main() {
    std::printf("=== imageloader JPEG test ===\n");

    test_invalid();
    test_decode_file("test_images/test_red_8x8.jpg",  8,   8,  "kirmizi 8x8");
    test_decode_file("test_images/test_gray_16x16.jpg", 16, 16, "gray 16x16");
    test_decode_file("test_images/test_photo_64x64.jpg", 64, 64, "foto 64x64");

    if (g_fail == 0) {
        std::printf("\nTUM JPEG TESTLERI GECTI\n");
        return 0;
    }
    std::printf("\n%d test basarisiz\n", g_fail);
    return 1;
}