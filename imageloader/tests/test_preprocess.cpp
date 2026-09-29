// preprocess_letterbox + load_and_preprocess testleri
#include "imageloader/preprocess.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("  [FAIL] %s\n", msg); ++g_fail; } \
    else         { std::printf("  [ OK ] %s\n", msg); } \
} while (0)

#define CHECK_NEAR(a, b, eps, msg) do { \
    if (std::fabs((a) - (b)) > (eps)) { \
        std::printf("  [FAIL] %s (%.6f vs %.6f)\n", \
                    msg, (double)(a), (double)(b)); \
        ++g_fail; \
    } else { std::printf("  [ OK ] %s\n", msg); } \
} while (0)

// ------------------------------------------------------------------
static void test_square_no_padding() {
    std::printf("[TEST] Kare goruntu, kare hedef\n");

    imageloader::Image img;
    img.width  = 4;
    img.height = 4;
    img.format = imageloader::PixelFormat::RGB8;
    img.pixels.assign(4 * 4 * 3, 128);

    imageloader::PreprocessConfig cfg;
    cfg.target_w = 8;
    cfg.target_h = 8;
    cfg.scale = 1.0f / 255.0f;

    auto r = imageloader::preprocess_letterbox(img, cfg);

    CHECK(r.valid(), "sonuc gecerli");
    CHECK(r.target_w == 8 && r.target_h == 8, "boyut 8x8");
    CHECK(r.data.size() == 3 * 8 * 8, "3x8x8 float");
    CHECK(r.pad_x == 0 && r.pad_y == 0, "padding yok");
    CHECK(r.scaled_w == 8 && r.scaled_h == 8, "scale 8x8");

    CHECK_NEAR(r.data[0], 128.0f / 255.0f, 0.01f, "normalize piksel");
}

// ------------------------------------------------------------------
static void test_wide_letterbox() {
    std::printf("[TEST] Genis goruntu letterbox\n");

    imageloader::Image img;
    img.width  = 100;
    img.height = 50;
    img.format = imageloader::PixelFormat::RGB8;
    img.pixels.assign(100 * 50 * 3, 200);

    imageloader::PreprocessConfig cfg;
    cfg.target_w = 100;
    cfg.target_h = 100;
    cfg.scale = 1.0f / 255.0f;

    auto r = imageloader::preprocess_letterbox(img, cfg);

    CHECK(r.valid(), "sonuc gecerli");
    CHECK(r.scaled_w == 100 && r.scaled_h == 50, "scale 100x50");
    CHECK(r.pad_x == 0 && r.pad_y == 25, "padding (0,25)");

    // Ust padding = 0
    CHECK_NEAR(r.data[50], 0.0f, 0.001f, "ust padding = 0");

    // Orta satir = 200/255
    size_t idx_mid = (size_t)50 * 100 + 50;
    CHECK_NEAR(r.data[idx_mid], 200.0f / 255.0f, 0.02f, "orta deger");
}

// ------------------------------------------------------------------
static void test_gray_to_rgb() {
    std::printf("[TEST] Gray -> 3 kanal\n");

    imageloader::Image img;
    img.width  = 4;
    img.height = 4;
    img.format = imageloader::PixelFormat::GRAY8;
    img.pixels.assign(4 * 4, 100);

    imageloader::PreprocessConfig cfg;
    cfg.target_w = 4;
    cfg.target_h = 4;
    cfg.scale = 1.0f / 255.0f;

    auto r = imageloader::preprocess_letterbox(img, cfg);

    CHECK(r.data.size() == 3 * 4 * 4, "3 kanal");

    size_t plane = 16;
    CHECK_NEAR(r.data[0], r.data[plane],     0.001f, "R == G");
    CHECK_NEAR(r.data[0], r.data[2 * plane], 0.001f, "R == B");
    CHECK_NEAR(r.data[0], 100.0f / 255.0f,   0.01f,  "deger dogru");
}

// ------------------------------------------------------------------
static void test_rgba_to_rgb() {
    std::printf("[TEST] RGBA -> 3 kanal\n");

    imageloader::Image img;
    img.width  = 2;
    img.height = 2;
    img.format = imageloader::PixelFormat::RGBA8;
    img.pixels = {
        255,0,0,255,  0,255,0,128,
        0,0,255,50,   255,255,255,0
    };

    imageloader::PreprocessConfig cfg;
    cfg.target_w = 2;
    cfg.target_h = 2;
    cfg.scale = 1.0f / 255.0f;

    auto r = imageloader::preprocess_letterbox(img, cfg);

    CHECK(r.data.size() == 3 * 2 * 2, "3 kanal, 2x2");

    size_t plane = 4;
    CHECK_NEAR(r.data[0],         1.0f, 0.01f, "R=1");
    CHECK_NEAR(r.data[plane],     0.0f, 0.01f, "G=0");
    CHECK_NEAR(r.data[2 * plane], 0.0f, 0.01f, "B=0");
}

// ------------------------------------------------------------------
static void test_invalid_input() {
    std::printf("[TEST] Gecersiz input\n");

    imageloader::Image img;
    auto r = imageloader::preprocess_letterbox(img);
    CHECK(!r.valid(), "bos goruntu -> invalid");

    imageloader::Image img2;
    img2.width = 4; img2.height = 4;
    img2.format = imageloader::PixelFormat::RGB8;
    img2.pixels.assign(48, 0);

    imageloader::PreprocessConfig cfg;
    cfg.target_w = 0;
    cfg.target_h = 0;
    auto r2 = imageloader::preprocess_letterbox(img2, cfg);
    CHECK(!r2.valid(), "gecersiz hedef -> invalid");
}

// ------------------------------------------------------------------
static void test_load_and_preprocess() {
    std::printf("[TEST] load_and_preprocess (dosyadan)\n");

    const char* path = "test_images/test_rgb_8x8.png";

    imageloader::PreprocessConfig cfg;
    cfg.target_w = 16;
    cfg.target_h = 16;
    cfg.scale = 1.0f / 255.0f;

    try {
        auto r = imageloader::load_and_preprocess(path, cfg);
        CHECK(r.valid(), "sonuc gecerli");
        CHECK(r.target_w == 16 && r.target_h == 16, "hedef boyut");
        CHECK(r.data.size() == 3 * 16 * 16, "tensor boyutu");
    } catch (const std::exception& e) {
        std::printf("  [FAIL] exception: %s\n", e.what());
        ++g_fail;
    }
}

// ------------------------------------------------------------------
static void test_batch() {
    std::printf("[TEST] Batch preprocessing\n");

    std::vector<std::string> paths = {
        "test_images/test_rgb_8x8.png",
        "test_images/test_rgba_4x4.png",
        "test_images/test_gray_16x16.png",
        "test_images/yok_boyle_dosya.png"
    };

    imageloader::PreprocessConfig cfg;
    cfg.target_w = 32;
    cfg.target_h = 32;

    auto results = imageloader::load_and_preprocess_batch(paths, cfg);

    CHECK(results.size() == 4, "4 sonuc donduruldu");
    CHECK(results[0].valid(), "PNG 1 gecerli");
    CHECK(results[1].valid(), "PNG 2 gecerli");
    CHECK(results[2].valid(), "PNG 3 gecerli");
    CHECK(!results[3].valid(), "hatali dosya -> invalid");
}

// ------------------------------------------------------------------
int main() {
    std::printf("=== imageloader preprocess test ===\n");
    test_square_no_padding();
    test_wide_letterbox();
    test_gray_to_rgb();
    test_rgba_to_rgb();
    test_invalid_input();
    test_load_and_preprocess();
    test_batch();

    if (g_fail == 0) {
        std::printf("\nTUM PREPROCESS TESTLERI GECTI\n");
        return 0;
    }
    std::printf("\n%d test basarisiz\n", g_fail);
    return 1;
}