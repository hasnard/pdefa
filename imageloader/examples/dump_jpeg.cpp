// Kullanim: dump_jpeg <image.jpg>
#include "imageloader/loader.hpp"
#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "kullanim: %s <image.jpg>\n", argv[0]);
        return 1;
    }
    try {
        imageloader::Image img = imageloader::load_image(argv[1]);
        std::printf("Dosya : %s\n", argv[1]);
        std::printf("Boyut : %ux%u\n", img.width, img.height);
        std::printf("Format: RGB8 (3 kanal)\n\n");

        for (uint32_t y = 0; y < img.height; ++y) {
            for (uint32_t x = 0; x < img.width; ++x) {
                size_t i = ((size_t)y * img.width + x) * 3;
                std::printf("(%3u,%3u) = %3u %3u %3u\n",
                            x, y,
                            img.pixels[i], img.pixels[i+1], img.pixels[i+2]);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hata: %s\n", e.what());
        return 1;
    }
    return 0;
}