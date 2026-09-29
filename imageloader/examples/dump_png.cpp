// PNG oku → ekrana RGB(A) / Gray değerlerini bas
#include "imageloader/loader.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "kullanım: %s <image.png>\n", argv[0]);
        return 1;
    }

    try {
        imageloader::Image img = imageloader::load_image(argv[1]);
        const size_t ch = img.channels();

        const char* fmt_name = "?";
        switch (img.format) {
            case imageloader::PixelFormat::GRAY8:  fmt_name = "GRAY8";  break;
            case imageloader::PixelFormat::RGB8:   fmt_name = "RGB8";   break;
            case imageloader::PixelFormat::RGBA8:  fmt_name = "RGBA8";  break;
            case imageloader::PixelFormat::GRAY16: fmt_name = "GRAY16"; break;
        }

        std::printf("Dosya : %s\n", argv[1]);
        std::printf("Boyut : %ux%u\n", img.width, img.height);
        std::printf("Format: %s (%zu kanal)\n\n", fmt_name, ch);

        for (uint32_t y = 0; y < img.height; ++y) {
            for (uint32_t x = 0; x < img.width; ++x) {
                size_t idx = ((size_t)y * img.width + x) * ch;
                std::printf("(%3u,%3u) = ", x, y);
                for (size_t c = 0; c < ch; ++c)
                    std::printf("%3u ", (unsigned)img.pixels[idx + c]);
                std::printf("\n");
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hata: %s\n", e.what());
        return 1;
    }
    return 0;
}