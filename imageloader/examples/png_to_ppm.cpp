// PNG → PPM (P6 binary, RGB 8-bit)
// Sonra `display out.ppm` / `feh out.ppm` / Windows'ta IrfanView ile aç
#include "imageloader/loader.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <exception>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "kullanım: %s <in.png> <out.ppm>\n", argv[0]);
        return 1;
    }
    try {
        imageloader::Image img = imageloader::load_image(argv[1]);

        std::ofstream f(argv[2], std::ios::binary);
        if (!f) { std::fprintf(stderr, "yazamadım: %s\n", argv[2]); return 1; }

        f << "P6\n" << img.width << " " << img.height << "\n255\n";

        const size_t ch = img.channels();
        for (size_t i = 0; i < (size_t)img.width * img.height; ++i) {
            uint8_t r = 0, g = 0, b = 0;
            if (ch == 1) {
                r = g = b = img.pixels[i];
            } else if (ch == 3) {
                r = img.pixels[i*3 + 0];
                g = img.pixels[i*3 + 1];
                b = img.pixels[i*3 + 2];
            } else if (ch == 4) {
                r = img.pixels[i*4 + 0];
                g = img.pixels[i*4 + 1];
                b = img.pixels[i*4 + 2];
            }
            f.put((char)r); f.put((char)g); f.put((char)b);
        }

        std::printf("Yazıldı: %s  (%ux%u, %s)\n",
                    argv[2], img.width, img.height,
                    ch == 1 ? "Gray→RGB" :
                    ch == 3 ? "RGB" :
                    ch == 4 ? "RGBA→RGB" : "?");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hata: %s\n", e.what());
        return 1;
    }
    return 0;
}