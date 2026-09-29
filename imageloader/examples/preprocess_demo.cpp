// Kullanim: preprocess_demo <image.png>
// PNG'yi yukler, 640x640 letterbox yapar, tensor bilgisi basar.
#include "imageloader/preprocess.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "kullanim: %s <image.png>\n", argv[0]);
        return 1;
    }

    try {
        imageloader::PreprocessConfig cfg;
        cfg.target_w  = 640;
        cfg.target_h  = 640;
        cfg.scale     = 1.0f / 255.0f;
        cfg.pad_value = 0.0f;

        auto r = imageloader::load_and_preprocess(argv[1], cfg);

        if (!r.valid()) {
            std::fprintf(stderr, "hata: preprocess basarisiz\n");
            return 1;
        }

        std::printf("Dosya  : %s\n", argv[1]);
        std::printf("Tensor : [3][%d][%d]  (%zu float)\n",
                    r.target_h, r.target_w, r.data.size());
        std::printf("Scale  : %dx%d (ratio=%.4f)\n",
                    r.scaled_w, r.scaled_h, r.ratio);
        std::printf("Pad    : x=%d y=%d\n", r.pad_x, r.pad_y);
        std::printf("Ornek  : data[0]=%.4f  data[son]=%.4f\n",
                    r.data.front(), r.data.back());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hata: %s\n", e.what());
        return 1;
    }
    return 0;
}