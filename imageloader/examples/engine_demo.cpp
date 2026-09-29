// Kullanim: engine_demo <image.png|jpg>
#include "imageloader/engine_api.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "kullanim: %s <image>\n", argv[0]);
        return 1;
    }

    try {
        // ------------------------------------------------------
        // 1) Hazirla: dosya -> NCHW tensor + meta
        // ------------------------------------------------------
        imageloader::EngineInput input =
            imageloader::prepare_for_engine(argv[1], 640, 640);

        if (!input.valid()) {
            std::fprintf(stderr, "hata: input hazirlanamadi\n");
            return 1;
        }

        std::printf("Dosya       : %s\n", argv[1]);
        std::printf("Orijinal    : %dx%d\n",
                    input.info.orig_w, input.info.orig_h);
        std::printf("Tensor      : [%d][%d][%d][%d] (%zu float)\n",
                    input.N, input.C, input.H, input.W,
                    input.size());
        std::printf("Letterbox   : scale=%dx%d ratio=%.4f pad=(%d,%d)\n",
                    input.info.scaled_w, input.info.scaled_h,
                    input.info.ratio,
                    input.info.pad_x, input.info.pad_y);

        // ------------------------------------------------------
        // 2) MOTORUNA VER (burada sahte)
        // ------------------------------------------------------
        // Gercek kullanim:
        //   auto output = my_engine.infer(input.data(),
        //                                  input.size(),
        //                                  input.shape());
        std::printf("\n[engine] input.data() = %p, size = %zu\n",
                    (const void*)input.data(), input.size());

        // ------------------------------------------------------
        // 3) Post-process ornegi
        // ------------------------------------------------------
        // Motor 640x640'da box (x1,y1,x2,y2) uretti diyelim
        imageloader::Box model_box{ 320.f, 240.f, 480.f, 400.f };  // ortada kutu

        imageloader::Box orig_box =
            imageloader::map_box_to_original(model_box, input.info);

        std::printf("\nModel box   : (%.1f, %.1f, %.1f, %.1f)\n",
                    model_box.x1, model_box.y1, model_box.x2, model_box.y2);
        std::printf("Orijinal box: (%.1f, %.1f, %.1f, %.1f)\n",
                    orig_box.x1, orig_box.y1, orig_box.x2, orig_box.y2);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hata: %s\n", e.what());
        return 1;
    }
    return 0;
}