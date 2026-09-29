#include "pipeline.hpp"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("Kullanim: %s <model.engine> <foto.png>\n", argv[0]);
        std::printf("Ornek  : %s yolov8n.engine zidane.png\n", argv[0]);
        return 1;
    }

    engine_app::PipelineConfig cfg;
    cfg.conf_thres = 0.25f;
    cfg.iou_thres  = 0.45f;
    cfg.num_threads = 12;

    auto r = engine_app::run_pipeline(argv[1], argv[2], cfg);

    if (!r.ok) {
        std::fprintf(stderr, "HATA: %s\n", r.error.c_str());
        return 1;
    }

    std::printf("=== Sonuclar ===\n");
    std::printf("Preprocess  : %6.2f ms\n", r.preprocess_ms);
    std::printf("Inference   : %6.2f ms\n", r.inference_ms);
    std::printf("Postprocess : %6.2f ms\n", r.postprocess_ms);
    std::printf("Toplam      : %6.2f ms\n",
                r.preprocess_ms + r.inference_ms + r.postprocess_ms);
    std::printf("\nTespit sayisi: %d\n", (int)r.detections.size());
    const size_t show = std::min<size_t>(20, r.detections.size());
for (size_t i = 0; i < show; ++i) {
    const auto& d = r.detections[i];
    std::printf("  [%.3f] sinif=%-2d  box=[%7.1f, %7.1f, %7.1f, %7.1f]\n",
                d.confidence, d.class_id, d.x1, d.y1, d.x2, d.y2);
}
if (r.detections.size() > 20)
    std::printf("  ... (%d tespit daha)\n",
                (int)(r.detections.size() - 20));
    return 0;
}