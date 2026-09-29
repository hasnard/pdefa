#pragma once
#include "imageloader/engine_api.hpp"
#include <string>
#include <vector>

namespace engine_app {

struct Detection {
    int    class_id   = -1;
    float  confidence = 0.f;
    float  x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;   // orijinal goruntu koordinati
};

struct PipelineResult {
    bool                    ok = false;
    std::string             error;
    double                  preprocess_ms = 0.0;
    double                  inference_ms  = 0.0;
    double                  postprocess_ms = 0.0;
    std::vector<Detection>  detections;
};

struct PipelineConfig {
    int   input_w    = 640;
    int   input_h    = 640;
    float conf_thres = 0.25f;
    float iou_thres  = 0.45f;
    int   num_threads = 12;
};

// Tek görüntü -> tespit listesi
PipelineResult run_pipeline(const char* engine_path,
                            const char* image_path,
                            const PipelineConfig& cfg = {});

}  // namespace engine_app// run_yolo_pipeline() API
