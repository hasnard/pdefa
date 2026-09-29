#pragma once
#include "imageloader/image.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace imageloader {

// ==================================================================
// Inference engine icin hazir input
// ==================================================================
struct EngineInput {
    // Ana tensor: NCHW float
    std::vector<float> tensor;
    int N = 1;
    int C = 3;
    int H = 0;
    int W = 0;

    // Post-process icin meta (YOLO letterbox bilgileri)
    struct ImageInfo {
        int   orig_w   = 0;
        int   orig_h   = 0;
        int   pad_x    = 0;
        int   pad_y    = 0;
        int   scaled_w = 0;
        int   scaled_h = 0;
        float ratio    = 1.0f;
    } info;

    const float* data() const { return tensor.data(); }
    float*       data()       { return tensor.data(); }
    size_t       size() const { return tensor.size(); }

    // Engine'in bekledigi sekil: {1, 3, H, W}
    std::vector<int64_t> shape() const {
        return { (int64_t)N, (int64_t)C, (int64_t)H, (int64_t)W };
    }

    bool valid() const { return !tensor.empty() && H > 0 && W > 0; }
};

// ==================================================================
// Ana giris noktasi: dosya yolu -> NCHW tensor + meta
// ==================================================================
EngineInput prepare_for_engine(const std::string& path,
                                int   target_w = 640,
                                int   target_h = 640,
                                float scale    = 1.0f / 255.0f,
                                float pad_val  = 0.0f);

// Zaten yuklenmis Image'dan
EngineInput prepare_from_image(const Image& img,
                                int   target_w = 640,
                                int   target_h = 640,
                                float scale    = 1.0f / 255.0f,
                                float pad_val  = 0.0f);

// ==================================================================
// Post-process: model koordinatlarini orijinal goruntuye geri cevir
// ==================================================================
struct Box { float x1, y1, x2, y2; };

// YOLO tarzi: 640x640 model cikisindaki box -> orijinal goruntu
Box  map_box_to_original(const Box& box, const EngineInput::ImageInfo& info);

// Tek nokta (pose, keypoint vs.)
void map_point_to_original(float& x, float& y,
                            const EngineInput::ImageInfo& info);

// ==================================================================
// Batch
// ==================================================================
std::vector<EngineInput> prepare_batch(const std::vector<std::string>& paths,
                                        int target_w = 640,
                                        int target_h = 640);

} // namespace imageloader