#include "pipeline.hpp"
#include "imageloader/loader.hpp"
#include "imageloader/preprocess.hpp"
#include "engine/loader/custom_loader.hpp"
#include "engine/session.hpp"
#include "engine/tensor.hpp"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace engine_app {

namespace {

struct RawBox {
    float x1, y1, x2, y2;
    float score;
    int   cls;
};

float iou(const RawBox& a, const RawBox& b) {
    const float xx1 = std::max(a.x1, b.x1);
    const float yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2);
    const float yy2 = std::min(a.y2, b.y2);
    const float w = std::max(0.f, xx2 - xx1);
    const float h = std::max(0.f, yy2 - yy1);
    const float inter = w * h;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return (uni > 0.f) ? (inter / uni) : 0.f;
}

void nms(std::vector<RawBox>& boxes, float iou_thres) {
    std::sort(boxes.begin(), boxes.end(),
              [](const RawBox& a, const RawBox& b){ return a.score > b.score; });

    std::vector<RawBox> keep;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        keep.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            if (boxes[i].cls != boxes[j].cls) continue;
            if (iou(boxes[i], boxes[j]) > iou_thres)
                suppressed[j] = true;
        }
    }
    boxes = std::move(keep);
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<RawBox> decode_yolov8(const float* data,
                                   int dim1, int dim2,
                                   float conf_thres,
                                   bool& out_need_sigmoid)
{
    int num_channels, num_preds;
    bool channel_first;

    if (dim1 < dim2) {
        num_channels  = dim1;
        num_preds     = dim2;
        channel_first = true;
    } else {
        num_preds     = dim1;
        num_channels  = dim2;
        channel_first = false;
    }

    const int num_classes = num_channels - 4;

    bool need_sigmoid = false;
    {
        int sample = std::min(num_preds, 100);
        float mx = -1e30f;
        for (int i = 0; i < sample; ++i) {
            for (int c = 0; c < num_classes; ++c) {
                float s = channel_first ? data[(4 + c) * num_preds + i]
                                        : data[i * num_channels + 4 + c];
                if (s > mx) mx = s;
            }
        }
        need_sigmoid = (mx > 1.5f);
    }
    out_need_sigmoid = need_sigmoid;

    std::vector<RawBox> boxes;
    boxes.reserve(256);

    for (int i = 0; i < num_preds; ++i) {
        float cx, cy, w, h;
        if (channel_first) {
            cx = data[0 * num_preds + i];
            cy = data[1 * num_preds + i];
            w  = data[2 * num_preds + i];
            h  = data[3 * num_preds + i];
        } else {
            cx = data[i * num_channels + 0];
            cy = data[i * num_channels + 1];
            w  = data[i * num_channels + 2];
            h  = data[i * num_channels + 3];
        }

        float best_score = 0.f;
        int   best_cls   = -1;
        for (int c = 0; c < num_classes; ++c) {
            float s = channel_first ? data[(4 + c) * num_preds + i]
                                    : data[i * num_channels + 4 + c];
            if (need_sigmoid) s = sigmoid(s);
            if (s > best_score) {
                best_score = s;
                best_cls   = c;
            }
        }

        if (best_score < conf_thres) continue;

        RawBox b;
        b.x1 = cx - w * 0.5f;
        b.y1 = cy - h * 0.5f;
        b.x2 = cx + w * 0.5f;
        b.y2 = cy + h * 0.5f;
        b.score = best_score;
        b.cls   = best_cls;
        boxes.push_back(b);
    }
    return boxes;
}

}  // namespace

PipelineResult run_pipeline(const char* engine_path,
                            const char* image_path,
                            const PipelineConfig& cfg)
{
    PipelineResult res;

    // ============ PREPROCESS (iki parcali olcum) ============
    auto t_load0 = std::chrono::high_resolution_clock::now();

    imageloader::Image img;
    try {
        img = imageloader::load_image(image_path);
    } catch (const std::exception& e) {
        res.error = std::string("decode: ") + e.what();
        return res;
    }
    if (img.width == 0 || img.height == 0) {
        res.error = "decode: gecersiz resim";
        return res;
    }

    auto t_load1 = std::chrono::high_resolution_clock::now();

    imageloader::PreprocessConfig pcfg;
    pcfg.target_w = cfg.input_w;
    pcfg.target_h = cfg.input_h;

    imageloader::PreprocessResult pr = imageloader::preprocess_letterbox(img, pcfg);
    if (!pr.valid()) {
        res.error = "preprocess: gecersiz sonuc";
        return res;
    }

    auto t_load2 = std::chrono::high_resolution_clock::now();

    const double decode_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();
    const double resize_ms = std::chrono::duration<double, std::milli>(t_load2 - t_load1).count();

    std::fprintf(stderr, "[TIME] decode=%.2f ms   resize=%.2f ms\n", decode_ms, resize_ms);

    res.preprocess_ms = decode_ms + resize_ms;

    // ============ MODEL ============
    engine::Model model;
    if (!engine::loader::load_engine_file(engine_path, model)) {
        res.error = "model yuklenemedi";
        return res;
    }

    engine::runtime::GlobalPool::set_num_threads(cfg.num_threads);
    engine::Session session(model);

    int64_t dims[4] = {1, 3, cfg.input_h, cfg.input_w};
    engine::Tensor t_in = engine::Tensor::from_external(
        const_cast<float*>(pr.data.data()),
        engine::Shape(dims, 4),
        engine::DType::F32);

    session.set_input(0, &t_in);

    for (int i = 0; i < 3; ++i) session.run();

    // ============ INFERENCE ============
    auto t2 = std::chrono::high_resolution_clock::now();
    if (!session.run()) {
        res.error = "inference FAIL";
        return res;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    res.inference_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ============ OUTPUT ============
    const engine::Tensor* out = session.output(0);
    if (!out || out->is_empty()) {
        res.error = "output yok";
        return res;
    }
    if (out->rank() != 3) {
        res.error = "beklenmeyen output rank";
        return res;
    }

    const int dim1 = (int)out->dim(1);
    const int dim2 = (int)out->dim(2);

    // ============ DECODE + NMS ============
    auto t4 = std::chrono::high_resolution_clock::now();

    bool need_sigmoid = false;
    const float* data = out->data<float>();
    auto raw = decode_yolov8(data, dim1, dim2, cfg.conf_thres, need_sigmoid);
    nms(raw, cfg.iou_thres);

    auto t5 = std::chrono::high_resolution_clock::now();
    res.postprocess_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // ============ ORIGINAL KOORDINAT ============
    res.detections.reserve(raw.size());
    for (const auto& b : raw) {
        imageloader::Box box;
        box.x1 = b.x1; box.y1 = b.y1; box.x2 = b.x2; box.y2 = b.y2;
        imageloader::EngineInput::ImageInfo info;
        info.orig_w   = (int)img.width;
        info.orig_h   = (int)img.height;
        info.pad_x    = pr.pad_x;
        info.pad_y    = pr.pad_y;
        info.scaled_w = pr.scaled_w;
        info.scaled_h = pr.scaled_h;
        info.ratio    = pr.ratio;

        imageloader::Box orig = imageloader::map_box_to_original(box, info);

        Detection d;
        d.class_id   = b.cls;
        d.confidence = b.score;
        d.x1 = orig.x1;
        d.y1 = orig.y1;
        d.x2 = orig.x2;
        d.y2 = orig.y2;
        res.detections.push_back(d);
    }

    res.ok = true;
    return res;
}

}  // namespace engine_app