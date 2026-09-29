#include "imageloader/engine_api.hpp"
#include "imageloader/loader.hpp"
#include "imageloader/preprocess.hpp"

#include <algorithm>
#include <stdexcept>

namespace imageloader {

// ------------------------------------------------------------------
EngineInput prepare_from_image(const Image& img,
                                int   target_w,
                                int   target_h,
                                float scale,
                                float pad_val)
{
    PreprocessConfig cfg;
    cfg.target_w  = target_w;
    cfg.target_h  = target_h;
    cfg.scale     = scale;
    cfg.pad_value = pad_val;

    PreprocessResult pr = preprocess_letterbox(img, cfg);

    EngineInput out;
    if (!pr.valid()) return out;

    out.tensor = std::move(pr.data);
    out.N = 1;
    out.C = 3;
    out.H = pr.target_h;
    out.W = pr.target_w;

    out.info.orig_w   = (int)img.width;
    out.info.orig_h   = (int)img.height;
    out.info.pad_x    = pr.pad_x;
    out.info.pad_y    = pr.pad_y;
    out.info.scaled_w = pr.scaled_w;
    out.info.scaled_h = pr.scaled_h;
    out.info.ratio    = pr.ratio;

    return out;
}

// ------------------------------------------------------------------
EngineInput prepare_for_engine(const std::string& path,
                                int   target_w,
                                int   target_h,
                                float scale,
                                float pad_val)
{
    Image img = load_image(path);
    return prepare_from_image(img, target_w, target_h, scale, pad_val);
}

// ------------------------------------------------------------------
// Post-process: model koordinati -> orijinal
// ------------------------------------------------------------------
Box map_box_to_original(const Box& box, const EngineInput::ImageInfo& info) {
    auto map_x = [&](float x) -> float {
        float v = (x - (float)info.pad_x) / info.ratio;
        if (v < 0.f) v = 0.f;
        if (v > (float)info.orig_w) v = (float)info.orig_w;
        return v;
    };
    auto map_y = [&](float y) -> float {
        float v = (y - (float)info.pad_y) / info.ratio;
        if (v < 0.f) v = 0.f;
        if (v > (float)info.orig_h) v = (float)info.orig_h;
        return v;
    };

    Box out;
    out.x1 = map_x(box.x1);
    out.y1 = map_y(box.y1);
    out.x2 = map_x(box.x2);
    out.y2 = map_y(box.y2);
    return out;
}

void map_point_to_original(float& x, float& y,
                            const EngineInput::ImageInfo& info)
{
    x = (x - (float)info.pad_x) / info.ratio;
    y = (y - (float)info.pad_y) / info.ratio;
    if (x < 0.f) x = 0.f;
    if (y < 0.f) y = 0.f;
    if (x > (float)info.orig_w) x = (float)info.orig_w;
    if (y > (float)info.orig_h) y = (float)info.orig_h;
}

// ------------------------------------------------------------------
// Batch
// ------------------------------------------------------------------
std::vector<EngineInput> prepare_batch(const std::vector<std::string>& paths,
                                        int target_w,
                                        int target_h)
{
    std::vector<EngineInput> out;
    out.reserve(paths.size());

    for (const auto& p : paths) {
        try {
            out.push_back(prepare_for_engine(p, target_w, target_h));
        } catch (const std::exception&) {
            out.emplace_back();   // invalid input, sira bozulmaz
        }
    }
    return out;
}

} // namespace imageloader