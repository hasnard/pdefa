#include "pdefa_c_api.h"

#include "engine/engine_c_api.h"       /* public C API */

#include "imageloader/loader.hpp"
#include "imageloader/image.hpp"
#include "imageloader/preprocess.hpp"
#include "imageloader/engine_api.hpp"

#include <cstring>
#include <exception>
#include <memory>
#include <string>

namespace {
thread_local std::string g_err;
inline void set_err(const char* m) { try { g_err = m ? m : ""; } catch (...) {} }
inline void clear_err()             { try { g_err.clear(); } catch (...) {} }

template <typename Fn>
EngineStatus safe_call(Fn&& fn) noexcept {
    try { return fn(); }
    catch (const std::bad_alloc&) { set_err("out of memory"); return ENGINE_ERR_OUT_OF_MEMORY; }
    catch (const std::exception& e) { set_err(e.what()); return ENGINE_ERR_RUNTIME; }
    catch (...) { set_err("unknown exception"); return ENGINE_ERR_UNKNOWN; }
}
} // namespace

/* ============ PdefaImage ============ */
struct PdefaImage {
    imageloader::Image inner;
};

/* ============ META ============ */
extern "C" PDEFA_API const char* PDEFA_CALL pdefa_version(void) {
    return engine_version();
}
extern "C" PDEFA_API const char* PDEFA_CALL pdefa_last_error(void) {
    return g_err.c_str();
}

/* ============ IMAGE ============ */
extern "C" PDEFA_API PdefaImage* PDEFA_CALL pdefa_image_load(const char* path) {
    if (!path) { set_err("path null"); return nullptr; }
    clear_err();
    try {
        auto img = std::make_unique<PdefaImage>();
        img->inner = imageloader::load_image(path);
        return img.release();
    } catch (const std::exception& e) { set_err(e.what()); return nullptr; }
      catch (...)                    { set_err("unknown"); return nullptr; }
}
extern "C" PDEFA_API void PDEFA_CALL pdefa_image_free(PdefaImage* img) { delete img; }

extern "C" PDEFA_API int32_t PDEFA_CALL pdefa_image_width(const PdefaImage* img) {
    return img ? (int32_t)img->inner.width : 0;
}
extern "C" PDEFA_API int32_t PDEFA_CALL pdefa_image_height(const PdefaImage* img) {
    return img ? (int32_t)img->inner.height : 0;
}
extern "C" PDEFA_API int32_t PDEFA_CALL pdefa_image_channels(const PdefaImage* img) {
    return img ? (int32_t)img->inner.channels() : 0;
}
extern "C" PDEFA_API int32_t PDEFA_CALL pdefa_image_format(const PdefaImage* img) {
    if (!img) return -1;
    switch (img->inner.format) {
        case imageloader::PixelFormat::GRAY8:  return PDEFA_PIXFMT_GRAY8;
        case imageloader::PixelFormat::RGB8:   return PDEFA_PIXFMT_RGB8;
        case imageloader::PixelFormat::RGBA8:  return PDEFA_PIXFMT_RGBA8;
        case imageloader::PixelFormat::GRAY16: return PDEFA_PIXFMT_GRAY16;
    }
    return -1;
}
extern "C" PDEFA_API const uint8_t* PDEFA_CALL pdefa_image_pixels(
    const PdefaImage* img, size_t* out_bytes)
{
    if (!img) return nullptr;
    if (out_bytes) *out_bytes = img->inner.pixels.size();
    return img->inner.pixels.data();
}

/* ============ PREPROCESS ============ */
extern "C" PDEFA_API EngineStatus PDEFA_CALL pdefa_preprocess_letterbox(
    const PdefaImage* img,
    const PdefaPreprocessConfig* cfg,
    EngineTensor** out_tensor,
    PdefaPreprocessMeta* out_meta)
{
    if (!img || !cfg || !out_tensor) { set_err("null arg"); return ENGINE_ERR_NULL_PTR; }
    *out_tensor = nullptr;

    return safe_call([&]() -> EngineStatus {
        clear_err();

        imageloader::PreprocessConfig icfg;
        icfg.target_w  = cfg->target_w;
        icfg.target_h  = cfg->target_h;
        icfg.scale     = cfg->scale;
        icfg.pad_value = cfg->pad_value;

        imageloader::PreprocessResult pr =
            imageloader::preprocess_letterbox(img->inner, icfg);

        if (out_meta) {
            out_meta->target_w = pr.target_w;
            out_meta->target_h = pr.target_h;
            out_meta->orig_w   = (int32_t)img->inner.width;
            out_meta->orig_h   = (int32_t)img->inner.height;
            out_meta->pad_x    = pr.pad_x;
            out_meta->pad_y    = pr.pad_y;
            out_meta->scaled_w = pr.scaled_w;
            out_meta->scaled_h = pr.scaled_h;
            out_meta->ratio    = pr.ratio;
        }

        int64_t shape[4] = { 1, 3, (int64_t)pr.target_h, (int64_t)pr.target_w };

        /* PUBLIC C API kullaniyoruz — EngineTensor'a dogrudan erismiyoruz */
        return engine_tensor_create_from_data(
            out_tensor,
            shape, 4,
            ENGINE_F32,
            pr.data.data(),
            pr.data.size() * sizeof(float));
    });
}

extern "C" PDEFA_API EngineStatus PDEFA_CALL pdefa_prepare_for_engine(
    const char* path,
    int32_t target_w, int32_t target_h,
    float scale, float pad_value,
    EngineTensor** out_tensor,
    PdefaPreprocessMeta* out_meta)
{
    if (!path || !out_tensor) { set_err("null arg"); return ENGINE_ERR_NULL_PTR; }
    *out_tensor = nullptr;

    return safe_call([&]() -> EngineStatus {
        clear_err();

        imageloader::EngineInput in = imageloader::prepare_for_engine(
            path, target_w, target_h, scale, pad_value);
        if (!in.valid()) { set_err("prepare_for_engine failed"); return ENGINE_ERR_RUNTIME; }

        if (out_meta) {
            out_meta->target_w = in.W;
            out_meta->target_h = in.H;
            out_meta->orig_w   = in.info.orig_w;
            out_meta->orig_h   = in.info.orig_h;
            out_meta->pad_x    = in.info.pad_x;
            out_meta->pad_y    = in.info.pad_y;
            out_meta->scaled_w = in.info.scaled_w;
            out_meta->scaled_h = in.info.scaled_h;
            out_meta->ratio    = in.info.ratio;
        }

        int64_t shape[4] = { 1, 3, (int64_t)in.H, (int64_t)in.W };

        return engine_tensor_create_from_data(
            out_tensor,
            shape, 4,
            ENGINE_F32,
            in.tensor.data(),
            in.tensor.size() * sizeof(float));
    });
}

extern "C" PDEFA_API EngineStatus PDEFA_CALL pdefa_load_image_as_tensor(
    const char* image_path, int32_t target_w, int32_t target_h,
    EngineTensor** out_tensor)
{
    return pdefa_prepare_for_engine(image_path, target_w, target_h,
                                    1.0f / 255.0f, 0.0f, out_tensor, nullptr);
}

/* ============ POST-PROCESS ============ */
extern "C" PDEFA_API EngineStatus PDEFA_CALL pdefa_map_box_to_original(
    const float* in_box, const PdefaPreprocessMeta* meta, float* out_box)
{
    if (!in_box || !meta || !out_box) { set_err("null arg"); return ENGINE_ERR_NULL_PTR; }

    imageloader::EngineInput::ImageInfo info;
    info.orig_w   = meta->orig_w;
    info.orig_h   = meta->orig_h;
    info.pad_x    = meta->pad_x;
    info.pad_y    = meta->pad_y;
    info.scaled_w = meta->scaled_w;
    info.scaled_h = meta->scaled_h;
    info.ratio    = meta->ratio;

    imageloader::Box b{ in_box[0], in_box[1], in_box[2], in_box[3] };
    imageloader::Box r = imageloader::map_box_to_original(b, info);
    out_box[0] = r.x1; out_box[1] = r.y1;
    out_box[2] = r.x2; out_box[3] = r.y2;
    return ENGINE_OK;
}

extern "C" PDEFA_API EngineStatus PDEFA_CALL pdefa_map_point_to_original(
    float* x, float* y, const PdefaPreprocessMeta* meta)
{
    if (!x || !y || !meta) { set_err("null arg"); return ENGINE_ERR_NULL_PTR; }

    imageloader::EngineInput::ImageInfo info;
    info.orig_w   = meta->orig_w;
    info.orig_h   = meta->orig_h;
    info.pad_x    = meta->pad_x;
    info.pad_y    = meta->pad_y;
    info.scaled_w = meta->scaled_w;
    info.scaled_h = meta->scaled_h;
    info.ratio    = meta->ratio;

    imageloader::map_point_to_original(*x, *y, info);
    return ENGINE_OK;
}