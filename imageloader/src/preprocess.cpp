#include "imageloader/preprocess.hpp"
#include "imageloader/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace imageloader {

// ------------------------------------------------------------------
// 1. Sıfır-Tahsis (Zero-Allocation) & Cache-Dostu Çekirdek
// ------------------------------------------------------------------
void preprocess_letterbox_direct(const Image& img, 
                                 const PreprocessConfig& cfg, 
                                 float* out_tensor, 
                                 PreprocessResult& r) 
{
    if (!out_tensor || img.width == 0 || img.height == 0 || img.pixels.empty()) return;

    const int W = cfg.target_w;
    const int H = cfg.target_h;
    const int src_w = static_cast<int>(img.width);
    const int src_h = static_cast<int>(img.height);
    const size_t src_ch = img.channels();

    const float ratio = std::min(static_cast<float>(W) / src_w, static_cast<float>(H) / src_h);
    int scaled_w = std::clamp(static_cast<int>(std::round(src_w * ratio)), 1, W);
    int scaled_h = std::clamp(static_cast<int>(std::round(src_h * ratio)), 1, H);

    const int pad_x = (W - scaled_w) / 2;
    const int pad_y = (H - scaled_h) / 2;
    const size_t plane = static_cast<size_t>(W) * H;

    r.target_w = W; r.target_h = H;
    r.pad_x = pad_x; r.pad_y = pad_y;
    r.scaled_w = scaled_w; r.scaled_h = scaled_h;
    r.ratio = ratio;

    float* dst_r = out_tensor + 0 * plane;
    float* dst_g = out_tensor + 1 * plane;
    float* dst_b = out_tensor + 2 * plane;

    // Üst Dolgu (Tek geçiş)
    if (pad_y > 0) {
        size_t top_len = static_cast<size_t>(pad_y) * W;
        std::fill_n(dst_r, top_len, cfg.pad_value);
        std::fill_n(dst_g, top_len, cfg.pad_value);
        std::fill_n(dst_b, top_len, cfg.pad_value);
    }

    const float inv_ratio = 1.0f / ratio;
    std::vector<int> col_x0(scaled_w), col_x1(scaled_w);
    std::vector<float> col_fx(scaled_w);

    for (int dx = 0; dx < scaled_w; ++dx) {
        float sx = dx * inv_ratio;
        int x0 = static_cast<int>(sx);
        if (x0 > src_w - 2 && src_w > 1) x0 = src_w - 2;
        col_x0[dx] = x0;
        col_x1[dx] = (x0 + 1 < src_w) ? x0 + 1 : x0;
        col_fx[dx] = sx - x0;
    }

    const uint8_t* src = img.pixels.data();
    const float ns = cfg.scale;

    // Orta Görüntü + Yan Dolgular
    for (int dy = 0; dy < scaled_h; ++dy) {
        size_t row_idx = static_cast<size_t>(dy + pad_y) * W;

        // Sol dolgu
        if (pad_x > 0) {
            std::fill_n(dst_r + row_idx, pad_x, cfg.pad_value);
            std::fill_n(dst_g + row_idx, pad_x, cfg.pad_value);
            std::fill_n(dst_b + row_idx, pad_x, cfg.pad_value);
        }

        float sy = dy * inv_ratio;
        int yy0 = static_cast<int>(sy);
        if (yy0 > src_h - 2 && src_h > 1) yy0 = src_h - 2;
        int yy1 = (yy0 + 1 < src_h) ? yy0 + 1 : yy0;
        float fy = sy - yy0;

        float wy0 = (1.0f - fy) * ns;
        float wy1 = fy * ns;

        const uint8_t* row0 = src + static_cast<size_t>(yy0) * src_w * src_ch;
        const uint8_t* row1 = src + static_cast<size_t>(yy1) * src_w * src_ch;

        size_t img_start = row_idx + pad_x;

        for (int dx = 0; dx < scaled_w; ++dx) {
            int x0 = col_x0[dx], x1 = col_x1[dx];
            float fx = col_fx[dx];
            float w0 = 1.0f - fx;
            float w1 = fx;
            size_t idx = img_start + dx;

            if (src_ch == 3) {
                const uint8_t* p00 = row0 + x0 * 3;
                const uint8_t* p10 = row0 + x1 * 3;
                const uint8_t* p01 = row1 + x0 * 3;
                const uint8_t* p11 = row1 + x1 * 3;

                dst_r[idx] = (p00[0] * w0 + p10[0] * w1) * wy0 + (p01[0] * w0 + p11[0] * w1) * wy1;
                dst_g[idx] = (p00[1] * w0 + p10[1] * w1) * wy0 + (p01[1] * w0 + p11[1] * w1) * wy1;
                dst_b[idx] = (p00[2] * w0 + p10[2] * w1) * wy0 + (p01[2] * w0 + p11[2] * w1) * wy1;
            } else if (src_ch == 1) {
                float val = (row0[x0] * w0 + row0[x1] * w1) * wy0 + (row1[x0] * w0 + row1[x1] * w1) * wy1;
                dst_r[idx] = val; dst_g[idx] = val; dst_b[idx] = val;
            } else { // 4 channels
                const uint8_t* p00 = row0 + x0 * 4;
                const uint8_t* p10 = row0 + x1 * 4;
                const uint8_t* p01 = row1 + x0 * 4;
                const uint8_t* p11 = row1 + x1 * 4;

                dst_r[idx] = (p00[0] * w0 + p10[0] * w1) * wy0 + (p01[0] * w0 + p11[0] * w1) * wy1;
                dst_g[idx] = (p00[1] * w0 + p10[1] * w1) * wy0 + (p01[1] * w0 + p11[1] * w1) * wy1;
                dst_b[idx] = (p00[2] * w0 + p10[2] * w1) * wy0 + (p01[2] * w0 + p11[2] * w1) * wy1;
            }
        }

        // Sağ dolgu
        int right_pad = W - (pad_x + scaled_w);
        if (right_pad > 0) {
            size_t r_start = row_idx + pad_x + scaled_w;
            std::fill_n(dst_r + r_start, right_pad, cfg.pad_value);
            std::fill_n(dst_g + r_start, right_pad, cfg.pad_value);
            std::fill_n(dst_b + r_start, right_pad, cfg.pad_value);
        }
    }

    // Alt Dolgu
    int bottom_y = pad_y + scaled_h;
    if (bottom_y < H) {
        size_t b_start = static_cast<size_t>(bottom_y) * W;
        size_t b_len = static_cast<size_t>(H - bottom_y) * W;
        std::fill_n(dst_r + b_start, b_len, cfg.pad_value);
        std::fill_n(dst_g + b_start, b_len, cfg.pad_value);
        std::fill_n(dst_b + b_start, b_len, cfg.pad_value);
    }
}

// ------------------------------------------------------------------
// 2. Standart Kopya API
// ------------------------------------------------------------------
PreprocessResult preprocess_letterbox(const Image& img, const PreprocessConfig& cfg) {
    PreprocessResult r;
    r.data.resize(3 * static_cast<size_t>(cfg.target_w) * cfg.target_h);
    preprocess_letterbox_direct(img, cfg, r.data.data(), r);
    return r;
}

// ------------------------------------------------------------------
// 3. Yüksek Seviye Dosya Yükleme API'leri
// ------------------------------------------------------------------
bool load_and_preprocess_direct(const std::string& path,
                                const PreprocessConfig& cfg,
                                float* out_tensor,
                                PreprocessResult& meta)
{
    try {
        Image img = load_image(path);
        preprocess_letterbox_direct(img, cfg, out_tensor, meta);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

PreprocessResult load_and_preprocess(const std::string& path, const PreprocessConfig& cfg) {
    Image img = load_image(path);
    return preprocess_letterbox(img, cfg);
}

std::vector<PreprocessResult> load_and_preprocess_batch(
    const std::vector<std::string>& paths,
    const PreprocessConfig& cfg)
{
    std::vector<PreprocessResult> results;
    results.reserve(paths.size());
    for (const auto& p : paths) {
        try {
            results.push_back(load_and_preprocess(p, cfg));
        } catch (const std::exception&) {
            results.emplace_back();
        }
    }
    return results;
}

} // namespace imageloader