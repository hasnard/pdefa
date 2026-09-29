/* ============================================================================
 *  src/ops/upsample.cpp
 *  NHWC DÜZENİ:
 *    Input  : [N, H, W, C]
 *    Output : [N, Hout, Wout, C]
 * ============================================================================ */

#include "engine/ops/upsample.hpp"
#include "engine/runtime/thread_pool.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace engine {
namespace ops {

namespace {

struct XInterp {
    int64_t x0c;
    int64_t x1c;
    float dx;
};

void upsample_nearest_nhwc(const Tensor& input, Tensor& output,
                           int64_t scale_h, int64_t scale_w) noexcept
{
    const int64_t N = input.dim(0);
    const int64_t H = input.dim(1);
    const int64_t W = input.dim(2);
    const int64_t C = input.dim(3);

    const int64_t Hout = output.dim(1);
    const int64_t Wout = output.dim(2);

    const float* in = input.data<float>();
    float* out = output.data<float>();

    auto& pool = runtime::GlobalPool::get();
    const int64_t total_rows = N * Hout;
    const size_t c_bytes = static_cast<size_t>(C) * sizeof(float);

    pool.parallel_for(0, total_rows, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t n    = r / Hout;
            const int64_t hout = r % Hout;
            const int64_t hin  = hout / scale_h;

            const float* in_row = in + (n * H + hin) * W * C;
            float* out_row      = out + (n * Hout + hout) * Wout * C;

            for (int64_t wout = 0; wout < Wout; ++wout) {
                const int64_t win = wout / scale_w;
                std::memcpy(out_row + wout * C, in_row + win * C, c_bytes);
            }
        }
    });
}

void upsample_bilinear_nhwc(const Tensor& input, Tensor& output) noexcept
{
    const int64_t N = input.dim(0);
    const int64_t H = input.dim(1);
    const int64_t W = input.dim(2);
    const int64_t C = input.dim(3);

    const int64_t Hout = output.dim(1);
    const int64_t Wout = output.dim(2);

    const float* in = input.data<float>();
    float* out = output.data<float>();

    const float sh = static_cast<float>(H) / static_cast<float>(Hout);
    const float sw = static_cast<float>(W) / static_cast<float>(Wout);

    // Yatay koordinat ve ağırlıkları döngü öncesi önbelleğe al (Pre-caching)
    std::vector<XInterp> x_lut(Wout);
    for (int64_t wout = 0; wout < Wout; ++wout) {
        const float fx = (static_cast<float>(wout) + 0.5f) * sw - 0.5f;
        const int64_t x0 = static_cast<int64_t>(std::floor(fx));
        x_lut[wout].x0c = std::max<int64_t>(0, std::min<int64_t>(x0, W - 1));
        x_lut[wout].x1c = std::max<int64_t>(0, std::min<int64_t>(x0 + 1, W - 1));
        x_lut[wout].dx  = fx - static_cast<float>(x0);
    }

    auto& pool = runtime::GlobalPool::get();
    const int64_t total_rows = N * Hout;

    pool.parallel_for(0, total_rows, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t n    = r / Hout;
            const int64_t hout = r % Hout;

            const float fy = (static_cast<float>(hout) + 0.5f) * sh - 0.5f;
            const int64_t y0 = static_cast<int64_t>(std::floor(fy));
            const float dy = fy - static_cast<float>(y0);

            const int64_t y0c = std::max<int64_t>(0, std::min<int64_t>(y0, H - 1));
            const int64_t y1c = std::max<int64_t>(0, std::min<int64_t>(y0 + 1, H - 1));

            const float* const in_batch = in + (n * H) * W * C;
            float* const out_row = out + (n * Hout + hout) * Wout * C;

            for (int64_t wout = 0; wout < Wout; ++wout) {
                const int64_t x0c = x_lut[wout].x0c;
                const int64_t x1c = x_lut[wout].x1c;
                const float dx    = x_lut[wout].dx;

                const float w00 = (1.0f - dx) * (1.0f - dy);
                const float w01 = dx * (1.0f - dy);
                const float w10 = (1.0f - dx) * dy;
                const float w11 = dx * dy;

                const float* const p00 = in_batch + (y0c * W + x0c) * C;
                const float* const p01 = in_batch + (y0c * W + x1c) * C;
                const float* const p10 = in_batch + (y1c * W + x0c) * C;
                const float* const p11 = in_batch + (y1c * W + x1c) * C;
                float* const dst = out_row + wout * C;

                const __m256 vw00 = _mm256_set1_ps(w00);
                const __m256 vw01 = _mm256_set1_ps(w01);
                const __m256 vw10 = _mm256_set1_ps(w10);
                const __m256 vw11 = _mm256_set1_ps(w11);

                int64_t c = 0;
                // NHWC'de kanal ekseni bitişik olduğu için doğrudan AVX2 FMA çalışır
                for (; c + 7 < C; c += 8) {
                    const __m256 v00 = _mm256_loadu_ps(p00 + c);
                    const __m256 v01 = _mm256_loadu_ps(p01 + c);
                    const __m256 v10 = _mm256_loadu_ps(p10 + c);
                    const __m256 v11 = _mm256_loadu_ps(p11 + c);

                    __m256 res = _mm256_mul_ps(v00, vw00);
                    res = _mm256_fmadd_ps(v01, vw01, res);
                    res = _mm256_fmadd_ps(v10, vw10, res);
                    res = _mm256_fmadd_ps(v11, vw11, res);

                    _mm256_storeu_ps(dst + c, res);
                }

                // Skaler artıklar
                for (; c < C; ++c) {
                    dst[c] = p00[c] * w00 + p01[c] * w01 + p10[c] * w10 + p11[c] * w11;
                }
            }
        }
    });
}

} /* anonymous namespace */

void upsample(const Tensor& input, Tensor& output,
              int64_t scale_h, int64_t scale_w,
              UpsampleMode mode) noexcept
{
    if (input.dtype() != DType::F32 || output.dtype() != DType::F32) return;
    if (input.rank() != 4 || output.rank() != 4) return;
    if (scale_h <= 0 || scale_w <= 0) return;

    if (mode == UpsampleMode::Nearest) {
        upsample_nearest_nhwc(input, output, scale_h, scale_w);
    } else {
        upsample_bilinear_nhwc(input, output);
    }
}

Tensor upsample(const Tensor& input, int64_t scale_h, int64_t scale_w,
                UpsampleMode mode) noexcept
{
    if (input.rank() != 4) return Tensor{};

    // NHWC: [N, H * scale_h, W * scale_w, C]
    int64_t dims[4] = {
        input.dim(0),
        input.dim(1) * scale_h,
        input.dim(2) * scale_w,
        input.dim(3)
    };
    Tensor out = Tensor::empty(Shape(dims, 4), DType::F32);
    if (out.is_empty()) return out;

    upsample(input, out, scale_h, scale_w, mode);
    return out;
}

Tensor resize(const Tensor& input, int64_t target_h, int64_t target_w,
              UpsampleMode mode) noexcept
{
    if (input.rank() != 4) return Tensor{};
    if (target_h <= 0 || target_w <= 0) return Tensor{};

    const int64_t H = input.dim(1);
    const int64_t W = input.dim(2);
    const int64_t C = input.dim(3);

    // NHWC: [N, target_h, target_w, C]
    int64_t dims[4] = { input.dim(0), target_h, target_w, C };
    Tensor out = Tensor::empty(Shape(dims, 4), DType::F32);
    if (out.is_empty()) return out;

    if (mode == UpsampleMode::Nearest && target_h % H == 0 && target_w % W == 0) {
        upsample(input, out, target_h / H, target_w / W, mode);
    } else if (mode == UpsampleMode::Nearest) {
        const float sh = static_cast<float>(H) / static_cast<float>(target_h);
        const float sw = static_cast<float>(W) / static_cast<float>(target_w);
        const int64_t N = input.dim(0);
        const float* in = input.data<float>();
        float* op = out.data<float>();

        std::vector<int64_t> x_in_lut(target_w);
        for (int64_t x = 0; x < target_w; ++x) {
            x_in_lut[x] = static_cast<int64_t>(x * sw);
        }

        const size_t c_bytes = static_cast<size_t>(C) * sizeof(float);
        auto& pool = runtime::GlobalPool::get();

        pool.parallel_for(0, N * target_h, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r) {
                const int64_t n = r / target_h;
                const int64_t y = r % target_h;
                const int64_t yi = static_cast<int64_t>(y * sh);

                const float* in_row = in + (n * H + yi) * W * C;
                float* out_row = op + (n * target_h + y) * target_w * C;

                for (int64_t x = 0; x < target_w; ++x) {
                    std::memcpy(out_row + x * C, in_row + x_in_lut[x] * C, c_bytes);
                }
            }
        });
    } else {
        upsample_bilinear_nhwc(input, out);
    }
    return out;
}

} /* namespace ops */
} /* namespace engine */