#include "engine/ops/pooling.hpp"
#include "engine/runtime/thread_pool.hpp"
#include <cfloat>
#include <cstring>
#include <immintrin.h>
#include <algorithm>

namespace engine {
namespace ops {

namespace {

inline int64_t pool_out(int64_t in, int64_t k, int64_t s, int64_t p) noexcept {
    return (in + 2*p - k) / s + 1;
}

} /* anonymous namespace */

Shape maxpool2d_output_shape(const Shape& input, const Pool2dConfig& cfg) noexcept {
    if (input.rank() != 4) return Shape{};
    /* NHWC: [N, H, W, C] */
    int64_t d[4] = {
        input.dim(0),
        pool_out(input.dim(1), cfg.kernel_h, cfg.stride_h, cfg.pad_h),
        pool_out(input.dim(2), cfg.kernel_w, cfg.stride_w, cfg.pad_w),
        input.dim(3)
    };
    return Shape(d, 4);
}

Shape avgpool2d_output_shape(const Shape& input, const Pool2dConfig& cfg) noexcept {
    return maxpool2d_output_shape(input, cfg);
}

void maxpool2d(const Tensor& input, Tensor& output, const Pool2dConfig& cfg) noexcept {
    if (input.dtype() != DType::F32 || output.dtype() != DType::F32) return;
    if (input.rank() != 4 || output.rank() != 4) return;

    /* NHWC */
    const int64_t N = input.dim(0);
    const int64_t H = input.dim(1), W = input.dim(2), C = input.dim(3);
    const int64_t Hout = output.dim(1), Wout = output.dim(2);
    const int64_t KH = cfg.kernel_h, KW = cfg.kernel_w;
    const int64_t SH = cfg.stride_h, SW = cfg.stride_w;
    const int64_t PH = cfg.pad_h, PW = cfg.pad_w;

    const float* in = input.data<float>();
    float* out = output.data<float>();

    auto& pool = runtime::GlobalPool::get();
    const int64_t total_pixels = N * Hout * Wout;

    const __m256 vmin = _mm256_set1_ps(-FLT_MAX);

    pool.parallel_for(0, total_pixels, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
            const int64_t wout = t % Wout;
            const int64_t t1_  = t / Wout;
            const int64_t hout = t1_ % Hout;
            const int64_t n    = t1_ / Hout;

            const int64_t h0 = hout * SH - PH;
            const int64_t w0 = wout * SW - PW;

            float* const out_pix = out + ((n * Hout + hout) * Wout + wout) * C;

            /* AVX2: 8 kanal birlikte */
            int64_t c = 0;
            for (; c + 7 < C; c += 8) {
                __m256 best = vmin;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t h = h0 + kh;
                    if (h < 0 || h >= H) continue;
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t w = w0 + kw;
                        if (w < 0 || w >= W) continue;
                        const float* p = in + ((n * H + h) * W + w) * C + c;
                        __m256 v = _mm256_loadu_ps(p);
                        best = _mm256_max_ps(best, v);
                    }
                }
                _mm256_storeu_ps(out_pix + c, best);
            }
            /* Tail */
            for (; c < C; ++c) {
                float best = -FLT_MAX;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t h = h0 + kh;
                    if (h < 0 || h >= H) continue;
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t w = w0 + kw;
                        if (w < 0 || w >= W) continue;
                        const float v = in[((n * H + h) * W + w) * C + c];
                        if (v > best) best = v;
                    }
                }
                out_pix[c] = best;
            }
        }
    });
}

void avgpool2d(const Tensor& input, Tensor& output, const Pool2dConfig& cfg) noexcept {
    if (input.dtype() != DType::F32 || output.dtype() != DType::F32) return;
    if (input.rank() != 4 || output.rank() != 4) return;

    const int64_t N = input.dim(0);
    const int64_t H = input.dim(1), W = input.dim(2), C = input.dim(3);
    const int64_t Hout = output.dim(1), Wout = output.dim(2);
    const int64_t KH = cfg.kernel_h, KW = cfg.kernel_w;
    const int64_t SH = cfg.stride_h, SW = cfg.stride_w;
    const int64_t PH = cfg.pad_h, PW = cfg.pad_w;

    const float* in = input.data<float>();
    float* out = output.data<float>();
    const float inv_k = 1.0f / static_cast<float>(KH * KW);

    auto& pool = runtime::GlobalPool::get();
    const int64_t total_pixels = N * Hout * Wout;

    pool.parallel_for(0, total_pixels, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
            const int64_t wout = t % Wout;
            const int64_t t1_  = t / Wout;
            const int64_t hout = t1_ % Hout;
            const int64_t n    = t1_ / Hout;

            const int64_t h0 = hout * SH - PH;
            const int64_t w0 = wout * SW - PW;

            float* const out_pix = out + ((n * Hout + hout) * Wout + wout) * C;

            int64_t c = 0;
            for (; c + 7 < C; c += 8) {
                __m256 acc = _mm256_setzero_ps();
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t h = h0 + kh;
                    if (h < 0 || h >= H) continue;
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t w = w0 + kw;
                        if (w < 0 || w >= W) continue;
                        const float* p = in + ((n * H + h) * W + w) * C + c;
                        acc = _mm256_add_ps(acc, _mm256_loadu_ps(p));
                    }
                }
                const __m256 vk = _mm256_set1_ps(inv_k);
                _mm256_storeu_ps(out_pix + c, _mm256_mul_ps(acc, vk));
            }
            for (; c < C; ++c) {
                float sum = 0.0f;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t h = h0 + kh;
                    if (h < 0 || h >= H) continue;
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t w = w0 + kw;
                        if (w < 0 || w >= W) continue;
                        sum += in[((n * H + h) * W + w) * C + c];
                    }
                }
                out_pix[c] = sum * inv_k;
            }
        }
    });
}

Tensor maxpool2d(const Tensor& input, const Pool2dConfig& cfg) noexcept {
    const Shape s = maxpool2d_output_shape(input.shape(), cfg);
    if (s.rank() == 0) return Tensor{};
    Tensor out = Tensor::empty(s, DType::F32);
    if (!out.is_empty()) maxpool2d(input, out, cfg);
    return out;
}

Tensor avgpool2d(const Tensor& input, const Pool2dConfig& cfg) noexcept {
    const Shape s = avgpool2d_output_shape(input.shape(), cfg);
    if (s.rank() == 0) return Tensor{};
    Tensor out = Tensor::empty(s, DType::F32);
    if (!out.is_empty()) avgpool2d(input, out, cfg);
    return out;
}

} /* namespace ops */
} /* namespace engine */