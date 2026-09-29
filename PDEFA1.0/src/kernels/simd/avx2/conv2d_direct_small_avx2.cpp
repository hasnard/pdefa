#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include "engine/runtime/thread_pool.hpp"

namespace {

inline __m256 apply_act_avx2(__m256 v, int act, float slope) noexcept {
    if (act == 1) return _mm256_max_ps(v, _mm256_setzero_ps());
    if (act == 2) {
        __m256 zero = _mm256_setzero_ps();
        __m256 scaled = _mm256_mul_ps(v, _mm256_set1_ps(slope));
        return _mm256_blendv_ps(scaled, v, _mm256_cmp_ps(v, zero, _CMP_GT_OS));
    }
    if (act != 3 && act != 4 && act != 5) return v;

    alignas(32) float buf[8];
    _mm256_store_ps(buf, v);
    for (int i = 0; i < 8; ++i) {
        if (act == 3) {
            buf[i] = buf[i] / (1.0f + std::exp(-buf[i]));
        } else if (act == 4) {
            const float inner = 0.7978845608f * (buf[i] + 0.044715f * buf[i] * buf[i] * buf[i]);
            buf[i] = 0.5f * buf[i] * (1.0f + std::tanh(inner));
        } else if (act == 5) {                                          // ← YENİ SiLU
            buf[i] = buf[i] / (1.0f + std::exp(-buf[i]));
        }
    }
    return _mm256_load_ps(buf);
}

inline __m256 load_strided2(const float* row, int64_t w_start) noexcept {
    __m256 v0 = _mm256_loadu_ps(row + w_start);
    __m256 v1 = _mm256_loadu_ps(row + w_start + 8);
    __m256 even = _mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(2, 0, 2, 0));
    const __m256i perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    return _mm256_permutevar8x32_ps(even, perm);
}

} // namespace

extern "C" void engine_apply_bias_act_avx2(
    float* output, const float* bias, int64_t Cout, int64_t HW,
    int act, float slope) noexcept
{
    auto& pool = engine::runtime::GlobalPool::get();
    pool.parallel_for(0, Cout, [&](int64_t c0, int64_t c1) {
        for (int64_t c = c0; c < c1; ++c) {
            float* row = output + c * HW;
            __m256 b_vec = bias ? _mm256_set1_ps(bias[c]) : _mm256_setzero_ps();
            int64_t hw = 0;
            for (; hw + 7 < HW; hw += 8) {
                __m256 v = _mm256_loadu_ps(row + hw);
                if (bias) v = _mm256_add_ps(v, b_vec);
                if (act > 0) v = apply_act_avx2(v, act, slope);
                _mm256_storeu_ps(row + hw, v);
            }
            for (; hw < HW; ++hw) {
                float v = row[hw];
                if (bias) v += bias[c];
                if (act == 1)      v = std::max(v, 0.0f);
                else if (act == 2) v = v > 0.0f ? v : v * slope;
                else if (act == 3) v = v / (1.0f + std::exp(-v));
                else if (act == 4) {
                    const float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
                    v = 0.5f * v * (1.0f + std::tanh(inner));
                }
                row[hw] = v;
            }
        }
    });
}

extern "C" int engine_conv2d_direct_small_avx2(
    const float* input, const float* weight, const float* bias,
    float* output, int64_t N, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout, int64_t KH, int64_t KW,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int fused_activation, float leaky_slope) noexcept
{
    if (stride_h < 1 || stride_h > 2 || stride_w < 1 || stride_w > 2) return 0;

    const int64_t Hout = (H + 2 * pad_h - KH) / stride_h + 1;
    const int64_t Wout = (W + 2 * pad_w - KW) / stride_w + 1;
    if (Hout <= 0 || Wout <= 0) return 0;

    auto& pool = engine::runtime::GlobalPool::get();
    const int64_t HW = Hout * Wout;
    const int64_t co_blocks = (Cout + 7) / 8;

    const int64_t total_tasks = N * co_blocks * Hout;

    pool.parallel_for(0, total_tasks, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
            const int64_t ho = t % Hout;
            const int64_t cb = (t / Hout) % co_blocks;
            const int64_t n  = (t / Hout) / co_blocks;

            const int64_t co_start = cb * 8;
            const int64_t co_count = std::min<int64_t>(8, Cout - co_start);
            const int64_t h_base   = ho * stride_h - pad_h;

            const float* in_n  = input  + n * Cin * H * W;
            float*       out_n = output + n * Cout * HW;

            for (int64_t wo = 0; wo < Wout; wo += 8) {
                const int64_t w_len  = std::min<int64_t>(8, Wout - wo);
                const int64_t w_base = wo * stride_w - pad_w;

                __m256 acc[8];
                for (int c = 0; c < 8; ++c) {
                    acc[c] = (c < co_count && bias) ? _mm256_set1_ps(bias[co_start + c]) : _mm256_setzero_ps();
                }

                for (int64_t ci = 0; ci < Cin; ++ci) {
                    const float* in_c = in_n + ci * H * W;

                    for (int64_t kh = 0; kh < KH; ++kh) {
                        const int64_t hi = h_base + kh;
                        if (hi < 0 || hi >= H) continue;
                        const float* in_row = in_c + hi * W;

                        for (int64_t kw = 0; kw < KW; ++kw) {
                            const int64_t wi = w_base + kw;
                            __m256 v;

                            const bool safe = (wi >= 0) && (wi + (stride_w == 2 ? 14 : 7) < W);
                            if (safe && w_len == 8) {
                                v = (stride_w == 1) ? _mm256_loadu_ps(in_row + wi) : load_strided2(in_row, wi);
                            } else {
                                alignas(32) float vbuf[8] = {0};
                                for (int i = 0; i < w_len; ++i) {
                                    const int64_t xi = wi + i * stride_w;
                                    if (xi >= 0 && xi < W) vbuf[i] = in_row[xi];
                                }
                                v = _mm256_load_ps(vbuf);
                            }

                            for (int c = 0; c < co_count; ++c) {
                                const float wv = weight[((co_start + c) * Cin + ci) * KH * KW + kh * KW + kw];
                                acc[c] = _mm256_fmadd_ps(_mm256_set1_ps(wv), v, acc[c]);
                            }
                        }
                    }
                }

                for (int c = 0; c < co_count; ++c) {
                    __m256 v_out = apply_act_avx2(acc[c], fused_activation, leaky_slope);
                    float* dst = out_n + (co_start + c) * HW + ho * Wout + wo;
                    if (w_len == 8) {
                        _mm256_storeu_ps(dst, v_out);
                    } else {
                        alignas(32) float tmp[8];
                        _mm256_store_ps(tmp, v_out);
                        for (int i = 0; i < w_len; ++i) dst[i] = tmp[i];
                    }
                }
            }
        }
    });

    return 1;
}