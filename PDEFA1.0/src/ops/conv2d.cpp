/* ============================================================================
 *  src/ops/conv2d.cpp  —  NHWC Dispatcher (Final Optimized Engine)
 * ========================================================================== */

#include "engine/ops/conv2d.hpp"
#include "engine/ops/matmul.hpp"
#include "engine/ops/activation.hpp"
#include "engine/kernels/kernel_registry.hpp"
#include "engine/runtime/thread_pool.hpp"
#include "engine/runtime/cpu_info.hpp"
#include "engine/memory/arena.hpp"

#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <new>

/* Conv dispatcher'ın seçtiği yolu dışa bildir (benchmark ve debug için).
 * Anonim namespace DIŞINDA tanımlanır, benchmark_conv2d.cpp bunu okur. */
thread_local const char* engine_conv_last_path = "none";

extern "C" {
    int engine_conv2d_gemm(
        const float* input, const float* weight, const float* bias, float* output,
        int64_t N_batch, int64_t Cin, int64_t H, int64_t W,
        int64_t Cout, int64_t KH, int64_t KW,
        int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w, int fused_activation, float leaky_slope) noexcept;

    int engine_conv2d_winograd_3x3(
        const float* input, const float* weight, const float* bias, float* output,
        int64_t N, int64_t Cin, int64_t H, int64_t W,
        int64_t Cout, int64_t pad_h, int64_t pad_w,
        int fused_activation, float leaky_slope) noexcept;

    int engine_conv2d_depthwise_avx2(
        const float* input, const float* weight, const float* bias, float* output,
        int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW,
        int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w, int fused_activation, float leaky_slope) noexcept;

    int engine_conv2d_grouped_gemm(
        const float* input, const float* weight, const float* bias, float* output,
        int64_t N, int64_t Cin, int64_t H, int64_t W,
        int64_t Cout, int64_t KH, int64_t KW, int64_t groups,
        int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w, int fused_activation, float leaky_slope) noexcept;

    int engine_conv2d_direct_nhwc(
        const float* input, const float* weight, const float* bias, float* output,
        int64_t N, int64_t Cin, int64_t H, int64_t W,
        int64_t Cout, int64_t KH, int64_t KW,
        int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w, int fused_activation, float leaky_slope) noexcept;
}

namespace {

/* ============================================================================
 *  HIZLI AVX2 MATEMATİK — Vektörel exp/sigmoid/silu/tanh
 * ========================================================================== */
inline __m256 fast_exp_avx2(__m256 x) noexcept {
    const __m256 kHiBound = _mm256_set1_ps(88.3762626647949f);
    const __m256 kLoBound = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, kHiBound);
    x = _mm256_max_ps(x, kLoBound);

    const __m256 kLog2e = _mm256_set1_ps(1.44269504088896341f);
    __m256 fx = _mm256_mul_ps(x, kLog2e);
    __m256 n  = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    const __m256 kC1 = _mm256_set1_ps(0.693359375f);
    const __m256 kC2 = _mm256_set1_ps(-2.121944400546905827679e-4f);
    __m256 y = _mm256_fnmadd_ps(n, kC1, x);
    y = _mm256_fnmadd_ps(n, kC2, y);

    const __m256 c0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 c1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 c2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 c3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 c4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 c5 = _mm256_set1_ps(5.0000001201e-1f);

    __m256 p = _mm256_fmadd_ps(c0, y, c1);
    p = _mm256_fmadd_ps(p, y, c2);
    p = _mm256_fmadd_ps(p, y, c3);
    p = _mm256_fmadd_ps(p, y, c4);
    p = _mm256_fmadd_ps(p, y, c5);

    __m256 y2 = _mm256_mul_ps(y, y);
    p = _mm256_fmadd_ps(p, y2, y);
    p = _mm256_add_ps(p, _mm256_set1_ps(1.0f));

    __m256i n_int = _mm256_cvtps_epi32(n);
    __m256 scale = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(n_int, _mm256_set1_epi32(127)), 23));

    return _mm256_mul_ps(p, scale);
}

inline __m256 fast_rcp_avx2(__m256 d) noexcept {
    const __m256 two = _mm256_set1_ps(2.0f);
    __m256 rcp = _mm256_rcp_ps(d);
    return _mm256_mul_ps(rcp, _mm256_fnmadd_ps(d, rcp, two));
}

inline __m256 fast_sigmoid_avx2(__m256 x) noexcept {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 e = fast_exp_avx2(neg_x);
    return fast_rcp_avx2(_mm256_add_ps(one, e));
}

inline __m256 fast_silu_avx2(__m256 x) noexcept {
    return _mm256_mul_ps(x, fast_sigmoid_avx2(x));
}

inline __m256 fast_tanh_avx2(__m256 x) noexcept {
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_sub_ps(
        _mm256_mul_ps(two, fast_sigmoid_avx2(_mm256_mul_ps(two, x))),
        one);
}

/* ============================================================================
 *  SCALAR YARDIMCILAR
 * ========================================================================== */
inline float apply_act_local(float v, int fused_activation, float slope) noexcept {
    switch (fused_activation) {
        case 1: return v > 0.f ? v : 0.f;
        case 2: return v > 0.f ? v : v * slope;
        case 3: return 1.f / (1.f + std::exp(-v));
        case 4: return std::tanh(v);
        case 5: return v / (1.f + std::exp(-v));
        default: return v;
    }
}

inline __m256 apply_act_avx2(__m256 acc, int fused_activation,
                              __m256 vslope, __m256 vzero) noexcept {
    switch (fused_activation) {
        case 1:
            return _mm256_max_ps(acc, vzero);
        case 2: {
            __m256 pos = _mm256_max_ps(acc, vzero);
            __m256 neg = _mm256_min_ps(acc, vzero);
            return _mm256_fmadd_ps(neg, vslope, pos);
        }
        case 3: return fast_sigmoid_avx2(acc);
        case 4: return fast_tanh_avx2(acc);
        case 5: return fast_silu_avx2(acc);
        default: return acc;
    }
}

} // anonymous

/* ============================================================================
 *  DEPTHWISE (NHWC)
 * ========================================================================== */
extern "C" int engine_conv2d_depthwise_avx2(
    const float* input, const float* weight, const float* bias, float* output,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w, int fused_activation, float leaky_slope) noexcept
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return 0;
    const int64_t ekh = (KH - 1) * dil_h + 1;
    const int64_t ekw = (KW - 1) * dil_w + 1;
    const int64_t Hout = (H + 2 * pad_h - ekh) / stride_h + 1;
    const int64_t Wout = (W + 2 * pad_w - ekw) / stride_w + 1;
    if (Hout <= 0 || Wout <= 0) return 0;

    const float slope = (fused_activation == 2) ? leaky_slope : 0.0f;
    const __m256 vzero  = _mm256_setzero_ps();
    const __m256 vslope = _mm256_set1_ps(slope);

    const int64_t K_total = KH * KW;
    const int64_t c_blocks = (C + 7) / 8;

    /* PARALELLEŞTİRME: (N, Hout, C-block) uzayında görev dağıtımı.
     * Önceden tek-thread'di — EfficientNet/MobileNetV3'ün en büyük
     * performans darboğazı buydu. */
    auto& pool = engine::runtime::GlobalPool::get();
    const int64_t total_tasks = N * Hout * c_blocks;

    pool.parallel_for(0, total_tasks, [&](int64_t t0, int64_t t1) {
        /* Thread-local weight buffer (K_total=9 için 9 register, stack) */
        alignas(32) __m256 w_vec_stack[64];
        __m256* w_vec = (K_total <= 64) ? w_vec_stack : nullptr;
        std::vector<__m256> w_heap;
        if (!w_vec) {
            w_heap.resize(static_cast<size_t>(K_total));
            w_vec = w_heap.data();
        }

        for (int64_t t = t0; t < t1; ++t) {
            const int64_t cb  = t % c_blocks;
            const int64_t tmp = t / c_blocks;
            const int64_t oh  = tmp % Hout;
            const int64_t n   = tmp / Hout;
            const int64_t c   = cb * 8;
            const int64_t h_base = oh * stride_h - pad_h;

            if (c + 8 <= C) {
                /* AVX2 yol: 8 kanal bir seferde */
                for (int64_t kh = 0; kh < KH; ++kh) {
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t tap = kh * KW + kw;
                        w_vec[tap] = _mm256_set_ps(
                            weight[(c + 7) * K_total + tap],
                            weight[(c + 6) * K_total + tap],
                            weight[(c + 5) * K_total + tap],
                            weight[(c + 4) * K_total + tap],
                            weight[(c + 3) * K_total + tap],
                            weight[(c + 2) * K_total + tap],
                            weight[(c + 1) * K_total + tap],
                            weight[(c + 0) * K_total + tap]);
                    }
                }

                const __m256 vb = bias ? _mm256_loadu_ps(bias + c) : vzero;

                /* ============================================================
                 *  Border peeling: sadece sol/sağ kenar branch'li, iç bölge
                 *  tamamen branchless + 4-wide ow unroll (FMA latency hide).
                 * ============================================================ */
                const int64_t w_last_safe =
                    W - 1 - (KW - 1) * dil_w - (stride_w - 1);
                const int64_t ow_safe_end =
                    (w_last_safe >= 0)
                        ? std::min<int64_t>(Wout, (w_last_safe + stride_w) / stride_w)
                        : 0;

                int64_t ow = 0;

                /* --- Sol kenar (nadir) --- */
                const int64_t ow_left =
                    (pad_w + stride_w - 1) / stride_w;
                for (; ow < std::min<int64_t>(Wout, ow_left); ++ow) {
                    __m256 acc = vb;
                    const int64_t w_base = ow * stride_w - pad_w;
                    for (int64_t kh = 0; kh < KH; ++kh) {
                        const int64_t ih = h_base + kh * dil_h;
                        if (ih < 0 || ih >= H) continue;
                        for (int64_t kw = 0; kw < KW; ++kw) {
                            const int64_t iw = w_base + kw * dil_w;
                            if (iw < 0 || iw >= W) continue;
                            const float* in_ptr = input + ((n * H + ih) * W + iw) * C + c;
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(in_ptr),
                                                  w_vec[kh * KW + kw], acc);
                        }
                    }
                    acc = apply_act_avx2(acc, fused_activation, vslope, vzero);
                    _mm256_storeu_ps(output + ((n * Hout + oh) * Wout + ow) * C + c, acc);
                }

                /* --- İç bölge: 4-wide unroll (stride=1 için), branchless --- */
                if (stride_w == 1) {
                    for (; ow + 3 < ow_safe_end; ow += 4) {
                        __m256 acc0 = vb, acc1 = vb, acc2 = vb, acc3 = vb;
                        for (int64_t kh = 0; kh < KH; ++kh) {
                            const int64_t ih = h_base + kh * dil_h;
                            if (ih < 0 || ih >= H) continue;
                            const float* row = input
                                + ((n * H + ih) * W + ow - pad_w) * C + c;
                            for (int64_t kw = 0; kw < KW; ++kw) {
                                const __m256 wv = w_vec[kh * KW + kw];
                                const float* p = row + kw * dil_w * C;
                                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(p),         wv, acc0);
                                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + C),     wv, acc1);
                                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 2 * C), wv, acc2);
                                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 3 * C), wv, acc3);
                            }
                        }
                        float* out = output + ((n * Hout + oh) * Wout + ow) * C + c;
                        _mm256_storeu_ps(out,         apply_act_avx2(acc0, fused_activation, vslope, vzero));
                        _mm256_storeu_ps(out + C,     apply_act_avx2(acc1, fused_activation, vslope, vzero));
                        _mm256_storeu_ps(out + 2 * C, apply_act_avx2(acc2, fused_activation, vslope, vzero));
                        _mm256_storeu_ps(out + 3 * C, apply_act_avx2(acc3, fused_activation, vslope, vzero));
                    }
                }

                /* --- İç bölge kalan + sağ kenar (branch'li) --- */
                for (; ow < Wout; ++ow) {
                    __m256 acc = vb;
                    const int64_t w_base = ow * stride_w - pad_w;

                    for (int64_t kh = 0; kh < KH; ++kh) {
                        const int64_t ih = h_base + kh * dil_h;
                        if (ih < 0 || ih >= H) continue;

                        for (int64_t kw = 0; kw < KW; ++kw) {
                            const int64_t iw = w_base + kw * dil_w;
                            if (iw < 0 || iw >= W) continue;

                            const float* in_ptr = input + ((n * H + ih) * W + iw) * C + c;
                            const __m256 iv = _mm256_loadu_ps(in_ptr);
                            acc = _mm256_fmadd_ps(iv, w_vec[kh * KW + kw], acc);
                        }
                    }

                    acc = apply_act_avx2(acc, fused_activation, vslope, vzero);
                    float* out_ptr = output + ((n * Hout + oh) * Wout + ow) * C + c;
                    _mm256_storeu_ps(out_ptr, acc);
                }
            } else {
                /* Scalar tail: son 1-7 kanal */
                for (int64_t cc = c; cc < C; ++cc) {
                    const float b = bias ? bias[cc] : 0.0f;
                    const float* w_c = weight + cc * K_total;
                    for (int64_t ow = 0; ow < Wout; ++ow) {
                        float acc = b;
                        const int64_t w_base = ow * stride_w - pad_w;
                        for (int64_t kh = 0; kh < KH; ++kh) {
                            const int64_t ih = h_base + kh * dil_h;
                            if (ih < 0 || ih >= H) continue;
                            for (int64_t kw = 0; kw < KW; ++kw) {
                                const int64_t iw = w_base + kw * dil_w;
                                if (iw < 0 || iw >= W) continue;
                                acc += input[((n * H + ih) * W + iw) * C + cc]
                                     * w_c[kh * KW + kw];
                            }
                        }
                        output[((n * Hout + oh) * Wout + ow) * C + cc] =
                            apply_act_local(acc, fused_activation, slope);
                    }
                }
            }
        }
    });

    return 1;
}

/* ============================================================================
 *  GROUPED (NHWC)
 * ========================================================================== */
extern "C" int engine_conv2d_grouped_gemm(
    const float* input, const float* weight, const float* bias, float* output,
    int64_t N, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout, int64_t KH, int64_t KW, int64_t groups,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w, int fused_activation, float leaky_slope) noexcept
{
    if (groups <= 1) return 0;
    if (Cin % groups != 0) return 0;
    if (Cout % groups != 0) return 0;

    const int64_t Cin_g  = Cin / groups;
    const int64_t Cout_g = Cout / groups;
    const int64_t ekh = (KH - 1) * dil_h + 1;
    const int64_t ekw = (KW - 1) * dil_w + 1;
    const int64_t Hout = (H + 2 * pad_h - ekh) / stride_h + 1;
    const int64_t Wout = (W + 2 * pad_w - ekw) / stride_w + 1;
    if (Hout <= 0 || Wout <= 0) return 0;

    const size_t in_g_size  = static_cast<size_t>(H * W * Cin_g);
    const size_t out_g_size = static_cast<size_t>(Hout * Wout * Cout_g);

    float* in_buf  = new (std::nothrow) float[in_g_size];
    float* out_buf = new (std::nothrow) float[out_g_size];
    if (!in_buf || !out_buf) { delete[] in_buf; delete[] out_buf; return 0; }

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < groups; ++g) {
            for (int64_t i = 0; i < H * W; ++i) {
                std::memcpy(in_buf + i * Cin_g,
                            input + (n * H * W + i) * Cin + g * Cin_g,
                            Cin_g * sizeof(float));
            }
            const float* w_g = weight + (g * Cout_g) * Cin_g * KH * KW;
            const float* b_g = bias ? (bias + g * Cout_g) : nullptr;
            const int rc = engine_conv2d_gemm(
                in_buf, w_g, b_g, out_buf,
                1, Cin_g, H, W, Cout_g, KH, KW,
                stride_h, stride_w, pad_h, pad_w,
                dil_h, dil_w, fused_activation, leaky_slope);
            if (!rc) { delete[] in_buf; delete[] out_buf; return 0; }
            for (int64_t i = 0; i < Hout * Wout; ++i) {
                std::memcpy(output + (n * Hout * Wout + i) * Cout + g * Cout_g,
                            out_buf + i * Cout_g,
                            Cout_g * sizeof(float));
            }
        }
    }
    delete[] in_buf;
    delete[] out_buf;
    return 1;
}

namespace engine {
namespace ops {

namespace {

/* NHWC naive fallback */
void conv2d_naive_nhwc(
    const float* input, const float* weight, const float* bias, float* output,
    int64_t N, int64_t H, int64_t W, int64_t Cin,
    int64_t Cout, int64_t KH, int64_t KW,
    int64_t Hout, int64_t Wout, int64_t groups,
    int64_t sh, int64_t sw, int64_t ph, int64_t pw,
    int64_t dh, int64_t dw,
    int fused_activation, float leaky_slope) noexcept
{
    const float slope = (fused_activation == 2) ? leaky_slope : 0.0f;
    const int64_t Cin_g  = Cin / groups;
    const int64_t Cout_g = Cout / groups;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oh = 0; oh < Hout; ++oh) {
            for (int64_t ow = 0; ow < Wout; ++ow) {
                for (int64_t co = 0; co < Cout; ++co) {
                    const int64_t g = co / Cout_g;
                    float acc = bias ? bias[co] : 0.0f;
                    for (int64_t kh = 0; kh < KH; ++kh) {
                        const int64_t ih = oh * sh - ph + kh * dh;
                        if (ih < 0 || ih >= H) continue;
                        for (int64_t kw = 0; kw < KW; ++kw) {
                            const int64_t iw = ow * sw - pw + kw * dw;
                            if (iw < 0 || iw >= W) continue;
                            const float* in_ptr = input + ((n * H + ih) * W + iw) * Cin + g * Cin_g;
                            const float* w_ptr  = weight + ((co * Cin_g) * KH + kh) * KW + kw;
                            for (int64_t ci = 0; ci < Cin_g; ++ci) {
                                acc += in_ptr[ci] * w_ptr[ci * KH * KW];
                            }
                        }
                    }
                    output[((n * Hout + oh) * Wout + ow) * Cout + co] =
                        apply_act_local(acc, fused_activation, slope);
                }
            }
        }
    }
}

} // anonymous

int64_t conv_output_size(int64_t in_size, int64_t kernel, int64_t stride,
                         int64_t pad, int64_t dilation) noexcept {
    const int64_t effective_k = (kernel - 1) * dilation + 1;
    return (in_size + 2 * pad - effective_k) / stride + 1;
}

Shape conv2d_output_shape(const Shape& input, const Shape& weight,
                          const Conv2dConfig& cfg) noexcept
{
    if (input.rank() != 4 || weight.rank() != 4) return Shape{};

    const int64_t N   = input.dim(0);
    const int64_t H   = input.dim(1);
    const int64_t W   = input.dim(2);
    const int64_t Cin = input.dim(3);

    const int64_t Cout  = weight.dim(0);
    const int64_t Cin_w = weight.dim(1);
    const int64_t KH    = weight.dim(2);
    const int64_t KW    = weight.dim(3);

    const int64_t groups = (cfg.groups > 0) ? cfg.groups : 1;

    /* DFL HEAD özel durumu — transposed 1×1 */
    if (KH == 1 && KW == 1 && groups == 1 &&
        input.dim(1) == Cin_w && Cin_w != Cin) {
        int64_t dims[4] = {N, Cout, input.dim(2), input.dim(3)};
        return Shape(dims, 4);
    }

    if (Cout % groups != 0) return Shape{};
    if (Cin != Cin_w * groups) return Shape{};

    const int64_t Hout = conv_output_size(H, KH, cfg.stride_h, cfg.pad_h, cfg.dil_h);
    const int64_t Wout = conv_output_size(W, KW, cfg.stride_w, cfg.pad_w, cfg.dil_w);
    if (Hout <= 0 || Wout <= 0) return Shape{};

    int64_t dims[4] = {N, Hout, Wout, Cout};
    return Shape(dims, 4);
}

void conv2d(const Tensor& input, const Tensor& weight, const Tensor* bias,
            Tensor& output, const Conv2dConfig& cfg) noexcept
{
    engine_conv_last_path = "none";

    if (input.dtype() != DType::F32 || weight.dtype() != DType::F32 ||
        output.dtype() != DType::F32) return;
    if (input.rank() != 4 || weight.rank() != 4 || output.rank() != 4) return;

    const int64_t N    = input.dim(0);
    const int64_t H    = input.dim(1);
    const int64_t W    = input.dim(2);
    const int64_t Cin  = input.dim(3);

    const int64_t Cout = weight.dim(0);
    const int64_t KH   = weight.dim(2);
    const int64_t KW   = weight.dim(3);

    const int64_t Hout = output.dim(1);
    const int64_t Wout = output.dim(2);

    const int64_t groups = (cfg.groups > 0) ? cfg.groups : 1;

    if (Cin <= 0 || Cout <= 0 || Hout <= 0 || Wout <= 0) return;

    const int fused_act = static_cast<int>(cfg.activation);
    const float* bp = (bias && !bias->is_empty()) ? bias->data<float>() : nullptr;
    const float* ip = input.data<float>();
    const float* wp = weight.data<float>();
    float*       op = output.data<float>();

    /* ==========================================================
     *  0) DFL HEAD — transposed 1×1 conv — AVX2
     * ========================================================== */
    if (KH == 1 && KW == 1 && groups == 1 &&
        input.dim(1) == weight.dim(1) && weight.dim(1) != Cin) {

        engine_conv_last_path = "dfl";

        const int64_t Cin_alt = input.dim(1);
        const int64_t HW_t    = input.numel() / Cin_alt;

        const __m256 vzero  = _mm256_setzero_ps();
        const __m256 vslope = _mm256_set1_ps(cfg.leaky_slope);

        auto& pool = runtime::GlobalPool::get();
        constexpr int64_t T_BLK = 8;
        const int64_t n_blk = (HW_t + T_BLK - 1) / T_BLK;

        pool.parallel_for(0, n_blk, [&](int64_t b0, int64_t b1) {
            for (int64_t b = b0; b < b1; ++b) {
                const int64_t t = b * T_BLK;
                const int64_t valid = std::min<int64_t>(T_BLK, HW_t - t);

                for (int64_t co = 0; co < Cout; ++co) {
                    __m256 acc = bp ? _mm256_set1_ps(bp[co]) : vzero;
                    const float* __restrict w_row = wp + co * Cin_alt;

                    if (valid == T_BLK) {
                        for (int64_t ci = 0; ci < Cin_alt; ++ci) {
                            const __m256 iv = _mm256_loadu_ps(ip + ci * HW_t + t);
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(w_row[ci]), iv, acc);
                        }
                    } else {
                        for (int64_t ci = 0; ci < Cin_alt; ++ci) {
                            alignas(32) float tmp[8] = {0,0,0,0,0,0,0,0};
                            const float* src = ip + ci * HW_t + t;
                            for (int64_t k = 0; k < valid; ++k) tmp[k] = src[k];
                            const __m256 iv = _mm256_load_ps(tmp);
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(w_row[ci]), iv, acc);
                        }
                    }

                    acc = apply_act_avx2(acc, fused_act, vslope, vzero);

                    float* __restrict out_ptr = op + co * HW_t + t;
                    if (valid == T_BLK) {
                        _mm256_storeu_ps(out_ptr, acc);
                    } else {
                        alignas(32) float tmp[8];
                        _mm256_store_ps(tmp, acc);
                        for (int64_t k = 0; k < valid; ++k) out_ptr[k] = tmp[k];
                    }
                }
            }
        });
        return;
    }

    /* ==========================================================
     *  1) Depthwise Conv
     * ========================================================== */
    if (groups == Cin && groups == Cout) {
        if (engine_conv2d_depthwise_avx2(
                ip, wp, bp, op, N, Cin, H, W, KH, KW,
                cfg.stride_h, cfg.stride_w, cfg.pad_h, cfg.pad_w,
                cfg.dil_h, cfg.dil_w, fused_act, cfg.leaky_slope)) {
            engine_conv_last_path = "depthwise";
            return;
        }
    }

    /* ==========================================================
     *  2) Grouped Conv
     * ========================================================== */
    if (groups > 1) {
        if (engine_conv2d_grouped_gemm(
                ip, wp, bp, op, N, Cin, H, W, Cout, KH, KW, groups,
                cfg.stride_h, cfg.stride_w, cfg.pad_h, cfg.pad_w,
                cfg.dil_h, cfg.dil_w, fused_act, cfg.leaky_slope)) {
            engine_conv_last_path = "grouped";
            return;
        }
    }

    /* ==========================================================
     *  3) Direct NHWC — Stem Katmanları, Conv1D ve Küçük Cin (Cin <= 16)
     * ========================================================== */
    const bool is_1x1_pure = (KH == 1 && KW == 1 && cfg.pad_h == 0 && cfg.pad_w == 0 &&
                              cfg.stride_h == 1 && cfg.stride_w == 1 &&
                              cfg.dil_h == 1 && cfg.dil_w == 1);

    if (!is_1x1_pure && Cin <= 16 && groups == 1 && KH <= 7 && KW <= 7) {
        if (engine_conv2d_direct_nhwc(
                ip, wp, bp, op, N, Cin, H, W, Cout, KH, KW,
                cfg.stride_h, cfg.stride_w, cfg.pad_h, cfg.pad_w,
                cfg.dil_h, cfg.dil_w, fused_act, cfg.leaky_slope)) {
            engine_conv_last_path = "direct_nhwc";
            return;
        }
    }

    /* ==========================================================
     *  4) Winograd F(2x2, 3x3) — Cin > 16, 3x3, Stride=1, Dilation=1
     * ========================================================== */
    const bool can_winograd = (KH == 3 && KW == 3 &&
                               cfg.stride_h == 1 && cfg.stride_w == 1 &&
                               cfg.dil_h == 1 && cfg.dil_w == 1 &&
                               groups == 1 &&
                               Cin <= 512);
    if (can_winograd) {
        if (engine_conv2d_winograd_3x3(
                ip, wp, bp, op, N, Cin, H, W, Cout,
                cfg.pad_h, cfg.pad_w, fused_act, cfg.leaky_slope)) {
            engine_conv_last_path = "winograd";
            return;
        }
    }

    /* ==========================================================
     *  5) NHWC GEMM / 1x1 Zero-Copy Fused Direct Microkernel
     * ========================================================== */
    if (engine_conv2d_gemm(
            ip, wp, bp, op, N, Cin, H, W, Cout, KH, KW,
            cfg.stride_h, cfg.stride_w, cfg.pad_h, cfg.pad_w,
            cfg.dil_h, cfg.dil_w, fused_act, cfg.leaky_slope)) {
        engine_conv_last_path = "gemm";
        return;
    }

    /* ==========================================================
     *  6) Skaler Fallback
     * ========================================================== */
    engine_conv_last_path = "scalar";
    conv2d_naive_nhwc(ip, wp, bp, op, N, H, W, Cin, Cout, KH, KW, Hout, Wout, groups,
                      cfg.stride_h, cfg.stride_w, cfg.pad_h, cfg.pad_w,
                      cfg.dil_h, cfg.dil_w, fused_act, cfg.leaky_slope);
}

Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor* bias,
              const Conv2dConfig& cfg) noexcept
{
    const Shape out_shape = conv2d_output_shape(input.shape(), weight.shape(), cfg);
    if (out_shape.rank() == 0) return Tensor{};
    Tensor out = Tensor::empty(out_shape, DType::F32);
    if (out.is_empty()) return out;
    conv2d(input, weight, bias, out, cfg);
    return out;
}

Tensor conv2d_in_arena(memory::Arena& arena, const Tensor& input,
                       const Tensor& weight, const Tensor* bias,
                       const Conv2dConfig& cfg) noexcept
{
    const Shape out_shape = conv2d_output_shape(input.shape(), weight.shape(), cfg);
    if (out_shape.rank() == 0) return Tensor{};
    Tensor out = Tensor::create_in_arena(arena, out_shape, DType::F32);
    if (out.is_empty()) return out;
    conv2d(input, weight, bias, out, cfg);
    return out;
}

} /* namespace ops */
} /* namespace engine */