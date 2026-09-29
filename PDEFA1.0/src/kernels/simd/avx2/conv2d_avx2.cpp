/* ============================================================================
 *  src/kernels/simd/avx2/conv2d_avx2.cpp
 *  Engine-AI — Conv2d AVX2 (Direct + Cout-Blocking + Packed Weights)
 *
 *  Optimizasyonlar:
 *    1. Weight packing (3x3): Sadece 1 Kez (Lock-Free Shared Mutex)
 *    2. CO_BLOCK = 64 → weight L2'ye sığar
 *    3. HO_BLOCK = 4 → 4 satır aynı weight'i tekrar kullanır
 * ========================================================================== */
#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <shared_mutex>

#include "engine/runtime/thread_pool.hpp"
#include "engine/runtime/cpu_info.hpp"

#if defined(__GNUC__) || defined(__clang__)
#  define ENGINE_RESTRICT __restrict__
#else
#  define ENGINE_RESTRICT
#endif

namespace {

inline __m256 load_strided2(const float* ptr) noexcept {
    __m256 v0 = _mm256_loadu_ps(ptr);
    __m256 v1 = _mm256_loadu_ps(ptr + 8);
    __m256 even = _mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(2, 0, 2, 0));
    const __m256i perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    return _mm256_permutevar8x32_ps(even, perm);
}

inline __m256 load_8_safe(const float* row, int64_t w_start,
                          int64_t stride, int64_t W) noexcept {
    alignas(32) float tmp[8];
    for (int i = 0; i < 8; ++i) {
        const int64_t w = w_start + i * stride;
        tmp[i] = (w >= 0 && w < W) ? row[w] : 0.0f;
    }
    return _mm256_load_ps(tmp);
}

inline __m256 fast_silu_avx2(__m256 x) noexcept {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    // clamp
    const __m256 kHi = _mm256_set1_ps(88.376f);
    const __m256 kLo = _mm256_set1_ps(-88.376f);
    neg_x = _mm256_min_ps(neg_x, kHi);
    neg_x = _mm256_max_ps(neg_x, kLo);
    // exp
    const __m256 kLog2e = _mm256_set1_ps(1.44269504088896341f);
    __m256 fx = _mm256_mul_ps(neg_x, kLog2e);
    __m256 n = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 kC1 = _mm256_set1_ps(0.693359375f);
    const __m256 kC2 = _mm256_set1_ps(-2.121944400546905827679e-4f);
    __m256 y = _mm256_fnmadd_ps(n, kC1, neg_x);
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
    p = _mm256_add_ps(p, one);
    __m256i n_int = _mm256_cvtps_epi32(n);
    __m256 scale = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(n_int, _mm256_set1_epi32(127)), 23));
    __m256 e = _mm256_mul_ps(p, scale);
    // sigmoid = 1 / (1 + e)
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 denom = _mm256_add_ps(one, e);
    __m256 rcp = _mm256_rcp_ps(denom);
    __m256 sig = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, two));
    return _mm256_mul_ps(x, sig);
}

inline __m256 apply_act_avx2(__m256 v, int act, float slope) noexcept {
    if (act == 1) {
        return _mm256_max_ps(v, _mm256_setzero_ps());
    } else if (act == 2) {
        __m256 zero = _mm256_setzero_ps();
        __m256 scaled = _mm256_mul_ps(v, _mm256_set1_ps(slope));
        __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OS);
        return _mm256_blendv_ps(scaled, v, mask);
    } else if (act == 5) {
        return fast_silu_avx2(v);   // ← AVX2 versiyonu
    } else if (act == 3) {
        return _mm256_div_ps(_mm256_set1_ps(1.0f),
                             _mm256_add_ps(_mm256_set1_ps(1.0f),
                             /* exp(-v) inline */ v));  // basit sigmoid
    }
    // Geri kalan act'ler için scalar fallback
    alignas(32) float buf[8];
    _mm256_store_ps(buf, v);
    for (int i = 0; i < 8; ++i) {
        if (act == 4) {
            const float inner = 0.7978845608f * (buf[i] + 0.044715f * buf[i] * buf[i] * buf[i]);
            buf[i] = 0.5f * buf[i] * (1.0f + std::tanh(inner));
        }
    }
    return _mm256_load_ps(buf);
}

/* ============================================================================
 *  LOCK-FREE WEIGHT PACK CACHE (mevcut direct conv icin)
 * ========================================================================== */
std::shared_mutex g_direct_wcache_mtx;
std::unordered_map<const float*, std::vector<float>> g_direct_wcache;

void pack_weight_3x3_impl(const float* weight, float* packed, int64_t Cin, int64_t Cout) noexcept {
    const int64_t K_dim = Cin * 9;
    const int64_t cout_blocks = (Cout + 7) / 8;

    for (int64_t cb = 0; cb < cout_blocks; ++cb) {
        const int64_t co_start = cb * 8;
        const int64_t num_c = std::min<int64_t>(8, Cout - co_start);

        for (int64_t k = 0; k < K_dim; ++k) {
            float* dst = packed + cb * K_dim * 8 + k * 8;
            for (int64_t c = 0; c < num_c; ++c) {
                dst[c] = weight[(co_start + c) * K_dim + k];
            }
            for (int64_t c = num_c; c < 8; ++c) {
                dst[c] = 0.0f;
            }
        }
    }
}

const std::vector<float>* get_packed_weight_3x3(const float* weight, int64_t Cin, int64_t Cout) noexcept {
    {
        std::shared_lock<std::shared_mutex> read_lock(g_direct_wcache_mtx);
        auto it = g_direct_wcache.find(weight);
        if (it != g_direct_wcache.end()) return &it->second;
    }

    std::unique_lock<std::shared_mutex> write_lock(g_direct_wcache_mtx);
    auto it = g_direct_wcache.find(weight);
    if (it != g_direct_wcache.end()) return &it->second;

    const int64_t K_dim = Cin * 9;
    const int64_t cout_blocks = (Cout + 7) / 8;
    std::vector<float> packed(static_cast<size_t>(cout_blocks) * K_dim * 8, 0.0f);

    pack_weight_3x3_impl(weight, packed.data(), Cin, Cout);

    auto [new_it, inserted] = g_direct_wcache.emplace(weight, std::move(packed));
    return &new_it->second;
}

// Helper: 16 kanal (2x AVX2 vektörü) için aktivasyon ve güvenli yazma
static inline void store_16_outputs(
    float* dst, __m256 acc0, __m256 acc1,
    int fused_activation, float leaky_slope, int64_t rem) noexcept
{
    acc0 = apply_act_avx2(acc0, fused_activation, leaky_slope);
    acc1 = apply_act_avx2(acc1, fused_activation, leaky_slope);

    if (rem >= 16) {
        _mm256_storeu_ps(dst, acc0);
        _mm256_storeu_ps(dst + 8, acc1);
    } else if (rem >= 8) {
        _mm256_storeu_ps(dst, acc0);
        const int64_t r1 = rem - 8;
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc1);
        for (int64_t i = 0; i < r1; ++i) dst[8 + i] = tmp[i];
    } else if (rem > 0) {
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc0);
        for (int64_t i = 0; i < rem; ++i) dst[i] = tmp[i];
    }
}

} /* anonymous namespace */

extern "C" {

/* ============================================================================
 *  NHWC Direct Conv — küçük Cin için (stem: 3→16, 3→32)
 *  Vektörleştirme: 8 çıkış kanalı (Cout) üzerinden
 *  Düzen: input[N,H,W,Cin], weight[Cout,Cin,KH,KW], output[N,Hout,Wout,Cout]
 * ========================================================================== */
int engine_conv2d_direct_nhwc(
    const float* input, const float* weight, const float* bias, float* output,
    int64_t N, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout, int64_t KH, int64_t KW,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w, int fused_activation, float leaky_slope) noexcept
{
    const int64_t Hout = (H + 2 * pad_h - (KH - 1) * dil_h - 1) / stride_h + 1;
    const int64_t Wout = (W + 2 * pad_w - (KW - 1) * dil_w - 1) / stride_w + 1;
    if (Hout <= 0 || Wout <= 0) return 0;

    const int64_t taps = KH * KW * Cin;
    // 2x YMM (16 float) vektör hizalaması için pad
    const int64_t cout_pad = (Cout + 15) & ~(int64_t)15;

    /* --- Ağırlık Paketleme: [taps][cout_pad] --- */
    static thread_local std::vector<float> t_packed;
    if ((int64_t)t_packed.size() < taps * cout_pad)
        t_packed.resize(taps * cout_pad);
    float* packed = t_packed.data();

    for (int64_t kh = 0; kh < KH; ++kh) {
        for (int64_t kw = 0; kw < KW; ++kw) {
            for (int64_t ci = 0; ci < Cin; ++ci) {
                const int64_t tap = (kh * KW + kw) * Cin + ci;
                float* dst = packed + tap * cout_pad;
                for (int64_t co = 0; co < Cout; ++co)
                    dst[co] = weight[((co * Cin + ci) * KH + kh) * KW + kw];
                for (int64_t co = Cout; co < cout_pad; ++co)
                    dst[co] = 0.0f;
            }
        }
    }

    // Sınır analizi: wi değerinin daima [0, W) kaldığı güvenli iç aralığı belirle
    const int64_t wo_left_safe = (pad_w > 0) ? (pad_w + stride_w - 1) / stride_w : 0;
    const int64_t limit_right = W + pad_w - (KW - 1) * dil_w;
    const int64_t wo_right_safe = (limit_right > 0) ? (limit_right - 1) / stride_w + 1 : 0;

    const int64_t wo_fast_start = std::max((int64_t)0, std::min(Wout, wo_left_safe));
    const int64_t wo_fast_end   = std::max(wo_fast_start, std::min(Wout, wo_right_safe));

    auto& pool = engine::runtime::GlobalPool::get();
    const int64_t total = N * Hout;

    pool.parallel_for(0, total, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
            const int64_t ho = t % Hout;
            const int64_t n  = t / Hout;
            const int64_t h_base = ho * stride_h - pad_h;

            const float* in_n = input + n * H * W * Cin;
            float* out_row = output + (n * Hout + ho) * Wout * Cout;

            // Dış döngü 16'lık Cout blokları: Ağırlıkların L1D cache'te kalmasını sağlar
            for (int64_t cg = 0; cg < cout_pad; cg += 16) {
                const int64_t rem = Cout - cg;

                // Bias yükleme (satır boyunca 1 kez yüklenip YMM'de tutulur)
                __m256 b0 = _mm256_setzero_ps();
                __m256 b1 = _mm256_setzero_ps();
                if (bias) {
                    if (rem >= 16) {
                        b0 = _mm256_loadu_ps(bias + cg);
                        b1 = _mm256_loadu_ps(bias + cg + 8);
                    } else if (rem >= 8) {
                        b0 = _mm256_loadu_ps(bias + cg);
                        alignas(32) float tmp[8] = {0};
                        for (int64_t i = 0; i < rem - 8; ++i) tmp[i] = bias[cg + 8 + i];
                        b1 = _mm256_load_ps(tmp);
                    } else if (rem > 0) {
                        alignas(32) float tmp[8] = {0};
                        for (int64_t i = 0; i < rem; ++i) tmp[i] = bias[cg + i];
                        b0 = _mm256_load_ps(tmp);
                    }
                }

                // Tek piksel hesaplayan yardımcı (Kenar sınır bölgeleri ve artıklar için)
                auto compute_pixel_1x16 = [&](int64_t wo, bool check_bounds) {
                    __m256 acc0 = b0;
                    __m256 acc1 = b1;
                    const int64_t w_base = wo * stride_w - pad_w;

                    for (int64_t kh = 0; kh < KH; ++kh) {
                        const int64_t hi = h_base + kh * dil_h;
                        if (hi < 0 || hi >= H) continue;

                        for (int64_t kw = 0; kw < KW; ++kw) {
                            const int64_t wi = w_base + kw * dil_w;
                            if (check_bounds && (wi < 0 || wi >= W)) continue;

                            const float* in_pix = in_n + (hi * W + wi) * Cin;
                            const float* w_ptr = packed + ((kh * KW + kw) * Cin) * cout_pad + cg;

                            int64_t ci = 0;
                            for (; ci + 1 < Cin; ci += 2) {
                                const __m256 w0_0 = _mm256_loadu_ps(w_ptr);
                                const __m256 w0_1 = _mm256_loadu_ps(w_ptr + 8);
                                const __m256 in0  = _mm256_set1_ps(in_pix[ci]);
                                acc0 = _mm256_fmadd_ps(in0, w0_0, acc0);
                                acc1 = _mm256_fmadd_ps(in0, w0_1, acc1);

                                const __m256 w1_0 = _mm256_loadu_ps(w_ptr + cout_pad);
                                const __m256 w1_1 = _mm256_loadu_ps(w_ptr + cout_pad + 8);
                                const __m256 in1  = _mm256_set1_ps(in_pix[ci + 1]);
                                acc0 = _mm256_fmadd_ps(in1, w1_0, acc0);
                                acc1 = _mm256_fmadd_ps(in1, w1_1, acc1);

                                w_ptr += 2 * cout_pad;
                            }
                            if (ci < Cin) {
                                const __m256 w0_0 = _mm256_loadu_ps(w_ptr);
                                const __m256 w0_1 = _mm256_loadu_ps(w_ptr + 8);
                                const __m256 in0  = _mm256_set1_ps(in_pix[ci]);
                                acc0 = _mm256_fmadd_ps(in0, w0_0, acc0);
                                acc1 = _mm256_fmadd_ps(in0, w0_1, acc1);
                            }
                        }
                    }
                    store_16_outputs(out_row + wo * Cout + cg, acc0, acc1, fused_activation, leaky_slope, rem);
                };

                // --- 1. Sol Sınır Bölgesi (Padding Kontrollü) ---
                for (int64_t wo = 0; wo < wo_fast_start; ++wo) {
                    compute_pixel_1x16(wo, true);
                }

                // --- 2. Hızlı İç Bölge (4 Piksel x 16 Kanal - Sıfır Sınır Kontrolü) ---
                int64_t wo = wo_fast_start;
                const int64_t fast_stride = stride_w * Cin;

                for (; wo + 3 < wo_fast_end; wo += 4) {
                    __m256 a0_0 = b0, a0_1 = b1;
                    __m256 a1_0 = b0, a1_1 = b1;
                    __m256 a2_0 = b0, a2_1 = b1;
                    __m256 a3_0 = b0, a3_1 = b1;

                    for (int64_t kh = 0; kh < KH; ++kh) {
                        const int64_t hi = h_base + kh * dil_h;
                        if (hi < 0 || hi >= H) continue;

                        for (int64_t kw = 0; kw < KW; ++kw) {
                            const int64_t wi0 = wo * stride_w - pad_w + kw * dil_w;
                            const float* in0 = in_n + (hi * W + wi0) * Cin;
                            const float* in1 = in0 + fast_stride;
                            const float* in2 = in1 + fast_stride;
                            const float* in3 = in2 + fast_stride;

                            const float* w_ptr = packed + ((kh * KW + kw) * Cin) * cout_pad + cg;

                            int64_t ci = 0;
                            // 2x unrolled Ci: Ağırlık yüklerini 4 piksel arasında paylaş
                            for (; ci + 1 < Cin; ci += 2) {
                                const __m256 w0_0 = _mm256_loadu_ps(w_ptr);
                                const __m256 w0_1 = _mm256_loadu_ps(w_ptr + 8);
                                const float* w_next = w_ptr + cout_pad;
                                const __m256 w1_0 = _mm256_loadu_ps(w_next);
                                const __m256 w1_1 = _mm256_loadu_ps(w_next + 8);
                                w_ptr = w_next + cout_pad;

                                const __m256 i0_0 = _mm256_set1_ps(in0[ci]);
                                const __m256 i1_0 = _mm256_set1_ps(in1[ci]);
                                const __m256 i2_0 = _mm256_set1_ps(in2[ci]);
                                const __m256 i3_0 = _mm256_set1_ps(in3[ci]);

                                a0_0 = _mm256_fmadd_ps(i0_0, w0_0, a0_0);
                                a0_1 = _mm256_fmadd_ps(i0_0, w0_1, a0_1);
                                a1_0 = _mm256_fmadd_ps(i1_0, w0_0, a1_0);
                                a1_1 = _mm256_fmadd_ps(i1_0, w0_1, a1_1);
                                a2_0 = _mm256_fmadd_ps(i2_0, w0_0, a2_0);
                                a2_1 = _mm256_fmadd_ps(i2_0, w0_1, a2_1);
                                a3_0 = _mm256_fmadd_ps(i3_0, w0_0, a3_0);
                                a3_1 = _mm256_fmadd_ps(i3_0, w0_1, a3_1);

                                const __m256 i0_1 = _mm256_set1_ps(in0[ci + 1]);
                                const __m256 i1_1 = _mm256_set1_ps(in1[ci + 1]);
                                const __m256 i2_1 = _mm256_set1_ps(in2[ci + 1]);
                                const __m256 i3_1 = _mm256_set1_ps(in3[ci + 1]);

                                a0_0 = _mm256_fmadd_ps(i0_1, w1_0, a0_0);
                                a0_1 = _mm256_fmadd_ps(i0_1, w1_1, a0_1);
                                a1_0 = _mm256_fmadd_ps(i1_1, w1_0, a1_0);
                                a1_1 = _mm256_fmadd_ps(i1_1, w1_1, a1_1);
                                a2_0 = _mm256_fmadd_ps(i2_1, w1_0, a2_0);
                                a2_1 = _mm256_fmadd_ps(i2_1, w1_1, a2_1);
                                a3_0 = _mm256_fmadd_ps(i3_1, w1_0, a3_0);
                                a3_1 = _mm256_fmadd_ps(i3_1, w1_1, a3_1);
                            }
                            if (ci < Cin) {
                                const __m256 w0_0 = _mm256_loadu_ps(w_ptr);
                                const __m256 w0_1 = _mm256_loadu_ps(w_ptr + 8);
                                w_ptr += cout_pad;

                                const __m256 i0 = _mm256_set1_ps(in0[ci]);
                                const __m256 i1 = _mm256_set1_ps(in1[ci]);
                                const __m256 i2 = _mm256_set1_ps(in2[ci]);
                                const __m256 i3 = _mm256_set1_ps(in3[ci]);

                                a0_0 = _mm256_fmadd_ps(i0, w0_0, a0_0);
                                a0_1 = _mm256_fmadd_ps(i0, w0_1, a0_1);
                                a1_0 = _mm256_fmadd_ps(i1, w0_0, a1_0);
                                a1_1 = _mm256_fmadd_ps(i1, w0_1, a1_1);
                                a2_0 = _mm256_fmadd_ps(i2, w0_0, a2_0);
                                a2_1 = _mm256_fmadd_ps(i2, w0_1, a2_1);
                                a3_0 = _mm256_fmadd_ps(i3, w0_0, a3_0);
                                a3_1 = _mm256_fmadd_ps(i3, w0_1, a3_1);
                            }
                        }
                    }

                    store_16_outputs(out_row + (wo + 0) * Cout + cg, a0_0, a0_1, fused_activation, leaky_slope, rem);
                    store_16_outputs(out_row + (wo + 1) * Cout + cg, a1_0, a1_1, fused_activation, leaky_slope, rem);
                    store_16_outputs(out_row + (wo + 2) * Cout + cg, a2_0, a2_1, fused_activation, leaky_slope, rem);
                    store_16_outputs(out_row + (wo + 3) * Cout + cg, a3_0, a3_1, fused_activation, leaky_slope, rem);
                }

                // İç bölge artıkları (4'e tam bölünmeyen son 1-3 piksel)
                for (; wo < wo_fast_end; ++wo) {
                    compute_pixel_1x16(wo, false);
                }

                // --- 3. Sağ Sınır Bölgesi (Padding Kontrollü) ---
                for (; wo < Wout; ++wo) {
                    compute_pixel_1x16(wo, true);
                }
            }
        }
    });

    return 1;
}

/* ============================================================================
 *  ESKI: NCHW Direct Conv (artik kullanilmiyor, referans kalsin)
 *  NHWC motor icin YANLIS layout, cagrilmamali.
 * ========================================================================== */
int engine_conv2d_try_avx2_direct(
    const float* input, const float* weight, const float* bias, float* output,
    int64_t N, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout, int64_t KH, int64_t KW,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w, int fused_activation, float leaky_slope) noexcept
{
    const int64_t Hout = (H + 2 * pad_h - (KH - 1) * dil_h - 1) / stride_h + 1;
    const int64_t Wout = (W + 2 * pad_w - (KW - 1) * dil_w - 1) / stride_w + 1;
    if (Hout <= 0 || Wout <= 0) return 0;

    const int64_t K_dim = Cin * KH * KW;
    const bool use_packed = (KH == 3 && KW == 3);
    const float* w_ptr = weight;

    if (use_packed) {
        const auto* packed_vec = get_packed_weight_3x3(weight, Cin, Cout);
        if (!packed_vec) return 0;
        w_ptr = packed_vec->data();
    }

    auto& pool = engine::runtime::GlobalPool::get();

    const auto& cache_info = engine::runtime::Cpu::cache();

    int64_t l1_bytes = static_cast<int64_t>(cache_info.l1_data);
    int64_t l2_bytes = static_cast<int64_t>(cache_info.l2);
    int64_t l3_bytes = static_cast<int64_t>(cache_info.l3);
    if (l1_bytes <= 0) l1_bytes = 16 * 1024;
    if (l2_bytes <= 0) l2_bytes = l1_bytes * 4;
    if (l3_bytes <= 0) l3_bytes = l2_bytes * 4;

    int64_t CO_BLOCK = 0;
    {
        const int64_t budgets[2] = { l2_bytes * 3 / 4, l3_bytes * 3 / 4 };
        for (int b = 0; b < 2; ++b) {
            int64_t cand = budgets[b] / (K_dim * 4);
            if (cand >= 8) { CO_BLOCK = cand; break; }
        }
        if (CO_BLOCK < 8) CO_BLOCK = 8;
        CO_BLOCK = ((CO_BLOCK + 7) / 8) * 8;
        if (CO_BLOCK > Cout) CO_BLOCK = Cout;
    }

    int64_t HO_BLOCK = (l1_bytes * 3 / 4) / (Cin * W * 4);
    if (HO_BLOCK < 1)    HO_BLOCK = 1;
    if (HO_BLOCK > 32)   HO_BLOCK = 32;
    if (HO_BLOCK > Hout) HO_BLOCK = Hout;

    const int64_t co_groups  = (Cout + CO_BLOCK - 1) / CO_BLOCK;
    const int64_t ho_groups  = (Hout + HO_BLOCK - 1) / HO_BLOCK;
    const int64_t total_tasks = N * co_groups * ho_groups;

    pool.parallel_for(0, total_tasks, [&, w_ptr](int64_t t_start, int64_t t_end) {
        for (int64_t t = t_start; t < t_end; ++t) {
            const int64_t hg = t % ho_groups;
            const int64_t t1 = t / ho_groups;
            const int64_t cg = t1 % co_groups;
            const int64_t n  = t1 / co_groups;

            const int64_t ho_start = hg * HO_BLOCK;
            const int64_t ho_end = std::min<int64_t>(ho_start + HO_BLOCK, Hout);

            const int64_t co_g_start = cg * CO_BLOCK;
            const int64_t co_g_count = std::min<int64_t>(CO_BLOCK, Cout - co_g_start);

            const float* in_n = input + n * Cin * H * W;
            float* out_n = output + n * Cout * Hout * Wout;

            for (int64_t ho = ho_start; ho < ho_end; ++ho) {
                const int64_t h_base = ho * stride_h - pad_h;

                for (int64_t wo = 0; wo < Wout; wo += 8) {
                    const int64_t w_rem = Wout - wo;
                    const int64_t w_len = (w_rem < 8) ? w_rem : 8;

                    for (int64_t c_base = 0; c_base < co_g_count; c_base += 8) {
                        const int64_t co_start = co_g_start + c_base;
                        const int64_t num_c = std::min<int64_t>(8, co_g_count - c_base);
                        const int64_t cb = co_start / 8;

                        __m256 acc[8];
                        if (bias) {
                            for (int c = 0; c < num_c; ++c)
                                acc[c] = _mm256_set1_ps(bias[co_start + c]);
                        } else {
                            for (int c = 0; c < num_c; ++c)
                                acc[c] = _mm256_setzero_ps();
                        }

                        for (int64_t ci = 0; ci < Cin; ++ci) {
                            const float* in_c = in_n + ci * H * W;
                            for (int64_t kh = 0; kh < KH; ++kh) {
                                const int64_t hi = h_base + kh * dil_h;
                                if (hi < 0 || hi >= H) continue;
                                const float* in_row = in_c + hi * W;

                                for (int64_t kw = 0; kw < KW; ++kw) {
                                    const int64_t w_base = wo * stride_w - pad_w + kw * dil_w;
                                    __m256 v_in;

                                    if (w_base >= 0 && w_base + 7 * stride_w < W && w_len == 8) {
                                        if (stride_w == 1)      v_in = _mm256_loadu_ps(in_row + w_base);
                                        else if (stride_w == 2) v_in = load_strided2(in_row + w_base);
                                        else                    v_in = load_8_safe(in_row, w_base, stride_w, W);
                                    } else {
                                        v_in = load_8_safe(in_row, w_base, stride_w, W);
                                    }

                                    const int64_t w_off = ci * KH * KW + kh * KW + kw;

                                    if (use_packed) {
                                        const float* w_row = w_ptr + cb * K_dim * 8 + w_off * 8;
                                        for (int c = 0; c < num_c; ++c) {
                                            acc[c] = _mm256_fmadd_ps(
                                                _mm256_set1_ps(w_row[c]), v_in, acc[c]);
                                        }
                                    } else {
                                        for (int c = 0; c < num_c; ++c) {
                                            float w_val = w_ptr[(co_start + c) * K_dim + w_off];
                                            acc[c] = _mm256_fmadd_ps(
                                                _mm256_set1_ps(w_val), v_in, acc[c]);
                                        }
                                    }
                                }
                            }
                        }

                        for (int c = 0; c < num_c; ++c) {
                            __m256 v_out = apply_act_avx2(acc[c], fused_activation, leaky_slope);
                            float* out_ptr = out_n + (co_start + c) * Hout * Wout + ho * Wout + wo;
                            if (w_len == 8) {
                                _mm256_storeu_ps(out_ptr, v_out);
                            } else {
                                alignas(32) float tmp_out[8];
                                _mm256_store_ps(tmp_out, v_out);
                                for (int i = 0; i < w_len; ++i) out_ptr[i] = tmp_out[i];
                            }
                        }
                    }
                }
            }
        }
    });

    return 1;
}

void engine_conv2d_direct_clear_cache() noexcept {
    std::unique_lock<std::shared_mutex> lk(g_direct_wcache_mtx);
    g_direct_wcache.clear();
}

} /* extern "C" */