/* ============================================================================
 *  conv2d_pointwise_avx2.cpp  —  1×1 s1 p0, 8-co blocking + packed weights.
 *  SiLU (fused_activation == 5) AVX2 polinom exp ile desteklenir.
 * ========================================================================== */
#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include "engine/runtime/thread_pool.hpp"
#include <cmath>
namespace {

constexpr int64_t COB = 8;

std::shared_mutex g_wc_mtx;
std::unordered_map<const float*, std::vector<float>> g_wc;

/* ---------- AVX2 polinom exp (Jordanp tarzı) ------------------------------ */
inline __m256 exp_ps_avx2(__m256 x) noexcept {
    const __m256 kLog2e   = _mm256_set1_ps(1.44269504088896341f);
    const __m256 kHiBound = _mm256_set1_ps(88.3762626647949f);
    const __m256 kLoBound = _mm256_set1_ps(-88.3762626647949f);

    x = _mm256_min_ps(x, kHiBound);
    x = _mm256_max_ps(x, kLoBound);

    __m256 fx = _mm256_mul_ps(x, kLog2e);

    const __m256 kMagic = _mm256_set1_ps(12582912.0f);
    __m256 tmp = _mm256_add_ps(fx, kMagic);
    __m256 n   = _mm256_sub_ps(tmp, kMagic);
    __m256 f   = _mm256_sub_ps(fx, n);

    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));

    __m256i n_int = _mm256_cvtps_epi32(n);
    __m256 scale = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(n_int, _mm256_set1_epi32(127)), 23));

    return _mm256_mul_ps(p, scale);
}

inline __m256 sigmoid_ps_avx2(__m256 x) noexcept {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), x);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_ps_avx2(neg)));
}



// MSVC ve AVX2 için optimize edilmiş yüksek hassasiyetli inline SiLU fonksiyonu
inline __m256 silu_ps_avx2(__m256 x) noexcept {
    const __m256 lon_ln2hi = _mm256_set1_ps(-6.9314575195e-1f);
    const __m256 lon_ln2lo = _mm256_set1_ps(-1.4286068203e-7f);
    const __m256 lon_invln2 = _mm256_set1_ps(1.44269504088896340735f);
    const __m256 ones = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();

    __m256 neg_x = _mm256_sub_ps(zero, x);
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(neg_x, lon_invln2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    
    __m256 r = _mm256_fmadd_ps(fx, lon_ln2hi, neg_x);
    r = _mm256_fmadd_ps(fx, lon_ln2lo, r);

    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, r, ones);
    y = _mm256_fmadd_ps(y, r, ones);

    __m256i imm0 = _mm256_cvtps_epi32(fx);
    imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7F));
    __m256 pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(imm0, 23));

    __m256 exp_neg_x = _mm256_mul_ps(y, pow2n);
    __m256 denominator = _mm256_add_ps(ones, exp_neg_x);

    return _mm256_div_ps(x, denominator);
}


inline float silu_scalar(float x) noexcept {
    return x / (1.0f + std::exp(-x));
}

/* ---------- apply_act: 1..5 ---------------------------------------------- */
inline __m256 apply_act_avx2(__m256 v, int act, float slope) noexcept {
    if (act == 1) return _mm256_max_ps(v, _mm256_setzero_ps());           /* ReLU    */
    if (act == 2) {                                                       /* Leaky   */
        __m256 z = _mm256_setzero_ps();
        __m256 s = _mm256_mul_ps(v, _mm256_set1_ps(slope));
        return _mm256_blendv_ps(s, v, _mm256_cmp_ps(v, z, _CMP_GT_OS));
    }
    if (act == 3) return sigmoid_ps_avx2(v);                              /* Sigmoid */
    if (act == 4) {                                                       /* Tanh    */
        const __m256 two = _mm256_set1_ps(2.0f);
        const __m256 one = _mm256_set1_ps(1.0f);
        __m256 e = exp_ps_avx2(_mm256_sub_ps(_mm256_setzero_ps(),
                                              _mm256_mul_ps(v, two)));
        return _mm256_sub_ps(_mm256_mul_ps(two,
                    _mm256_div_ps(one, _mm256_add_ps(one, e))), one);
    }
    if (act == 5) return silu_ps_avx2(v);                                 /* SiLU    */  // ← YENİ
    return v;
}

const std::vector<float>* get_packed_weight(
    const float* weight, int64_t Cin, int64_t Cout) noexcept
{
    {
        std::shared_lock<std::shared_mutex> lk(g_wc_mtx);
        auto it = g_wc.find(weight);
        if (it != g_wc.end()) return &it->second;
    }
    std::unique_lock<std::shared_mutex> lk(g_wc_mtx);
    auto it = g_wc.find(weight);
    if (it != g_wc.end()) return &it->second;

    const int64_t n_coblk = (Cout + COB - 1) / COB;
    std::vector<float> packed(static_cast<size_t>(n_coblk) * Cin * COB, 0.0f);

    for (int64_t cb = 0; cb < n_coblk; ++cb) {
        const int64_t co_base = cb * COB;
        for (int64_t ci = 0; ci < Cin; ++ci) {
            float* dst = packed.data() + (cb * Cin + ci) * COB;
            for (int64_t c = 0; c < COB; ++c) {
                const int64_t co = co_base + c;
                dst[c] = (co < Cout) ? weight[co * Cin + ci] : 0.0f;
            }
        }
    }
    auto pr = g_wc.emplace(weight, std::move(packed));
    return &pr.first->second;
}

} // namespace

extern "C" int engine_conv2d_pointwise_avx2(
    const float* input, const float* weight, const float* bias,
    float* output, int64_t N, int64_t Cin, int64_t HW, int64_t Cout,
    int fused_activation, float leaky_slope) noexcept
{
    if (N <= 0 || Cin <= 0 || HW <= 0 || Cout <= 0) return 0;

    const std::vector<float>* packed = get_packed_weight(weight, Cin, Cout);
    if (!packed) return 0;
    const float* W = packed->data();

    const int64_t n_coblk = (Cout + COB - 1) / COB;
    auto& pool = engine::runtime::GlobalPool::get();

    const int64_t hw_blocks = (HW + 7) / 8;
    const int64_t total = N * hw_blocks;

    pool.parallel_for(0, total, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
            const int64_t n    = t / hw_blocks;
            const int64_t hw_b = t % hw_blocks;
            const int64_t hw   = hw_b * 8;
            const int64_t w_len = std::min<int64_t>(8, HW - hw);

            const float* in_n  = input  + n * Cin  * HW;
            float*       out_n = output + n * Cout * HW;

            for (int64_t cb = 0; cb < n_coblk; ++cb) {
                const int64_t co_base = cb * COB;
                const int64_t co_cnt  = std::min<int64_t>(COB, Cout - co_base);

                __m256 a0 = _mm256_setzero_ps();
                __m256 a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps();
                __m256 a3 = _mm256_setzero_ps();
                __m256 a4 = _mm256_setzero_ps();
                __m256 a5 = _mm256_setzero_ps();
                __m256 a6 = _mm256_setzero_ps();
                __m256 a7 = _mm256_setzero_ps();

                if (bias) {
                    if (co_cnt > 0) a0 = _mm256_set1_ps(bias[co_base + 0]);
                    if (co_cnt > 1) a1 = _mm256_set1_ps(bias[co_base + 1]);
                    if (co_cnt > 2) a2 = _mm256_set1_ps(bias[co_base + 2]);
                    if (co_cnt > 3) a3 = _mm256_set1_ps(bias[co_base + 3]);
                    if (co_cnt > 4) a4 = _mm256_set1_ps(bias[co_base + 4]);
                    if (co_cnt > 5) a5 = _mm256_set1_ps(bias[co_base + 5]);
                    if (co_cnt > 6) a6 = _mm256_set1_ps(bias[co_base + 6]);
                    if (co_cnt > 7) a7 = _mm256_set1_ps(bias[co_base + 7]);
                }

                const float* w_blk = W + cb * Cin * COB;

                for (int64_t ci = 0; ci < Cin; ++ci) {
                    __m256 in_v;
                    if (w_len == 8) {
                        in_v = _mm256_loadu_ps(in_n + ci * HW + hw);
                    } else {
                        alignas(32) float tmp[8] = {0,0,0,0,0,0,0,0};
                        for (int64_t i = 0; i < w_len; ++i) tmp[i] = in_n[ci * HW + hw + i];
                        in_v = _mm256_load_ps(tmp);
                    }
                    const float* wc = w_blk + ci * COB;
                    a0 = _mm256_fmadd_ps(_mm256_set1_ps(wc[0]), in_v, a0);
                    a1 = _mm256_fmadd_ps(_mm256_set1_ps(wc[1]), in_v, a1);
                    a2 = _mm256_fmadd_ps(_mm256_set1_ps(wc[2]), in_v, a2);
                    a3 = _mm256_fmadd_ps(_mm256_set1_ps(wc[3]), in_v, a3);
                    a4 = _mm256_fmadd_ps(_mm256_set1_ps(wc[4]), in_v, a4);
                    a5 = _mm256_fmadd_ps(_mm256_set1_ps(wc[5]), in_v, a5);
                    a6 = _mm256_fmadd_ps(_mm256_set1_ps(wc[6]), in_v, a6);
                    a7 = _mm256_fmadd_ps(_mm256_set1_ps(wc[7]), in_v, a7);
                }

                a0 = apply_act_avx2(a0, fused_activation, leaky_slope);
                a1 = apply_act_avx2(a1, fused_activation, leaky_slope);
                a2 = apply_act_avx2(a2, fused_activation, leaky_slope);
                a3 = apply_act_avx2(a3, fused_activation, leaky_slope);
                a4 = apply_act_avx2(a4, fused_activation, leaky_slope);
                a5 = apply_act_avx2(a5, fused_activation, leaky_slope);
                a6 = apply_act_avx2(a6, fused_activation, leaky_slope);
                a7 = apply_act_avx2(a7, fused_activation, leaky_slope);

                float* ob = out_n + co_base * HW + hw;
                if (w_len == 8) {
                    if (co_cnt > 0) _mm256_storeu_ps(ob + 0*HW, a0);
                    if (co_cnt > 1) _mm256_storeu_ps(ob + 1*HW, a1);
                    if (co_cnt > 2) _mm256_storeu_ps(ob + 2*HW, a2);
                    if (co_cnt > 3) _mm256_storeu_ps(ob + 3*HW, a3);
                    if (co_cnt > 4) _mm256_storeu_ps(ob + 4*HW, a4);
                    if (co_cnt > 5) _mm256_storeu_ps(ob + 5*HW, a5);
                    if (co_cnt > 6) _mm256_storeu_ps(ob + 6*HW, a6);
                    if (co_cnt > 7) _mm256_storeu_ps(ob + 7*HW, a7);
                } else {
                    alignas(32) float tmp[8];
                    auto st = [&](__m256 v, int c) {
                        if (c >= co_cnt) return;
                        _mm256_store_ps(tmp, v);
                        float* d = ob + c * HW;
                        for (int64_t i = 0; i < w_len; ++i) d[i] = tmp[i];
                    };
                    st(a0,0); st(a1,1); st(a2,2); st(a3,3);
                    st(a4,4); st(a5,5); st(a6,6); st(a7,7);
                }
            }
        }
    });

    return 1;
}

extern "C" void engine_conv2d_pointwise_clear_cache() noexcept {
    std::unique_lock<std::shared_mutex> lk(g_wc_mtx);
    g_wc.clear();
}