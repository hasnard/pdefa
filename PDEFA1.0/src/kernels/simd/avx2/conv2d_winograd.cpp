/* ============================================================================
 *  src/kernels/simd/avx2/conv2d_winograd.cpp
 *  Engine-AI — Winograd F(2x2, 3x3) AVX2 — NHWC
 *
 *  v4: 
 *    - Memory aligned caching (_mm_malloc, 32-byte)
 *    - CO_BLK bloklu ardışık (sequential) ağırlık paketlemesi
 *    - Tamamen vektörel SSE (__m128) input transform (V = B^T d B)
 *    - FMA Çekirdeği 2x Loop Unrolling ve ILP maksimizasyonu
 * ========================================================================== */
#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <algorithm>
#include <memory>

#include "engine/runtime/thread_pool.hpp"
#include "engine/runtime/cpu_info.hpp"

namespace {
struct AlignedFree {
    inline void operator()(void* ptr) const noexcept {
        if (ptr) _mm_free(ptr);
    }
};
constexpr int64_t CO_BLK = 4;

inline void weight_transform(const float g[3][3], float U[4][4]) noexcept {
    float t[4][3];
    for (int j = 0; j < 3; ++j) {
        t[0][j] = g[0][j];
        t[1][j] = 0.5f * (g[0][j] + g[1][j] + g[2][j]);
        t[2][j] = 0.5f * (g[0][j] - g[1][j] + g[2][j]);
        t[3][j] = g[2][j];
    }
    for (int i = 0; i < 4; ++i) {
        U[i][0] = t[i][0];
        U[i][1] = 0.5f * (t[i][0] + t[i][1] + t[i][2]);
        U[i][2] = 0.5f * (t[i][0] - t[i][1] + t[i][2]);
        U[i][3] = t[i][2];
    }
}

/* V = B^T d B dönüşümünün tamamen vektörel SSE/AVX uyarlaması */
inline void input_transform_simd(const float* __restrict d, float* __restrict V) noexcept {
    // 32-byte hizalı okuma
    __m128 d0 = _mm_load_ps(d + 0);
    __m128 d1 = _mm_load_ps(d + 4);
    __m128 d2 = _mm_load_ps(d + 8);
    __m128 d3 = _mm_load_ps(d + 12);
    
    // 1. Aşama: t = B^T * d (Satır bazlı hesaplama)
    __m128 t0 = _mm_sub_ps(d0, d2);
    __m128 t1 = _mm_add_ps(d1, d2);
    __m128 t2 = _mm_sub_ps(d2, d1);
    __m128 t3 = _mm_sub_ps(d1, d3);
    
    // Satırları sütunlara çevir (Transpose)
    _MM_TRANSPOSE4_PS(t0, t1, t2, t3);
    
    // 2. Aşama: V = t * B = (B^T * d) * B (Sütun bazlı hesaplama)
    __m128 v0 = _mm_sub_ps(t0, t2);
    __m128 v1 = _mm_add_ps(t1, t2);
    __m128 v2 = _mm_sub_ps(t2, t1);
    __m128 v3 = _mm_sub_ps(t1, t3);
    
    // Tekrar Transpose yaparak matrisi orijinal hizasına getir
    _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
    
    // 32-byte hizalı yazma
    _mm_store_ps(V + 0, v0);
    _mm_store_ps(V + 4, v1);
    _mm_store_ps(V + 8, v2);
    _mm_store_ps(V + 12, v3);
}

inline float fast_silu_scalar(float x) noexcept {
    float neg_x = -x;
    if (neg_x < -88.37626f) return x;
    if (neg_x >  88.37626f) return 0.0f;
    const float kLog2e = 1.44269504088896341f;
    float fx = neg_x * kLog2e;
    float n = std::round(fx);
    float f = fx - n;
    float p = 1.9875691500e-4f;
    p = p * f + 1.3981999507e-3f;
    p = p * f + 8.3334519073e-3f;
    p = p * f + 4.1665795894e-2f;
    p = p * f + 1.6666665459e-1f;
    p = p * f + 5.0000001201e-1f;
    p = p * f + 1.0f;
    int32_t exp_int = static_cast<int32_t>(n) + 127;
    exp_int <<= 23;
    float scale;
    std::memcpy(&scale, &exp_int, sizeof(float));
    return x / (1.0f + p * scale);
}

inline float apply_act(float v, int a, float slope) noexcept {
    switch (a) {
        case 1: return (v > 0.0f) ? v : 0.0f;
        case 2: return (v > 0.0f) ? v : (slope * v);
        case 3: return 1.0f / (1.0f + std::exp(-v));
        case 4: {
            const float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + std::tanh(inner));
        }
        case 5: return fast_silu_scalar(v);
        default: return v;
    }
}

inline void output_transform(const float M[16], float Y[4]) noexcept {
    const float t00 = M[0]  + M[1]  + M[2],  t01 = M[1]  - M[2]  - M[3];
    const float t10 = M[4]  + M[5]  + M[6],  t11 = M[5]  - M[6]  - M[7];
    const float t20 = M[8]  + M[9]  + M[10], t21 = M[9]  - M[10] - M[11];
    const float t30 = M[12] + M[13] + M[14], t31 = M[13] - M[14] - M[15];
    Y[0] = t00 + t10 + t20;
    Y[1] = t01 + t11 + t21;
    Y[2] = t10 - t20 - t30;
    Y[3] = t11 - t21 - t31;
}

std::shared_mutex g_cache_mtx;
std::unordered_map<const float*, std::shared_ptr<float>> g_cache;

/* Ağırlık Paketleme: _mm_malloc ile 32-byte hizalı, sıralı blok belleği */
const float* get_or_build_winograd_weights(
    const float* weight, int64_t Cin, int64_t Cout) noexcept
{
    {
        std::shared_lock<std::shared_mutex> rl(g_cache_mtx);
        auto it = g_cache.find(weight);
        if (it != g_cache.end()) return it->second.get();
    }
    std::unique_lock<std::shared_mutex> wl(g_cache_mtx);
    auto it = g_cache.find(weight);
    if (it != g_cache.end()) return it->second.get();

    int64_t Cout_pad = (Cout + CO_BLK - 1) & ~(CO_BLK - 1);
    float* u_aligned = static_cast<float*>(_mm_malloc(Cout_pad * Cin * 16 * sizeof(float), 32));
    std::memset(u_aligned, 0, Cout_pad * Cin * 16 * sizeof(float));

    for (int64_t cb = 0; cb < Cout_pad / CO_BLK; ++cb) {
        for (int64_t ci = 0; ci < Cin; ++ci) {
            for (int cc = 0; cc < CO_BLK; ++cc) {
                int64_t co = cb * CO_BLK + cc;
                if (co >= Cout) continue;

                const float* g = weight + (co * Cin + ci) * 9;
                float gmat[3][3] = {
                    {g[0], g[1], g[2]},
                    {g[3], g[4], g[5]},
                    {g[6], g[7], g[8]}
                };
                float U[4][4];
                weight_transform(gmat, U);

                float* dst = u_aligned + (cb * Cin * CO_BLK + ci * CO_BLK + cc) * 16;
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        dst[i * 4 + j] = U[i][j];
            }
        }
    }
    std::shared_ptr<float> packed(u_aligned, [](void* p) { if (p) _mm_free(p); });
    g_cache.emplace(weight, packed);
    return packed.get();
}

} /* anonymous namespace */

extern "C" {

int engine_conv2d_winograd_3x3(
    const float* input, const float* weight, const float* bias, float* output,
    int64_t N, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout, int64_t pad_h, int64_t pad_w,
    int fused_activation, float leaky_slope) noexcept
{
    const int64_t Hout = H + 2 * pad_h - 2;
    const int64_t Wout = W + 2 * pad_w - 2;
    if (Hout <= 0 || Wout <= 0 || Cin <= 0 || Cout <= 0) return 0;

    const float* U_packed = get_or_build_winograd_weights(weight, Cin, Cout);
    if (!U_packed) return 0;

    const int64_t tiles_h = (Hout + 1) / 2;
    const int64_t tiles_w = (Wout + 1) / 2;
    const int64_t total_tiles = N * tiles_h * tiles_w;

    auto& pool = engine::runtime::GlobalPool::get();
    const int64_t co_groups = (Cout + CO_BLK - 1) / CO_BLK;

    pool.parallel_for(0, total_tiles, [&](int64_t t0, int64_t t1) {
        // Thread-local bellek tahsisi (_mm_malloc ile 32-byte align garantisi)
        thread_local std::unique_ptr<float[], AlignedFree> t_V(nullptr);
        thread_local size_t t_V_cap = 0;
        if (t_V_cap < static_cast<size_t>(Cin) * 16) {
            t_V.reset(static_cast<float*>(_mm_malloc(Cin * 16 * sizeof(float), 32)));
            t_V_cap = Cin * 16;
        }
        float* __restrict V = t_V.get();

        alignas(32) float Mv[4][16];

        for (int64_t tile = t0; tile < t1; ++tile) {
            const int64_t tw = tile % tiles_w;
            const int64_t tt = tile / tiles_w;
            const int64_t th = tt % tiles_h;
            const int64_t n  = tt / tiles_h;

            /* ============ 1) V transform — BİR KEZ ============ */
            for (int64_t ci = 0; ci < Cin; ++ci) {
                alignas(32) float d[16];
                for (int i = 0; i < 4; ++i) {
                    const int64_t hi = 2 * th - pad_h + i;
                    const bool h_ok = (hi >= 0 && hi < H);
                    for (int j = 0; j < 4; ++j) {
                        const int64_t wi = 2 * tw - pad_w + j;
                        const bool ok = h_ok && (wi >= 0 && wi < W);
                        d[i * 4 + j] = ok ? input[((n * H + hi) * W + wi) * Cin + ci] : 0.0f;
                    }
                }
                input_transform_simd(d, V + ci * 16);
            }

            const int64_t ho = 2 * th, wo = 2 * tw;

            /* ============ 2) co_groups döngüsü ============ */
            for (int64_t cb = 0; cb < co_groups; ++cb) {
                const int64_t co_start = cb * CO_BLK;
                const int64_t co_cnt   = std::min<int64_t>(CO_BLK, Cout - co_start);
                const float* u_base    = U_packed + cb * Cin * CO_BLK * 16;

                __m256 m[CO_BLK][2];
                for (int cc = 0; cc < CO_BLK; ++cc) {
                    m[cc][0] = _mm256_setzero_ps();
                    m[cc][1] = _mm256_setzero_ps();
                }

                // Döngü Açılarak (Unrolling x2) FMA talimatları paralel besleniyor
                int64_t ci = 0;
                for (; ci <= Cin - 2; ci += 2) {
                    // _mm256_load_ps -> loadu'dan daha ucuz ve direkt Cache'den çeker
                    const __m256 v_lo0 = _mm256_load_ps(V + ci * 16);
                    const __m256 v_hi0 = _mm256_load_ps(V + ci * 16 + 8);
                    const __m256 v_lo1 = _mm256_load_ps(V + ci * 16 + 16);
                    const __m256 v_hi1 = _mm256_load_ps(V + ci * 16 + 24);

                    const float* u_ptr = u_base + ci * CO_BLK * 16;

                    m[0][0] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 0),  v_lo0, m[0][0]);
                    m[0][1] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 8),  v_hi0, m[0][1]);
                    m[1][0] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 16), v_lo0, m[1][0]);
                    m[1][1] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 24), v_hi0, m[1][1]);
                    m[2][0] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 32), v_lo0, m[2][0]);
                    m[2][1] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 40), v_hi0, m[2][1]);
                    m[3][0] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 48), v_lo0, m[3][0]);
                    m[3][1] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 56), v_hi0, m[3][1]);

                    m[0][0] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 64), v_lo1, m[0][0]);
                    m[0][1] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 72), v_hi1, m[0][1]);
                    m[1][0] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 80), v_lo1, m[1][0]);
                    m[1][1] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 88), v_hi1, m[1][1]);
                    m[2][0] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 96), v_lo1, m[2][0]);
                    m[2][1] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 104), v_hi1, m[2][1]);
                    m[3][0] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 112), v_lo1, m[3][0]);
                    m[3][1] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 120), v_hi1, m[3][1]);
                }
                
                // Tail (Artık) Kanal Hesabı
                for (; ci < Cin; ++ci) {
                    const __m256 v_lo = _mm256_load_ps(V + ci * 16);
                    const __m256 v_hi = _mm256_load_ps(V + ci * 16 + 8);
                    const float* u_ptr = u_base + ci * CO_BLK * 16;

                    m[0][0] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 0),  v_lo, m[0][0]);
                    m[0][1] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 8),  v_hi, m[0][1]);
                    m[1][0] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 16), v_lo, m[1][0]);
                    m[1][1] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 24), v_hi, m[1][1]);
                    m[2][0] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 32), v_lo, m[2][0]);
                    m[2][1] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 40), v_hi, m[2][1]);
                    m[3][0] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 48), v_lo, m[3][0]);
                    m[3][1] = _mm256_fmadd_ps(_mm256_load_ps(u_ptr + 56), v_hi, m[3][1]);
                }

                /* Stack'e yaz + output transform + bias + act */
                alignas(32) float Y[CO_BLK][4];
                for (int cc = 0; cc < CO_BLK; ++cc) {
                    _mm256_store_ps(Mv[cc] + 0, m[cc][0]);
                    _mm256_store_ps(Mv[cc] + 8, m[cc][1]);
                    float y[4];
                    output_transform(Mv[cc], y);
                    const float b = (bias && co_start + cc < Cout) ? bias[co_start + cc] : 0.0f;
                    Y[cc][0] = apply_act(y[0] + b, fused_activation, leaky_slope);
                    Y[cc][1] = apply_act(y[1] + b, fused_activation, leaky_slope);
                    Y[cc][2] = apply_act(y[2] + b, fused_activation, leaky_slope);
                    Y[cc][3] = apply_act(y[3] + b, fused_activation, leaky_slope);
                }

                /* ============ 3) Vektörel store ============ */
                float* base = output + ((n * Hout + ho) * Wout + wo) * Cout + co_start;

                if (co_cnt == CO_BLK) {
                    const __m128 y00 = _mm_set_ps(Y[3][0], Y[2][0], Y[1][0], Y[0][0]);
                    const __m128 y01 = _mm_set_ps(Y[3][1], Y[2][1], Y[1][1], Y[0][1]);
                    const __m128 y10 = _mm_set_ps(Y[3][2], Y[2][2], Y[1][2], Y[0][2]);
                    const __m128 y11 = _mm_set_ps(Y[3][3], Y[2][3], Y[1][3], Y[0][3]);

                    if (ho < Hout) {
                        if (wo < Wout)     _mm_storeu_ps(base, y00);
                        if (wo + 1 < Wout) _mm_storeu_ps(base + Cout, y01);
                    }
                    if (ho + 1 < Hout) {
                        if (wo < Wout)     _mm_storeu_ps(base + Wout * Cout, y10);
                        if (wo + 1 < Wout) _mm_storeu_ps(base + Wout * Cout + Cout, y11);
                    }
                } else {
                    for (int cc = 0; cc < co_cnt; ++cc) {
                        float* b2 = base + cc;
                        if (ho < Hout) {
                            if (wo < Wout)     b2[0]          = Y[cc][0];
                            if (wo + 1 < Wout) b2[Cout]       = Y[cc][1];
                        }
                        if (ho + 1 < Hout) {
                            if (wo < Wout)     b2[Wout*Cout]         = Y[cc][2];
                            if (wo + 1 < Wout) b2[Wout*Cout + Cout]  = Y[cc][3];
                        }
                    }
                }
            }
        }
    });

    return 1;
}

void engine_conv2d_winograd_clear_cache() noexcept {
    std::unique_lock<std::shared_mutex> lk(g_cache_mtx);
    g_cache.clear();
}

} /* extern "C" */