/* ============================================================================
 *  src/kernels/simd/avx2/activation_avx2.cpp
 *  Engine-AI — AVX2 activation kernels
 *
 *  8 float aynı anda (256-bit YMM).
 *  Yüksek hassasiyetli (Cephes minimax, max err < 1e-7) AVX2 + FMA exp.
 *  Sadece -mavx2 -mfma ile derlenir.
 * ========================================================================== */

#include <immintrin.h>
#include <cstdint>
#include <cmath>

namespace {

/* ============================================================================
 *  exp(x) AVX2 (8-wide) — Cody-Waite Range Reduction + Cephes Minimax
 * ========================================================================== */
inline __m256 exp_ps_avx2(__m256 x) noexcept {
    const __m256 kHiBound = _mm256_set1_ps(88.3762626647949f);
    const __m256 kLoBound = _mm256_set1_ps(-88.3762626647949f);

    // Taşma ve sıfıra düşme (underflow/overflow) sınırlandırması
    x = _mm256_min_ps(x, kHiBound);
    x = _mm256_max_ps(x, kLoBound);

    // n = round(x * log2(e))
    const __m256 kLog2e = _mm256_set1_ps(1.44269504088896341f);
    __m256 fx = _mm256_mul_ps(x, kLog2e);
    __m256 n  = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Cody-Waite hassas argüman indirgemesi: y = x - n * ln(2)
    // ln(2) float duyarlılık kaybını önlemek için C1 + C2 olarak bölünür:
    // C1 = 0.693359375 (355/512, float içinde tam temsil edilir)
    // C2 = -2.121944400546905827679e-4
    const __m256 kC1 = _mm256_set1_ps(0.693359375f);
    const __m256 kC2 = _mm256_set1_ps(-2.121944400546905827679e-4f);

    __m256 y = _mm256_fnmadd_ps(n, kC1, x); // y = x - n * C1
    y = _mm256_fnmadd_ps(n, kC2, y);        // y = y - n * C2

    // Cephes minimax polinomu: [-ln(2)/2, ln(2)/2] aralığında e^y yaklaşımı
    const __m256 c0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 c1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 c2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 c3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 c4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 c5 = _mm256_set1_ps(5.0000001201e-1f);

    // P(y) = ((((c0*y + c1)*y + c2)*y + c3)*y + c4)*y + c5
    __m256 p = _mm256_fmadd_ps(c0, y, c1);
    p = _mm256_fmadd_ps(p, y, c2);
    p = _mm256_fmadd_ps(p, y, c3);
    p = _mm256_fmadd_ps(p, y, c4);
    p = _mm256_fmadd_ps(p, y, c5);

    // e^y = 1 + y + y^2 * P(y)
    __m256 y2 = _mm256_mul_ps(y, y);
    p = _mm256_fmadd_ps(p, y2, y);
    p = _mm256_add_ps(p, _mm256_set1_ps(1.0f));

    // 2^n ile ölçekleme (bit shift ile doğrudan üs alanına yazma)
    __m256i n_int = _mm256_cvtps_epi32(n);
    __m256 scale = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(n_int, _mm256_set1_epi32(127)), 23)
    );

    return _mm256_mul_ps(p, scale);
}

inline __m256 sigmoid_ps_avx2(__m256 x) noexcept {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 e = exp_ps_avx2(neg_x);
    return _mm256_div_ps(one, _mm256_add_ps(one, e));
}

inline __m256 tanh_ps_avx2(__m256 x) noexcept {
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 x2 = _mm256_mul_ps(x, two);
    __m256 neg_x2 = _mm256_sub_ps(_mm256_setzero_ps(), x2);
    __m256 e = exp_ps_avx2(neg_x2);
    __m256 denom = _mm256_add_ps(one, e);
    __m256 sig = _mm256_div_ps(one, denom);
    return _mm256_sub_ps(_mm256_mul_ps(two, sig), one);
}

} /* anonymous namespace */

/* ============================================================================
 *  PUBLIC AVX2 KERNELS — extern "C"
 * ========================================================================== */
extern "C" {

void engine_activation_relu_avx2(float* data, int64_t n) noexcept {
    const __m256 zero = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        v = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(data + i, v);
    }
    for (; i < n; ++i) {
        const float v = data[i];
        data[i] = v > 0.0f ? v : 0.0f;
    }
}

void engine_activation_sigmoid_avx2(float* data, int64_t n) noexcept {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        v = sigmoid_ps_avx2(v);
        _mm256_storeu_ps(data + i, v);
    }
    for (; i < n; ++i) {
        const float x = data[i];
        data[i] = 1.0f / (1.0f + std::exp(-x));
    }
}

void engine_activation_tanh_avx2(float* data, int64_t n) noexcept {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        v = tanh_ps_avx2(v);
        _mm256_storeu_ps(data + i, v);
    }
    for (; i < n; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

void engine_activation_silu_avx2(float* data, int64_t n) noexcept {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        __m256 sig = sigmoid_ps_avx2(v);
        v = _mm256_mul_ps(v, sig);
        _mm256_storeu_ps(data + i, v);
    }
    for (; i < n; ++i) {
        const float x = data[i];
        data[i] = x / (1.0f + std::exp(-x));
    }
}

void engine_activation_gelu_avx2(float* data, int64_t n) noexcept {
    const __m256 kAlpha = _mm256_set1_ps(0.7978845608f);
    const __m256 kBeta  = _mm256_set1_ps(0.044715f);
    const __m256 kHalf  = _mm256_set1_ps(0.5f);
    const __m256 kOne   = _mm256_set1_ps(1.0f);

    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(data + i);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 inner = _mm256_fmadd_ps(kBeta, x3, x);
        inner = _mm256_mul_ps(kAlpha, inner);
        __m256 t = tanh_ps_avx2(inner);
        __m256 r = _mm256_mul_ps(kHalf, x);
        r = _mm256_mul_ps(r, _mm256_add_ps(kOne, t));
        _mm256_storeu_ps(data + i, r);
    }
    for (; i < n; ++i) {
        const float x = data[i];
        const float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        data[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void engine_activation_leaky_relu_avx2(float* data, int64_t n, float slope) noexcept {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 kSlope = _mm256_set1_ps(slope);

    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        __m256 neg = _mm256_mul_ps(v, kSlope);
        __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OS);
        __m256 r = _mm256_blendv_ps(neg, v, mask);
        _mm256_storeu_ps(data + i, r);
    }
    for (; i < n; ++i) {
        const float v = data[i];
        data[i] = v > 0.0f ? v : (slope * v);
    }
}

} /* extern "C" */