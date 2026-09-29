/* ============================================================================
 *  01_math.inc — AVX2 matematik (NHWC sürümü)
 *  transpose8x8_ps KALDIRILDI — output NHWC'de doğal olarak düz.
 * ========================================================================== */

constexpr int64_t NR = 16;
constexpr int64_t KC = 128;

inline __m256 exp_ps_avx2(__m256 x) noexcept {
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

inline __m256 sigmoid_ps_avx2(__m256 x) noexcept {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 e = exp_ps_avx2(neg_x);
    return _mm256_div_ps(one, _mm256_add_ps(one, e));
}

inline __m256 silu_ps_avx2(__m256 x) noexcept {
    return _mm256_mul_ps(x, sigmoid_ps_avx2(x));
}