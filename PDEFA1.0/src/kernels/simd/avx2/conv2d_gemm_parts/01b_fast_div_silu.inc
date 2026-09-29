/* ============================================================================
 *  01b_fast_div_silu.inc — Newton-Raphson hizli bolme
 *  _mm256_div_ps (latency ~11-14 cyc, throughput ~4-5) yerine
 *  _mm256_rcp_ps + 1 adim NR (latency ~7 cyc, throughput ~2).
 *  1 NR adimi = ~22-bit dogruluk → SiLU icin fazlasiyla yeterli.
 *  Risk: dusuk. Eski silu_ps_avx2 fallback olarak korunur.
 * ========================================================================== */

inline __m256 fast_reciprocal_ps(__m256 d) noexcept {
    const __m256 two = _mm256_set1_ps(2.0f);
    __m256 rcp = _mm256_rcp_ps(d);                        /* ~1.5e-3 hata */
    /* x1 = x0 * (2 - d * x0) */
    return _mm256_mul_ps(rcp, _mm256_fnmadd_ps(d, rcp, two)); /* < 1e-7 hata */
}

inline __m256 sigmoid_fast_avx2(__m256 x) noexcept {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 e = exp_ps_avx2(_mm256_sub_ps(_mm256_setzero_ps(), x));
    __m256 denom = _mm256_add_ps(one, e);
    return fast_reciprocal_ps(denom);
}

inline __m256 silu_fast_avx2(__m256 x) noexcept {
    return _mm256_mul_ps(x, sigmoid_fast_avx2(x));
}