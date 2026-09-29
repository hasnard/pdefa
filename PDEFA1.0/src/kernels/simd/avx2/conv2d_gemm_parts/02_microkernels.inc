/* ============================================================================
 *  AVX2 microkernel'ler: 6x16 ve 4x16
 *  Optimizasyon: B (weight) icin 4-iterasyon ileri prefetch ve Panel-A Packing
 * ========================================================================== */

inline void micro_6x16_packed(
    const float* ENGINE_RESTRICT A_packed,
    const float* ENGINE_RESTRICT B,
    float* ENGINE_RESTRICT C, int64_t ldc,
    int64_t kc, bool first_k) noexcept
{
    __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;

    if (first_k) {
        c00 = _mm256_setzero_ps(); c01 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps(); c11 = _mm256_setzero_ps();
        c20 = _mm256_setzero_ps(); c21 = _mm256_setzero_ps();
        c30 = _mm256_setzero_ps(); c31 = _mm256_setzero_ps();
        c40 = _mm256_setzero_ps(); c41 = _mm256_setzero_ps();
        c50 = _mm256_setzero_ps(); c51 = _mm256_setzero_ps();
    } else {
        c00 = _mm256_loadu_ps(C + 0*ldc + 0); c01 = _mm256_loadu_ps(C + 0*ldc + 8);
        c10 = _mm256_loadu_ps(C + 1*ldc + 0); c11 = _mm256_loadu_ps(C + 1*ldc + 8);
        c20 = _mm256_loadu_ps(C + 2*ldc + 0); c21 = _mm256_loadu_ps(C + 2*ldc + 8);
        c30 = _mm256_loadu_ps(C + 3*ldc + 0); c31 = _mm256_loadu_ps(C + 3*ldc + 8);
        c40 = _mm256_loadu_ps(C + 4*ldc + 0); c41 = _mm256_loadu_ps(C + 4*ldc + 8);
        c50 = _mm256_loadu_ps(C + 5*ldc + 0); c51 = _mm256_loadu_ps(C + 5*ldc + 8);
    }

    int64_t k = 0;
    for (; k + 1 < kc; k += 2) {
        /* Optimizasyon: B ve A'yı 4 iterasyon ileriye prefetch et */
        if (k + 4 < kc) {
            _mm_prefetch(reinterpret_cast<const char*>(B + (k + 4) * 16), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(A_packed + (k + 4) * 6), _MM_HINT_T0);
        }

        __m256 b0 = _mm256_loadu_ps(B + k*16 + 0);
        __m256 b1 = _mm256_loadu_ps(B + k*16 + 8);

        __m256 a;
        a = _mm256_broadcast_ss(A_packed + k*6 + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(A_packed + k*6 + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(A_packed + k*6 + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(A_packed + k*6 + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(A_packed + k*6 + 4); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(A_packed + k*6 + 5); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);

        b0 = _mm256_loadu_ps(B + (k+1)*16 + 0);
        b1 = _mm256_loadu_ps(B + (k+1)*16 + 8);

        a = _mm256_broadcast_ss(A_packed + (k+1)*6 + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(A_packed + (k+1)*6 + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(A_packed + (k+1)*6 + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(A_packed + (k+1)*6 + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(A_packed + (k+1)*6 + 4); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(A_packed + (k+1)*6 + 5); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
    }
    if (k < kc) {
        __m256 b0 = _mm256_loadu_ps(B + k*16 + 0);
        __m256 b1 = _mm256_loadu_ps(B + k*16 + 8);
        __m256 a;
        a = _mm256_broadcast_ss(A_packed + k*6 + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(A_packed + k*6 + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(A_packed + k*6 + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(A_packed + k*6 + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(A_packed + k*6 + 4); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(A_packed + k*6 + 5); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
    }

    _mm256_storeu_ps(C + 0*ldc + 0, c00); _mm256_storeu_ps(C + 0*ldc + 8, c01);
    _mm256_storeu_ps(C + 1*ldc + 0, c10); _mm256_storeu_ps(C + 1*ldc + 8, c11);
    _mm256_storeu_ps(C + 2*ldc + 0, c20); _mm256_storeu_ps(C + 2*ldc + 8, c21);
    _mm256_storeu_ps(C + 3*ldc + 0, c30); _mm256_storeu_ps(C + 3*ldc + 8, c31);
    _mm256_storeu_ps(C + 4*ldc + 0, c40); _mm256_storeu_ps(C + 4*ldc + 8, c41);
    _mm256_storeu_ps(C + 5*ldc + 0, c50); _mm256_storeu_ps(C + 5*ldc + 8, c51);
}

inline void micro_4x16_packed(
    const float* ENGINE_RESTRICT A_packed,
    const float* ENGINE_RESTRICT B,
    float* ENGINE_RESTRICT C, int64_t ldc,
    int64_t kc, bool first_k) noexcept
{
    __m256 c00, c01, c10, c11, c20, c21, c30, c31;

    if (first_k) {
        c00 = _mm256_setzero_ps(); c01 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps(); c11 = _mm256_setzero_ps();
        c20 = _mm256_setzero_ps(); c21 = _mm256_setzero_ps();
        c30 = _mm256_setzero_ps(); c31 = _mm256_setzero_ps();
    } else {
        c00 = _mm256_loadu_ps(C + 0*ldc + 0); c01 = _mm256_loadu_ps(C + 0*ldc + 8);
        c10 = _mm256_loadu_ps(C + 1*ldc + 0); c11 = _mm256_loadu_ps(C + 1*ldc + 8);
        c20 = _mm256_loadu_ps(C + 2*ldc + 0); c21 = _mm256_loadu_ps(C + 2*ldc + 8);
        c30 = _mm256_loadu_ps(C + 3*ldc + 0); c31 = _mm256_loadu_ps(C + 3*ldc + 8);
    }
    
    for (int64_t k = 0; k < kc; ++k) {
        if (k + 4 < kc) {
            _mm_prefetch(reinterpret_cast<const char*>(B + (k + 4) * 16), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(A_packed + (k + 4) * 4), _MM_HINT_T0);
        }
        __m256 b0 = _mm256_loadu_ps(B + k*16 + 0);
        __m256 b1 = _mm256_loadu_ps(B + k*16 + 8);
        
        __m256 a;
        a = _mm256_broadcast_ss(A_packed + k*4 + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(A_packed + k*4 + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(A_packed + k*4 + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(A_packed + k*4 + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
    }
    
    _mm256_storeu_ps(C + 0*ldc + 0, c00); _mm256_storeu_ps(C + 0*ldc + 8, c01);
    _mm256_storeu_ps(C + 1*ldc + 0, c10); _mm256_storeu_ps(C + 1*ldc + 8, c11);
    _mm256_storeu_ps(C + 2*ldc + 0, c20); _mm256_storeu_ps(C + 2*ldc + 8, c21);
    _mm256_storeu_ps(C + 3*ldc + 0, c30); _mm256_storeu_ps(C + 3*ldc + 8, c31);
}