/* ============================================================================
 *  05_gemm.inc — GEMM tile + AVX2 Panel-A Packing
 *  NOT: KC ve NR burada tanımlı değil; 01_math.inc'den geliyor.
 * ========================================================================== */

/* ============================================================================
 *  AVX2 6x8 → 8x6 transpose pack
 *  A[6 rows][kc] → A_packed[kc][6] (dense, row-minor)
 *  8 k-değeri bir seferde transpoze edilir (16× daha hızlı)
 * ========================================================================== */
inline void pack_A_panel(const float* ENGINE_RESTRICT A, int64_t lda,
                         float* ENGINE_RESTRICT A_packed,
                         int64_t m_count, int64_t kc) noexcept
{
    int64_t i = 0;
    float* dst = A_packed;

    /* ---- 6-row ana gövde (AVX2) ---- */
    for (; i + 6 <= m_count; i += 6) {
        const float* r0 = A + (i + 0) * lda;
        const float* r1 = A + (i + 1) * lda;
        const float* r2 = A + (i + 2) * lda;
        const float* r3 = A + (i + 3) * lda;
        const float* r4 = A + (i + 4) * lda;
        const float* r5 = A + (i + 5) * lda;

        int64_t k = 0;
        for (; k + 7 < kc; k += 8) {
            __m256 v0 = _mm256_loadu_ps(r0 + k);
            __m256 v1 = _mm256_loadu_ps(r1 + k);
            __m256 v2 = _mm256_loadu_ps(r2 + k);
            __m256 v3 = _mm256_loadu_ps(r3 + k);
            __m256 v4 = _mm256_loadu_ps(r4 + k);
            __m256 v5 = _mm256_loadu_ps(r5 + k);

            __m256 t0 = _mm256_unpacklo_ps(v0, v1);
            __m256 t1 = _mm256_unpackhi_ps(v0, v1);
            __m256 t2 = _mm256_unpacklo_ps(v2, v3);
            __m256 t3 = _mm256_unpackhi_ps(v2, v3);
            __m256 t4 = _mm256_unpacklo_ps(v4, v5);
            __m256 t5 = _mm256_unpackhi_ps(v4, v5);

            /* Sütun 0 ve 4 */
            __m256 a = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));
            __m256 b = _mm256_shuffle_ps(t4, t4, _MM_SHUFFLE(1,0,1,0));
            /* Sütun 1 ve 5 */
            __m256 c = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));
            __m256 d = _mm256_shuffle_ps(t4, t4, _MM_SHUFFLE(3,2,3,2));
            /* Sütun 2 ve 6 */
            __m256 e = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));
            __m256 f = _mm256_shuffle_ps(t5, t5, _MM_SHUFFLE(1,0,1,0));
            /* Sütun 3 ve 7 */
            __m256 g = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));
            __m256 h = _mm256_shuffle_ps(t5, t5, _MM_SHUFFLE(3,2,3,2));

            __m256 o0 = _mm256_permute2f128_ps(a, b, 0x20);
            __m256 o1 = _mm256_permute2f128_ps(c, d, 0x20);
            __m256 o2 = _mm256_permute2f128_ps(e, f, 0x20);
            __m256 o3 = _mm256_permute2f128_ps(g, h, 0x20);
            __m256 o4 = _mm256_permute2f128_ps(a, b, 0x31);
            __m256 o5 = _mm256_permute2f128_ps(c, d, 0x31);
            __m256 o6 = _mm256_permute2f128_ps(e, f, 0x31);
            __m256 o7 = _mm256_permute2f128_ps(g, h, 0x31);

            /* 6'şar float yaz (4+2) */
            #define STORE6(dptr, v) do { \
                _mm_storeu_ps((dptr), _mm256_castps256_ps128(v)); \
                _mm_storel_pi(reinterpret_cast<__m64*>((dptr) + 4), \
                              _mm256_extractf128_ps(v, 1)); \
            } while (0)

            STORE6(dst + (k+0)*6, o0);
            STORE6(dst + (k+1)*6, o1);
            STORE6(dst + (k+2)*6, o2);
            STORE6(dst + (k+3)*6, o3);
            STORE6(dst + (k+4)*6, o4);
            STORE6(dst + (k+5)*6, o5);
            STORE6(dst + (k+6)*6, o6);
            STORE6(dst + (k+7)*6, o7);

            #undef STORE6
        }
        /* kc tail — scalar */
        for (; k < kc; ++k) {
            dst[k*6 + 0] = r0[k];
            dst[k*6 + 1] = r1[k];
            dst[k*6 + 2] = r2[k];
            dst[k*6 + 3] = r3[k];
            dst[k*6 + 4] = r4[k];
            dst[k*6 + 5] = r5[k];
        }
        dst += 6 * kc;
    }

    /* ---- 4-row tail — scalar ---- */
    for (; i + 4 <= m_count; i += 4) {
        const float* a0 = A + (i + 0) * lda;
        const float* a1 = A + (i + 1) * lda;
        const float* a2 = A + (i + 2) * lda;
        const float* a3 = A + (i + 3) * lda;
        for (int64_t k = 0; k < kc; ++k) {
            dst[k*4 + 0] = a0[k];
            dst[k*4 + 1] = a1[k];
            dst[k*4 + 2] = a2[k];
            dst[k*4 + 3] = a3[k];
        }
        dst += 4 * kc;
    }

    /* ---- 1-3 row tail — scalar ---- */
    for (; i < m_count; ++i) {
        const float* src = A + i * lda;
        for (int64_t k = 0; k < kc; ++k) dst[k] = src[k];
        dst += kc;
    }
}

/* ============================================================================
 *  gemm_tile_16 — AYNI, sadece pack artık AVX2
 * ========================================================================== */
void gemm_tile_16(const float* ENGINE_RESTRICT col_buf, int64_t K,
                  const float* ENGINE_RESTRICT W_packed,
                  int64_t n_start, int64_t n_count,
                  float* ENGINE_RESTRICT tmp_buf, int64_t ldc,
                  int64_t m_count) noexcept
{
    thread_local std::vector<float> t_packed_a;
    const size_t pack_need = static_cast<size_t>(m_count) * KC;
    if (t_packed_a.size() < pack_need) t_packed_a.resize(pack_need);
    float* A_packed = t_packed_a.data();

    for (int64_t k = 0; k < K; k += KC) {
        const int64_t kc = std::min<int64_t>(KC, K - k);
        const bool first_k = (k == 0);

        pack_A_panel(col_buf + k, K, A_packed, m_count, kc);

        for (int64_t j = 0; j < n_count; j += NR) {
            const int64_t nj = std::min<int64_t>(NR, n_count - j);
            const float* B = W_packed + (n_start + j) * K + k * 16;
            int64_t i = 0;
            float* a_ptr = A_packed;

            if (nj == NR) {
                for (; i + 6 <= m_count; i += 6) {
                    float* C = tmp_buf + i * ldc + j;
                    micro_6x16_packed(a_ptr, B, C, ldc, kc, first_k);
                    a_ptr += 6 * kc;
                }
                for (; i + 4 <= m_count; i += 4) {
                    float* C = tmp_buf + i * ldc + j;
                    micro_4x16_packed(a_ptr, B, C, ldc, kc, first_k);
                    a_ptr += 4 * kc;
                }
                for (; i < m_count; ++i) {
                    float* C_ptr = tmp_buf + i * ldc + j;
                    if (first_k) {
                        for (int64_t jj = 0; jj < NR; ++jj) C_ptr[jj] = 0.0f;
                    }
                    for (int64_t kk = 0; kk < kc; ++kk) {
                        float a_val = a_ptr[kk];
                        for (int64_t jj = 0; jj < NR; ++jj) {
                            C_ptr[jj] += a_val * B[kk * 16 + jj];
                        }
                    }
                    a_ptr += kc;
                }
            } else {
                for (; i + 6 <= m_count; i += 6) {
                    float* C_ptr = tmp_buf + i * ldc + j;
                    if (first_k) {
                        for (int64_t ii = 0; ii < 6; ++ii)
                            for (int64_t jj = 0; jj < nj; ++jj)
                                C_ptr[ii * ldc + jj] = 0.0f;
                    }
                    for (int64_t kk = 0; kk < kc; ++kk) {
                        for (int64_t ii = 0; ii < 6; ++ii) {
                            float a_val = a_ptr[kk * 6 + ii];
                            for (int64_t jj = 0; jj < nj; ++jj)
                                C_ptr[ii * ldc + jj] += a_val * B[kk * 16 + jj];
                        }
                    }
                    a_ptr += 6 * kc;
                }
                for (; i + 4 <= m_count; i += 4) {
                    float* C_ptr = tmp_buf + i * ldc + j;
                    if (first_k) {
                        for (int64_t ii = 0; ii < 4; ++ii)
                            for (int64_t jj = 0; jj < nj; ++jj)
                                C_ptr[ii * ldc + jj] = 0.0f;
                    }
                    for (int64_t kk = 0; kk < kc; ++kk) {
                        for (int64_t ii = 0; ii < 4; ++ii) {
                            float a_val = a_ptr[kk * 4 + ii];
                            for (int64_t jj = 0; jj < nj; ++jj)
                                C_ptr[ii * ldc + jj] += a_val * B[kk * 16 + jj];
                        }
                    }
                    a_ptr += 4 * kc;
                }
                for (; i < m_count; ++i) {
                    float* C_ptr = tmp_buf + i * ldc + j;
                    if (first_k) {
                        for (int64_t jj = 0; jj < nj; ++jj) C_ptr[jj] = 0.0f;
                    }
                    for (int64_t kk = 0; kk < kc; ++kk) {
                        float a_val = a_ptr[kk];
                        for (int64_t jj = 0; jj < nj; ++jj)
                            C_ptr[jj] += a_val * B[kk * 16 + jj];
                    }
                    a_ptr += kc;
                }
            }
        }
    }
}