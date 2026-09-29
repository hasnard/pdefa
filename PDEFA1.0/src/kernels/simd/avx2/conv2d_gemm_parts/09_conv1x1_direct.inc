/* ============================================================================
 *  conv2d_gemm_parts/09_conv1x1_direct.inc
 *  Ultra-Fast 1x1 Direct Conv2D for NCHW layout (Zero-Copy + Fused Epilogue)
 * ========================================================================== */

ENGINE_INLINE void conv1x1_microkernel_4x16_avx2(
    const float* ENGINE_RESTRICT input,     /* [Cin, HW] */
    const float* ENGINE_RESTRICT weights,   /* [Cout, Cin] row-major */
    float* ENGINE_RESTRICT output,          /* [Cout, HW] */
    int64_t HW, int64_t Cin,
    const float* ENGINE_RESTRICT bias,
    int fused_act, float slope,
    int64_t m_start, int64_t m_end,
    int64_t hw_start, int64_t hw_end) noexcept 
{
    const __m256 vslope = _mm256_set1_ps(slope);
    const __m256 vzero  = _mm256_setzero_ps();

    for (int64_t m = m_start; m < m_end; m += 4) {
        const float* w0 = weights + (m + 0) * Cin;
        const float* w1 = weights + (m + 1) * Cin;
        const float* w2 = weights + (m + 2) * Cin;
        const float* w3 = weights + (m + 3) * Cin;

        float* out0 = output + (m + 0) * HW;
        float* out1 = output + (m + 1) * HW;
        float* out2 = output + (m + 2) * HW;
        float* out3 = output + (m + 3) * HW;

        const __m256 b0 = bias ? _mm256_set1_ps(bias[m + 0]) : vzero;
        const __m256 b1 = bias ? _mm256_set1_ps(bias[m + 1]) : vzero;
        const __m256 b2 = bias ? _mm256_set1_ps(bias[m + 2]) : vzero;
        const __m256 b3 = bias ? _mm256_set1_ps(bias[m + 3]) : vzero;

        for (int64_t hw = hw_start; hw < hw_end; hw += 16) {
            /* 8 Akümülatör: 4 Satır (Cout) x 2 Vektör (16 piksel) */
            __m256 acc00 = b0; __m256 acc01 = b0;
            __m256 acc10 = b1; __m256 acc11 = b1;
            __m256 acc20 = b2; __m256 acc21 = b2;
            __m256 acc30 = b3; __m256 acc31 = b3;

            /* K (Cin) indirgemesi: 2x unrolled */
            int64_t k = 0;
            for (; k <= Cin - 2; k += 2) {
                const float* in_k0 = input + (k + 0) * HW + hw;
                const float* in_k1 = input + (k + 1) * HW + hw;

                /* Cin = k */
                __m256 in0_0 = _mm256_loadu_ps(in_k0 + 0);
                __m256 in0_1 = _mm256_loadu_ps(in_k0 + 8);

                __m256 w0_0 = _mm256_set1_ps(w0[k]);
                __m256 w1_0 = _mm256_set1_ps(w1[k]);
                acc00 = _mm256_fmadd_ps(w0_0, in0_0, acc00);
                acc01 = _mm256_fmadd_ps(w0_0, in0_1, acc01);
                acc10 = _mm256_fmadd_ps(w1_0, in0_0, acc10);
                acc11 = _mm256_fmadd_ps(w1_0, in0_1, acc11);

                __m256 w2_0 = _mm256_set1_ps(w2[k]);
                __m256 w3_0 = _mm256_set1_ps(w3[k]);
                acc20 = _mm256_fmadd_ps(w2_0, in0_0, acc20);
                acc21 = _mm256_fmadd_ps(w2_0, in0_1, acc21);
                acc30 = _mm256_fmadd_ps(w3_0, in0_0, acc30);
                acc31 = _mm256_fmadd_ps(w3_0, in0_1, acc31);

                /* Cin = k + 1 */
                __m256 in1_0 = _mm256_loadu_ps(in_k1 + 0);
                __m256 in1_1 = _mm256_loadu_ps(in_k1 + 8);

                __m256 w0_1 = _mm256_set1_ps(w0[k + 1]);
                __m256 w1_1 = _mm256_set1_ps(w1[k + 1]);
                acc00 = _mm256_fmadd_ps(w0_1, in1_0, acc00);
                acc01 = _mm256_fmadd_ps(w0_1, in1_1, acc01);
                acc10 = _mm256_fmadd_ps(w1_1, in1_0, acc10);
                acc11 = _mm256_fmadd_ps(w1_1, in1_1, acc11);

                __m256 w2_1 = _mm256_set1_ps(w2[k + 1]);
                __m256 w3_1 = _mm256_set1_ps(w3[k + 1]);
                acc20 = _mm256_fmadd_ps(w2_1, in1_0, acc20);
                acc21 = _mm256_fmadd_ps(w2_1, in1_1, acc21);
                acc30 = _mm256_fmadd_ps(w3_1, in1_0, acc30);
                acc31 = _mm256_fmadd_ps(w3_1, in1_1, acc31);
            }

            for (; k < Cin; ++k) {
                const float* in_k = input + k * HW + hw;
                __m256 in_0 = _mm256_loadu_ps(in_k + 0);
                __m256 in_1 = _mm256_loadu_ps(in_k + 8);

                acc00 = _mm256_fmadd_ps(_mm256_set1_ps(w0[k]), in_0, acc00);
                acc01 = _mm256_fmadd_ps(_mm256_set1_ps(w0[k]), in_1, acc01);
                acc10 = _mm256_fmadd_ps(_mm256_set1_ps(w1[k]), in_0, acc10);
                acc11 = _mm256_fmadd_ps(_mm256_set1_ps(w1[k]), in_1, acc11);
                acc20 = _mm256_fmadd_ps(_mm256_set1_ps(w2[k]), in_0, acc20);
                acc21 = _mm256_fmadd_ps(_mm256_set1_ps(w2[k]), in_1, acc21);
                acc30 = _mm256_fmadd_ps(_mm256_set1_ps(w3[k]), in_0, acc30);
                acc31 = _mm256_fmadd_ps(_mm256_set1_ps(w3[k]), in_1, acc31);
            }

            /* Fused Epilogue: Tek geçişte aktivasyon ve Store */
            _mm256_storeu_ps(out0 + hw + 0, apply_act_avx2(acc00, fused_act, vslope, vzero));
            _mm256_storeu_ps(out0 + hw + 8, apply_act_avx2(acc01, fused_act, vslope, vzero));
            _mm256_storeu_ps(out1 + hw + 0, apply_act_avx2(acc10, fused_act, vslope, vzero));
            _mm256_storeu_ps(out1 + hw + 8, apply_act_avx2(acc11, fused_act, vslope, vzero));
            _mm256_storeu_ps(out2 + hw + 0, apply_act_avx2(acc20, fused_act, vslope, vzero));
            _mm256_storeu_ps(out2 + hw + 8, apply_act_avx2(acc21, fused_act, vslope, vzero));
            _mm256_storeu_ps(out3 + hw + 0, apply_act_avx2(acc30, fused_act, vslope, vzero));
            _mm256_storeu_ps(out3 + hw + 8, apply_act_avx2(acc31, fused_act, vslope, vzero));
        }
    }
}

void engine_conv2d_1x1_direct_avx2(
    const float* ENGINE_RESTRICT input,
    const float* ENGINE_RESTRICT weights,
    const float* ENGINE_RESTRICT bias,
    float* ENGINE_RESTRICT output,
    int64_t Cin, int64_t Cout, int64_t H, int64_t W,
    int fused_act, float slope) noexcept
{
    const int64_t HW = H * W;

    /* 40x40 (1600) ve 80x80 (6400) için iş parçacığı (Thread-Pool) paralel bölütleme */
    /* Kanallar (Cout) 4'ün katları halinde thread'lere dağıtılır */
    const int64_t m_blocks = Cout / 4;

    /* ThreadPool çağrısı (veya OpenMP parallel for) */
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < m_blocks; ++b) {
        int64_t m_start = b * 4;
        int64_t m_end   = m_start + 4;
        conv1x1_microkernel_4x16_avx2(
            input, weights, output,
            HW, Cin, bias, fused_act, slope,
            m_start, m_end, 0, HW
        );
    }
}