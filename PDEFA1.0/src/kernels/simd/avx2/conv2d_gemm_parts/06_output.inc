/* ============================================================================
 *  06_output.inc — NHWC output (transpose YOK)
 *  tmp_buf [M][ldc] = [HW][Cout_padded]
 *  output  [HW][Cout]  — aynı layout, sadece padding'i kırp
 * ========================================================================== */

void write_output_tile_fast(const float* ENGINE_RESTRICT tmp_buf, int64_t ldc,
                            float* ENGINE_RESTRICT output,
                            int64_t n_start, int64_t n_count,
                            int64_t m_start, int64_t m_count,
                            const float* ENGINE_RESTRICT bias,
                            int64_t Cout,
                            int act, float slope) noexcept
{
    for (int64_t i = 0; i < m_count; ++i) {
        const float* src_row = tmp_buf + i * ldc;
        float* dst_row = output + (m_start + i) * Cout + n_start;

        int64_t c = 0;
        for (; c + 7 < n_count; c += 8) {
            __m256 v = _mm256_loadu_ps(src_row + c);
            if (bias) {
                __m256 b_vec = _mm256_loadu_ps(bias + n_start + c);
                v = _mm256_add_ps(v, b_vec);
            }
            if (act == 1) {
                v = _mm256_max_ps(v, _mm256_setzero_ps());
            } else if (act == 2) {
                __m256 zero = _mm256_setzero_ps();
                __m256 s = _mm256_mul_ps(v, _mm256_set1_ps(slope));
                v = _mm256_blendv_ps(s, v, _mm256_cmp_ps(v, zero, _CMP_GT_OS));
            } else if (act == 5) {
                v = silu_ps_avx2(v);
            }
            _mm256_storeu_ps(dst_row + c, v);
        }
        for (; c < n_count; ++c) {
            const float b = bias ? bias[n_start + c] : 0.0f;
            float v = src_row[c] + b;
            if (act == 1)      v = (v > 0.0f) ? v : 0.0f;
            else if (act == 2) v = (v > 0.0f) ? v : (slope * v);
            else if (act == 5) v = v / (1.0f + std::exp(-v));
            dst_row[c] = v;
        }
    }
}