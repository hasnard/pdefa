/* ============================================================================
 *  04_im2col.inc — NHWC im2col
 *  Input layout : [H][W][Cin] — kanal contiguous (memcpy-friendly)
 *  Output layout: col_buf[M][K], K = KH*KW*Cin
 *  K indexing   : k = (kh*KW + kw)*Cin + ci
 * ========================================================================== */

void im2col_tile_nhwc(const float* ENGINE_RESTRICT input,
                      float* ENGINE_RESTRICT col_buf,
                      int64_t m_start, int64_t m_count,
                      int64_t Cin, int64_t H, int64_t W,
                      int64_t KH, int64_t KW,
                      int64_t stride_h, int64_t stride_w,
                      int64_t pad_h, int64_t pad_w,
                      int64_t dil_h, int64_t dil_w,
                      int64_t Wout, int64_t K) noexcept
{
    const size_t cin_bytes = static_cast<size_t>(Cin) * sizeof(float);

    for (int64_t m = 0; m < m_count; ++m) {
        const int64_t m_abs = m_start + m;
        const int64_t ho = m_abs / Wout;
        const int64_t wo = m_abs % Wout;
        const int64_t h_base = ho * stride_h - pad_h;
        const int64_t w_base = wo * stride_w - pad_w;

        float* col_row = col_buf + m * K;
        int64_t k_off = 0;

        for (int64_t kh = 0; kh < KH; ++kh) {
            const int64_t hi = h_base + kh * dil_h;
            const bool h_ok = (hi >= 0 && hi < H);
            for (int64_t kw = 0; kw < KW; ++kw) {
                const int64_t wi = w_base + kw * dil_w;
float* dst = col_row + k_off;
if (h_ok && wi >= 0 && wi < W) {
    const float* src = input + (hi * W + wi) * Cin;
    if (Cin == 3) {
        dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
    } else if (Cin == 4) {
        _mm_storeu_ps(dst, _mm_loadu_ps(src));
    } else {
        std::memcpy(dst, src, cin_bytes);
    }
} else {
    if (Cin == 3) { dst[0] = 0.f; dst[1] = 0.f; dst[2] = 0.f; }
    else if (Cin == 4) { _mm_storeu_ps(dst, _mm_setzero_ps()); }
    else { std::memset(dst, 0, cin_bytes); }
}
k_off += Cin;

            }
        }
    }
}