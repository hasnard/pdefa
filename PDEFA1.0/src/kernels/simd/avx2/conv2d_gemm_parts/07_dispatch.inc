/* ============================================================================
 *  07_dispatch.inc — engine_conv2d_gemm (v4 — SADE)
 *
 *  v4 değişiklikleri:
 *    [1] 1x1 stride-1 pad-0 → ZERO-COPY, doğrudan engine_matmul_avx2
 *        Custom microkernel (8x8 fused) kaldırıldı — matmul BLIS-blocking
 *        zaten W_kmn [K,N] layout'unda cache-optimal.
 *    [2] Epilogue (bias + activation) matmul çıktısı cache-hot iken tek geçişte.
 *    [3] Genel konv (3x3, 5x5, 1x1 stride-2, ...) davranışı DEĞİŞMEDİ.
 *    [4] Fallback (K<64): custom gemm_tile_16 backend korundu.
 * ========================================================================== */

namespace {

/* ============================================================================
 *  Fused Epilogue: bias + activation
 *  matmul çıktısı hâlâ L1/L2'de sıcakken uygulanır → ekstra cache round-trip yok.
 * ========================================================================== */
ENGINE_INLINE void conv_epilogue_bias_act(
    float* ENGINE_RESTRICT output,
    const float* ENGINE_RESTRICT bias,
    int64_t M_total, int64_t Cout,
    int fused_activation, float leaky_slope) noexcept
{
    if (!bias && fused_activation == 0) return;

    auto& pool = engine::runtime::GlobalPool::get();
    const __m256 vzero  = _mm256_setzero_ps();
    const __m256 vslope = _mm256_set1_ps(leaky_slope);

    pool.parallel_for(0, M_total, [&](int64_t m0, int64_t m1) {
        for (int64_t m = m0; m < m1; ++m) {
            float* ENGINE_RESTRICT row = output + m * Cout;
            int64_t c = 0;

            if (bias) {
                for (; c + 7 < Cout; c += 8) {
                    __m256 v = _mm256_loadu_ps(row + c);
                    v = _mm256_add_ps(v, _mm256_loadu_ps(bias + c));
                    v = apply_act_avx2(v, fused_activation, vslope, vzero);
                    _mm256_storeu_ps(row + c, v);
                }
            } else {
                for (; c + 7 < Cout; c += 8) {
                    __m256 v = _mm256_loadu_ps(row + c);
                    v = apply_act_avx2(v, fused_activation, vslope, vzero);
                    _mm256_storeu_ps(row + c, v);
                }
            }
            for (; c < Cout; ++c) {
                float v = row[c] + (bias ? bias[c] : 0.0f);
                row[c] = apply_act_local(v, fused_activation, leaky_slope);
            }
        }
    });
}

} // namespace


int engine_conv2d_gemm(
    const float* input, const float* weight, const float* bias, float* output,
    int64_t N_batch, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout, int64_t KH, int64_t KW,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w, int fused_activation, float leaky_slope) noexcept
{
    if (N_batch != 1 || Cin <= 0 || Cout <= 0 || H <= 0 || W <= 0 ||
        KH <= 0 || KW <= 0) return 0;

    const int64_t Hout = (H + 2 * pad_h - (KH - 1) * dil_h - 1) / stride_h + 1;
    const int64_t Wout = (W + 2 * pad_w - (KW - 1) * dil_w - 1) / stride_w + 1;
    if (Hout <= 0 || Wout <= 0) return 0;

    const int64_t K = Cin * KH * KW;
    const int64_t M_total = Hout * Wout;

    /* ========================================================================
     *  [1] 1x1 stride-1 pad-0 → ZERO-COPY matmul
     *
     *  input zaten [M_total, Cin] layout'unda. im2col GEREKSİZ.
     *  W_kmn = [Cin, Cout] transpoze edilmiş ağırlık (önceden cache'lenmiş).
     *
     *  engine_matmul_avx2 BLIS tipi M/N/K blocking yapar, W tek seferde
     *  L2'ye pinlenir, tüm M boyunca yeniden kullanılır.
     * ======================================================================== */
    const bool is_1x1_s1 = (KH == 1 && KW == 1 &&
                            stride_h == 1 && stride_w == 1 &&
                            pad_h == 0 && pad_w == 0 &&
                            dil_h == 1 && dil_w == 1);

    if (is_1x1_s1) {
        const float* W_kmn = get_weight_kmn(weight, K, Cout, Cin, KH, KW);
        if (!W_kmn) return 0;

        /* output[M,Cout] = input[M,Cin] @ W_kmn[Cin,Cout] */
        engine_matmul_avx2(input, W_kmn, output, M_total, Cout, Cin);

        /* Epilogue: matmul çıktısı cache-hot iken derhal uygula */
        conv_epilogue_bias_act(output, bias, M_total, Cout,
                               fused_activation, leaky_slope);
        return 1;
    }

    /* ========================================================================
     *  [2] Genel konv (3x3, 5x5, 1x1 stride-2, ...) → im2col + matmul
     *
     *  Eşik K>=64:
     *    Conv1 (K=27)  → custom gemm fallback
     *    Conv2 (K=144) → matmul
     *    Conv3 (K=288) → matmul
     *    Conv4 (K=576) → matmul
     *    Conv5 (K=1152) → matmul
     *    Conv6 (K=2304) → matmul
     * ======================================================================== */
    const bool use_matmul = (K >= 64 && Cout <= 512);

    if (use_matmul) {
        const float* W_kmn = get_weight_kmn(weight, K, Cout, Cin, KH, KW);
        if (!W_kmn) return 0;

        auto& pool = engine::runtime::GlobalPool::get();

        static thread_local std::vector<float> t_col;
        const size_t col_need = static_cast<size_t>(M_total) * static_cast<size_t>(K);
        if (t_col.size() < col_need) t_col.resize(col_need);
        float* col_buf = t_col.data();

        /* --- 2a) im2col: paralel, büyük tile ile --- */
        {
            constexpr int64_t MC_IM2COL = 512;
            const int64_t tiles = (M_total + MC_IM2COL - 1) / MC_IM2COL;
            pool.parallel_for(0, tiles, [&](int64_t t0, int64_t t1) {
                for (int64_t t = t0; t < t1; ++t) {
                    const int64_t m_start = t * MC_IM2COL;
                    const int64_t m_count = std::min<int64_t>(MC_IM2COL, M_total - m_start);
                    im2col_tile_nhwc(input, col_buf + m_start * K,
                                     m_start, m_count,
                                     Cin, H, W, KH, KW,
                                     stride_h, stride_w, pad_h, pad_w,
                                     dil_h, dil_w, Wout, K);
                }
            });
        }

        /* --- 2b) matmul: BLIS-blocked --- */
        engine_matmul_avx2(col_buf, W_kmn, output, M_total, Cout, K);

        /* --- 2c) Epilogue: cache-hot --- */
        conv_epilogue_bias_act(output, bias, M_total, Cout,
                               fused_activation, leaky_slope);
        return 1;
    }

    /* ========================================================================
     *  [3] Fallback: custom gemm_tile_16 backend
     *      Yalnızca K < 64 durumları için (Conv1 stem K=27 gibi).
     * ======================================================================== */
    const std::vector<float>* packed_vec = get_packed_weight_16(weight, K, Cout, Cin, KH, KW);
    if (!packed_vec) return 0;
    const float* W_packed = packed_vec->data();

    auto& pool = engine::runtime::GlobalPool::get();

    /* KÜÇÜK KATMANLAR (M_total <= 256) */
    if (M_total <= 256) {
        static thread_local std::vector<float> s_small_col;
        const size_t col_need = static_cast<size_t>(M_total) * static_cast<size_t>(K);
        if (s_small_col.size() < col_need) s_small_col.resize(col_need);
        float* col_ptr = s_small_col.data();

        im2col_tile_nhwc(input, col_ptr, 0, M_total,
                         Cin, H, W, KH, KW,
                         stride_h, stride_w, pad_h, pad_w,
                         dil_h, dil_w, Wout, K);

        constexpr int64_t NC_SMALL = 16;
        const int64_t n_tiles = (Cout + NC_SMALL - 1) / NC_SMALL;
        pool.parallel_for(0, n_tiles, [&](int64_t t0, int64_t t1) {
            thread_local std::vector<float> t_tmp_buf;
            const size_t tmp_need = static_cast<size_t>(M_total) * NC_SMALL;
            if (t_tmp_buf.size() < tmp_need) t_tmp_buf.resize(tmp_need);
            float* tmp_buf = t_tmp_buf.data();
            for (int64_t nt = t0; nt < t1; ++nt) {
                const int64_t n_start = nt * NC_SMALL;
                const int64_t n_count = std::min<int64_t>(NC_SMALL, Cout - n_start);
                gemm_tile_16(col_ptr, K, W_packed, n_start, n_count,
                             tmp_buf, n_count, M_total);
                write_output_tile_fast(tmp_buf, n_count, output,
                                       n_start, n_count, 0, M_total,
                                       bias, Cout,
                                       fused_activation, leaky_slope);
            }
        });
        return 1;
    }

    /* BÜYÜK KATMANLAR */
    int64_t MC;
    {
        const size_t l2_bytes = engine::runtime::Cpu::cache().l2;
        const int64_t k_bytes = K * static_cast<int64_t>(sizeof(float));
        if (l2_bytes > 0 && k_bytes > 0) {
            const int64_t budget = static_cast<int64_t>(l2_bytes) * 7 / 10;
            MC = budget / k_bytes;
            MC = std::clamp<int64_t>(MC, 16, 256);
        } else {
            if (K <= 128)        MC = 128;
            else if (K <= 256)   MC = 96;
            else if (K <= 512)   MC = 64;
            else if (K <= 1024)  MC = 32;
            else                 MC = 16;
        }
    }

    int64_t NC = 128;
    if (Cout < 128) NC = ((Cout + 15) / 16) * 16;
    if (NC > Cout) NC = Cout;

    const int64_t m_tiles = (M_total + MC - 1) / MC;

    pool.parallel_for(0, m_tiles, [&, MC, NC](int64_t t_start, int64_t t_end) {
        thread_local std::vector<float> t_col_buf;
        thread_local std::vector<float> t_tmp_buf;

        const size_t col_need = static_cast<size_t>(MC) * static_cast<size_t>(K);
        const size_t tmp_need = static_cast<size_t>(MC) * static_cast<size_t>(NC);
        if (t_col_buf.size() < col_need) t_col_buf.resize(col_need);
        if (t_tmp_buf.size() < tmp_need) t_tmp_buf.resize(tmp_need);

        float* col_buf = t_col_buf.data();
        float* tmp_buf = t_tmp_buf.data();

        for (int64_t mt = t_start; mt < t_end; ++mt) {
            const int64_t m_start = mt * MC;
            const int64_t m_count = std::min<int64_t>(MC, M_total - m_start);

            im2col_tile_nhwc(input, col_buf, m_start, m_count,
                             Cin, H, W, KH, KW,
                             stride_h, stride_w, pad_h, pad_w,
                             dil_h, dil_w, Wout, K);

            for (int64_t n_start = 0; n_start < Cout; n_start += NC) {
                const int64_t n_count = std::min<int64_t>(NC, Cout - n_start);
                gemm_tile_16(col_buf, K, W_packed, n_start, n_count,
                             tmp_buf, n_count, m_count);
                write_output_tile_fast(tmp_buf, n_count, output,
                                       n_start, n_count, m_start, m_count,
                                       bias, Cout,
                                       fused_activation, leaky_slope);
            }
        }
    });

    return 1;
}

void engine_conv2d_gemm_clear_cache() noexcept {
    std::unique_lock<std::shared_mutex> lk(g_wcache_mtx);
    g_wcache.clear();
    std::unique_lock<std::shared_mutex> lk2(g_wkmn_mtx);
    g_wkmn.clear();
}