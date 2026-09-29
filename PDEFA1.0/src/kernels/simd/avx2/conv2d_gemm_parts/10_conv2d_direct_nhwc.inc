/* ============================================================================
 *  conv2d_gemm_parts/10_conv2d_direct_nhwc.inc
 *  Ultra-Fast Direct NHWC Conv2D & Conv1D for Stem Layers (Cin <= 16)
 *  - Border Peeling: Sıfır dallanmalı iç çekirdek (Zero-Branch Inner Kernel)
 *  - 4x16 Register Tiling: 4 piksel x 16 kanal (Ağırlık okuma yükü %75 daha az)
 *  - Conv1D Adaptive Parallelism: Hout==1 durumunda Wout üzerinden paralelleme
 * ========================================================================== */

namespace {

/* Sınır bölgeleri için güvenli tekil piksel hesaplama */
ENGINE_INLINE void conv_direct_pixel_safe(
    const float* ENGINE_RESTRICT input,
    const float* ENGINE_RESTRICT W_kmn,
    const float* ENGINE_RESTRICT bias,
    float* ENGINE_RESTRICT output,
    int64_t Cin, int64_t H, int64_t W, int64_t Cout,
    int64_t KH, int64_t KW,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t oh, int64_t ow, int64_t Wout,
    int fused_act, float leaky_slope) noexcept
{
    const __m256 vzero  = _mm256_setzero_ps();
    const __m256 vslope = _mm256_set1_ps(leaky_slope);
    float* out_ptr = output + (oh * Wout + ow) * Cout;

    int64_t c = 0;
    for (; c + 15 < Cout; c += 16) {
        __m256 acc0 = bias ? _mm256_loadu_ps(bias + c + 0) : vzero;
        __m256 acc1 = bias ? _mm256_loadu_ps(bias + c + 8) : vzero;

        for (int64_t kh = 0; kh < KH; ++kh) {
            const int64_t ih = oh * stride_h - pad_h + kh * dil_h;
            if (ih < 0 || ih >= H) continue;

            for (int64_t kw = 0; kw < KW; ++kw) {
                const int64_t iw = ow * stride_w - pad_w + kw * dil_w;
                if (iw < 0 || iw >= W) continue;

                const float* in_pixel = input + (ih * W + iw) * Cin;
                const int64_t k_base  = (kh * KW + kw) * Cin;

                for (int64_t ci = 0; ci < Cin; ++ci) {
                    const __m256 in_v = _mm256_set1_ps(in_pixel[ci]);
                    const float* w_ptr = W_kmn + (k_base + ci) * Cout + c;
                    acc0 = _mm256_fmadd_ps(in_v, _mm256_loadu_ps(w_ptr + 0), acc0);
                    acc1 = _mm256_fmadd_ps(in_v, _mm256_loadu_ps(w_ptr + 8), acc1);
                }
            }
        }
        _mm256_storeu_ps(out_ptr + c + 0, apply_act_avx2(acc0, fused_act, vslope, vzero));
        _mm256_storeu_ps(out_ptr + c + 8, apply_act_avx2(acc1, fused_act, vslope, vzero));
    }

    for (; c + 7 < Cout; c += 8) {
        __m256 acc = bias ? _mm256_loadu_ps(bias + c) : vzero;
        for (int64_t kh = 0; kh < KH; ++kh) {
            const int64_t ih = oh * stride_h - pad_h + kh * dil_h;
            if (ih < 0 || ih >= H) continue;
            for (int64_t kw = 0; kw < KW; ++kw) {
                const int64_t iw = ow * stride_w - pad_w + kw * dil_w;
                if (iw < 0 || iw >= W) continue;

                const float* in_pixel = input + (ih * W + iw) * Cin;
                const int64_t k_base  = (kh * KW + kw) * Cin;
                for (int64_t ci = 0; ci < Cin; ++ci) {
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(in_pixel[ci]),
                                          _mm256_loadu_ps(W_kmn + (k_base + ci) * Cout + c), acc);
                }
            }
        }
        _mm256_storeu_ps(out_ptr + c, apply_act_avx2(acc, fused_act, vslope, vzero));
    }

    for (; c < Cout; ++c) {
        float v = bias ? bias[c] : 0.0f;
        for (int64_t kh = 0; kh < KH; ++kh) {
            const int64_t ih = oh * stride_h - pad_h + kh * dil_h;
            if (ih < 0 || ih >= H) continue;
            for (int64_t kw = 0; kw < KW; ++kw) {
                const int64_t iw = ow * stride_w - pad_w + kw * dil_w;
                if (iw < 0 || iw >= W) continue;

                const float* in_pixel = input + (ih * W + iw) * Cin;
                const int64_t k_base  = (kh * KW + kw) * Cin;
                for (int64_t ci = 0; ci < Cin; ++ci) {
                    v += in_pixel[ci] * W_kmn[(k_base + ci) * Cout + c];
                }
            }
        }
        out_ptr[c] = apply_act_local(v, fused_act, leaky_slope);
    }
}

/* İç bölge (Dallanmasız, 4 Piksel x 16 Kanal Microkernel) */
ENGINE_INLINE void conv_direct_interior_4x16_avx2(
    const float* ENGINE_RESTRICT input,
    const float* ENGINE_RESTRICT W_kmn,
    const float* ENGINE_RESTRICT bias,
    float* ENGINE_RESTRICT output,
    int64_t Cin, int64_t W, int64_t Cout,
    int64_t KH, int64_t KW,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t oh, int64_t ow, int64_t Wout,
    int fused_act, float leaky_slope) noexcept
{
    const __m256 vzero  = _mm256_setzero_ps();
    const __m256 vslope = _mm256_set1_ps(leaky_slope);

    float* out0 = output + (oh * Wout + ow + 0) * Cout;
    float* out1 = output + (oh * Wout + ow + 1) * Cout;
    float* out2 = output + (oh * Wout + ow + 2) * Cout;
    float* out3 = output + (oh * Wout + ow + 3) * Cout;

    const int64_t base_ih = oh * stride_h - pad_h;
    const int64_t base_iw = ow * stride_w - pad_w;

    int64_t c = 0;
    for (; c + 15 < Cout; c += 16) {
        __m256 acc00 = bias ? _mm256_loadu_ps(bias + c + 0) : vzero;
        __m256 acc01 = bias ? _mm256_loadu_ps(bias + c + 8) : vzero;
        __m256 acc10 = acc00; __m256 acc11 = acc01;
        __m256 acc20 = acc00; __m256 acc21 = acc01;
        __m256 acc30 = acc00; __m256 acc31 = acc01;

        for (int64_t kh = 0; kh < KH; ++kh) {
            const float* in_row = input + (base_ih + kh * dil_h) * W * Cin;

            for (int64_t kw = 0; kw < KW; ++kw) {
                const int64_t iw0 = (base_iw + 0 * stride_w + kw * dil_w) * Cin;
                const int64_t iw1 = (base_iw + 1 * stride_w + kw * dil_w) * Cin;
                const int64_t iw2 = (base_iw + 2 * stride_w + kw * dil_w) * Cin;
                const int64_t iw3 = (base_iw + 3 * stride_w + kw * dil_w) * Cin;

                const int64_t k_base = (kh * KW + kw) * Cin;

                for (int64_t ci = 0; ci < Cin; ++ci) {
                    /* Komşu 4 pikselin giriş değerleri */
                    const __m256 in0 = _mm256_set1_ps(in_row[iw0 + ci]);
                    const __m256 in1 = _mm256_set1_ps(in_row[iw1 + ci]);
                    const __m256 in2 = _mm256_set1_ps(in_row[iw2 + ci]);
                    const __m256 in3 = _mm256_set1_ps(in_row[iw3 + ci]);

                    /* 16 çıkış kanalı için 2 ağırlık vektörü tek seferde yüklenir */
                    const float* w_ptr = W_kmn + (k_base + ci) * Cout + c;
                    const __m256 w0 = _mm256_loadu_ps(w_ptr + 0);
                    const __m256 w1 = _mm256_loadu_ps(w_ptr + 8);

                    /* 8 FMA: Port 0 ve Port 1 boru hatlarında döngü gecikmesiz çalışır */
                    acc00 = _mm256_fmadd_ps(in0, w0, acc00);
                    acc01 = _mm256_fmadd_ps(in0, w1, acc01);
                    acc10 = _mm256_fmadd_ps(in1, w0, acc10);
                    acc11 = _mm256_fmadd_ps(in1, w1, acc11);
                    acc20 = _mm256_fmadd_ps(in2, w0, acc20);
                    acc21 = _mm256_fmadd_ps(in2, w1, acc21);
                    acc30 = _mm256_fmadd_ps(in3, w0, acc30);
                    acc31 = _mm256_fmadd_ps(in3, w1, acc31);
                }
            }
        }

        /* Fused Aktivasyon ve Kayıt */
        _mm256_storeu_ps(out0 + c + 0, apply_act_avx2(acc00, fused_act, vslope, vzero));
        _mm256_storeu_ps(out0 + c + 8, apply_act_avx2(acc01, fused_act, vslope, vzero));
        _mm256_storeu_ps(out1 + c + 0, apply_act_avx2(acc10, fused_act, vslope, vzero));
        _mm256_storeu_ps(out1 + c + 8, apply_act_avx2(acc11, fused_act, vslope, vzero));
        _mm256_storeu_ps(out2 + c + 0, apply_act_avx2(acc20, fused_act, vslope, vzero));
        _mm256_storeu_ps(out2 + c + 8, apply_act_avx2(acc21, fused_act, vslope, vzero));
        _mm256_storeu_ps(out3 + c + 0, apply_act_avx2(acc30, fused_act, vslope, vzero));
        _mm256_storeu_ps(out3 + c + 8, apply_act_avx2(acc31, fused_act, vslope, vzero));
    }

    /* 8 Kanal Kalanı */
    for (; c + 7 < Cout; c += 8) {
        __m256 acc0 = bias ? _mm256_loadu_ps(bias + c) : vzero;
        __m256 acc1 = acc0; __m256 acc2 = acc0; __m256 acc3 = acc0;

        for (int64_t kh = 0; kh < KH; ++kh) {
            const float* in_row = input + (base_ih + kh * dil_h) * W * Cin;
            for (int64_t kw = 0; kw < KW; ++kw) {
                const int64_t iw0 = (base_iw + 0 * stride_w + kw * dil_w) * Cin;
                const int64_t iw1 = (base_iw + 1 * stride_w + kw * dil_w) * Cin;
                const int64_t iw2 = (base_iw + 2 * stride_w + kw * dil_w) * Cin;
                const int64_t iw3 = (base_iw + 3 * stride_w + kw * dil_w) * Cin;

                const int64_t k_base = (kh * KW + kw) * Cin;
                for (int64_t ci = 0; ci < Cin; ++ci) {
                    const __m256 w = _mm256_loadu_ps(W_kmn + (k_base + ci) * Cout + c);
                    acc0 = _mm256_fmadd_ps(_mm256_set1_ps(in_row[iw0 + ci]), w, acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_set1_ps(in_row[iw1 + ci]), w, acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_set1_ps(in_row[iw2 + ci]), w, acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_set1_ps(in_row[iw3 + ci]), w, acc3);
                }
            }
        }
        _mm256_storeu_ps(out0 + c, apply_act_avx2(acc0, fused_act, vslope, vzero));
        _mm256_storeu_ps(out1 + c, apply_act_avx2(acc1, fused_act, vslope, vzero));
        _mm256_storeu_ps(out2 + c, apply_act_avx2(acc2, fused_act, vslope, vzero));
        _mm256_storeu_ps(out3 + c, apply_act_avx2(acc3, fused_act, vslope, vzero));
    }

    /* Skaler Kalan */
    for (; c < Cout; ++c) {
        float v0 = bias ? bias[c] : 0.0f;
        float v1 = v0, v2 = v0, v3 = v0;
        for (int64_t kh = 0; kh < KH; ++kh) {
            const float* in_row = input + (base_ih + kh * dil_h) * W * Cin;
            for (int64_t kw = 0; kw < KW; ++kw) {
                const int64_t iw0 = (base_iw + 0 * stride_w + kw * dil_w) * Cin;
                const int64_t iw1 = (base_iw + 1 * stride_w + kw * dil_w) * Cin;
                const int64_t iw2 = (base_iw + 2 * stride_w + kw * dil_w) * Cin;
                const int64_t iw3 = (base_iw + 3 * stride_w + kw * dil_w) * Cin;

                const int64_t k_base = (kh * KW + kw) * Cin;
                for (int64_t ci = 0; ci < Cin; ++ci) {
                    const float w = W_kmn[(k_base + ci) * Cout + c];
                    v0 += in_row[iw0 + ci] * w;
                    v1 += in_row[iw1 + ci] * w;
                    v2 += in_row[iw2 + ci] * w;
                    v3 += in_row[iw3 + ci] * w;
                }
            }
        }
        out0[c] = apply_act_local(v0, fused_act, leaky_slope);
        out1[c] = apply_act_local(v1, fused_act, leaky_slope);
        out2[c] = apply_act_local(v2, fused_act, leaky_slope);
        out3[c] = apply_act_local(v3, fused_act, leaky_slope);
    }
}

/* Satır içi yürütücü: Border Peeling ayrıştırmasını yönetir */
ENGINE_INLINE void process_conv_row(
    const float* ENGINE_RESTRICT input,
    const float* ENGINE_RESTRICT W_kmn,
    const float* ENGINE_RESTRICT bias,
    float* ENGINE_RESTRICT output,
    int64_t Cin, int64_t H, int64_t W, int64_t Cout,
    int64_t KH, int64_t KW,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t oh, int64_t Wout,
    int64_t ow_start_col, int64_t ow_end_col,
    int fused_act, float leaky_slope) noexcept
{
    const int64_t base_ih = oh * stride_h - pad_h;
    const bool row_has_vpad = (base_ih < 0) || (base_ih + (KH - 1) * dil_h >= H);

    if (row_has_vpad) {
        for (int64_t ow = ow_start_col; ow < ow_end_col; ++ow) {
            conv_direct_pixel_safe(
                input, W_kmn, bias, output, Cin, H, W, Cout,
                KH, KW, stride_h, stride_w, pad_h, pad_w,
                dil_h, dil_w, oh, ow, Wout, fused_act, leaky_slope
            );
        }
        return;
    }

    /* Yatay iç bölge sınırlarını belirle (Padding gerektirmeyen aralık) */
    const int64_t safe_ow_start = std::max<int64_t>(0, (pad_w + stride_w - 1) / stride_w);
    const int64_t last_valid_w  = W + pad_w - (KW - 1) * dil_w;
    const int64_t safe_ow_end   = (last_valid_w > 0) ? (last_valid_w / stride_w) : 0;

    int64_t ow = ow_start_col;

    /* 1. Sol Sınır (Left Pad) */
    for (; ow < std::min<int64_t>(ow_end_col, safe_ow_start); ++ow) {
        conv_direct_pixel_safe(
            input, W_kmn, bias, output, Cin, H, W, Cout,
            KH, KW, stride_h, stride_w, pad_h, pad_w,
            dil_h, dil_w, oh, ow, Wout, fused_act, leaky_slope
        );
    }

    /* 2. Ana Gövde (Dallanmasız 4x16 Vektörel Hızlı Yol) */
    const int64_t fast_end = std::min<int64_t>(ow_end_col, safe_ow_end);
    for (; ow + 3 < fast_end; ow += 4) {
        conv_direct_interior_4x16_avx2(
            input, W_kmn, bias, output, Cin, W, Cout,
            KH, KW, stride_h, stride_w, pad_h, pad_w,
            dil_h, dil_w, oh, ow, Wout, fused_act, leaky_slope
        );
    }

    /* 3. Sağ Sınır ve Kalanlar (Right Pad & Tails) */
    for (; ow < ow_end_col; ++ow) {
        conv_direct_pixel_safe(
            input, W_kmn, bias, output, Cin, H, W, Cout,
            KH, KW, stride_h, stride_w, pad_h, pad_w,
            dil_h, dil_w, oh, ow, Wout, fused_act, leaky_slope
        );
    }
}

} // namespace

int engine_conv2d_direct_nhwc(
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
    const float* W_kmn = get_weight_kmn(weight, K, Cout, Cin, KH, KW);
    if (!W_kmn) return 0;

    auto& pool = engine::runtime::GlobalPool::get();

    /* ========================================================================
     *  CONV1D UYARLAMASI (Hout == 1 veya H == 1)
     *  İş sadece Hout üzerinden dağıtılırsa tek çekirdeğe hapsolur.
     *  Bu durumda Wout (sekans/uzunluk) bloklara ayrılarak tüm çekirdeklere yayılır.
     * ======================================================================== */
    if (Hout == 1) {
        constexpr int64_t TILE_W = 64;
        const int64_t num_tiles = (Wout + TILE_W - 1) / TILE_W;

        pool.parallel_for(0, num_tiles, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                const int64_t ow_start = t * TILE_W;
                const int64_t ow_end   = std::min<int64_t>(Wout, ow_start + TILE_W);
                process_conv_row(
                    input, W_kmn, bias, output, Cin, H, W, Cout,
                    KH, KW, stride_h, stride_w, pad_h, pad_w,
                    dil_h, dil_w, 0, Wout, ow_start, ow_end,
                    fused_activation, leaky_slope
                );
            }
        });
        return 1;
    }

    /* ========================================================================
     *  CONV2D UYARLAMASI (Hout > 1)
     *  Satır bazlı paralel işleme (Parallel over Hout)
     * ======================================================================== */
    pool.parallel_for(0, Hout, [&](int64_t h0, int64_t h1) {
        for (int64_t oh = h0; oh < h1; ++oh) {
            process_conv_row(
                input, W_kmn, bias, output, Cin, H, W, Cout,
                KH, KW, stride_h, stride_w, pad_h, pad_w,
                dil_h, dil_w, oh, Wout, 0, Wout,
                fused_activation, leaky_slope
            );
        }
    });

    return 1;
}