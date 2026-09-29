#include "engine/optimizer/layout_transform.hpp"
#include "engine/memory/aligned_allocator.hpp"
#include "engine/runtime/thread_pool.hpp"
#include <cstring>
#include <immintrin.h>

namespace engine {
namespace optimizer {

namespace {

/* ============================================================
 *  AVX2 8x8 transpose — Core Engine
 * ============================================================ */
inline void transpose_8x8_avx2(
    __m256 r0, __m256 r1, __m256 r2, __m256 r3,
    __m256 r4, __m256 r5, __m256 r6, __m256 r7,
    __m256& o0, __m256& o1, __m256& o2, __m256& o3,
    __m256& o4, __m256& o5, __m256& o6, __m256& o7) noexcept
{
    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    o0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    o1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    o2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    o3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    o4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    o5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    o6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    o7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// NHWC okur (Kanal bazlı ardışık), NCHW yazar (Piksel bazlı ardışık)
inline void block8x8_nhwc_to_nchw(const float* src, float* dst, int64_t src_stride, int64_t dst_stride) noexcept {
    __m256 r0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 r1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 r3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 r5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 r7 = _mm256_loadu_ps(src + 7 * src_stride);

    __m256 o0, o1, o2, o3, o4, o5, o6, o7;
    transpose_8x8_avx2(r0, r1, r2, r3, r4, r5, r6, r7, o0, o1, o2, o3, o4, o5, o6, o7);

    _mm256_storeu_ps(dst + 0 * dst_stride, o0);
    _mm256_storeu_ps(dst + 1 * dst_stride, o1);
    _mm256_storeu_ps(dst + 2 * dst_stride, o2);
    _mm256_storeu_ps(dst + 3 * dst_stride, o3);
    _mm256_storeu_ps(dst + 4 * dst_stride, o4);
    _mm256_storeu_ps(dst + 5 * dst_stride, o5);
    _mm256_storeu_ps(dst + 6 * dst_stride, o6);
    _mm256_storeu_ps(dst + 7 * dst_stride, o7);
}

// NCHW okur (Piksel bazlı ardışık), NHWC yazar (Kanal bazlı ardışık)
inline void block8x8_nchw_to_nhwc(const float* src, float* dst, int64_t src_stride, int64_t dst_stride) noexcept {
    __m256 r0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 r1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 r3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 r5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 r7 = _mm256_loadu_ps(src + 7 * src_stride);

    __m256 o0, o1, o2, o3, o4, o5, o6, o7;
    transpose_8x8_avx2(r0, r1, r2, r3, r4, r5, r6, r7, o0, o1, o2, o3, o4, o5, o6, o7);

    _mm256_storeu_ps(dst + 0 * dst_stride, o0);
    _mm256_storeu_ps(dst + 1 * dst_stride, o1);
    _mm256_storeu_ps(dst + 2 * dst_stride, o2);
    _mm256_storeu_ps(dst + 3 * dst_stride, o3);
    _mm256_storeu_ps(dst + 4 * dst_stride, o4);
    _mm256_storeu_ps(dst + 5 * dst_stride, o5);
    _mm256_storeu_ps(dst + 6 * dst_stride, o6);
    _mm256_storeu_ps(dst + 7 * dst_stride, o7);
}

}  // namespace

/* ============================================================
 *  NHWC [N,H,W,C] → NCHW [N,C,H,W]
 *  Optimize edilmiş: AVX2 Block Tiling + Hızlı Pointer Aritmetiği
 * ============================================================ */
Tensor nhwc_to_nchw(const Tensor& src) noexcept {
    if (src.dtype() != DType::F32 || src.rank() != 4) return Tensor{};

    const int64_t N = src.dim(0), H = src.dim(1);
    const int64_t W = src.dim(2), C = src.dim(3);

    Tensor dst = Tensor::empty(Shape{N, C, H, W}, DType::F32);
    if (dst.is_empty()) return dst;

    const float* sp = src.data<float>();
    float* dp = dst.data<float>();
    
    const int64_t HW = H * W;
    const int64_t WC = W * C;
    const int64_t CHW = C * HW;

    auto& pool = runtime::GlobalPool::get();

    // Multithreading'i Outer Loop'a (N * H) çektik.
    pool.parallel_for(0, N * H, [&](int64_t start, int64_t end) {
        int64_t n = start / H;
        int64_t h = start % H;

        for (int64_t i = start; i < end; ++i) {
            const float* src_base = sp + (n * H + h) * WC;
            float* dst_base = dp + n * CHW + h * W;

            int64_t w = 0;
            // AVX2 ile 8x8 Tiling (Cache dostu bellek erişimi)
            for (; w <= W - 8; w += 8) {
                int64_t c = 0;
                for (; c <= C - 8; c += 8) {
                    block8x8_nhwc_to_nchw(
                        src_base + w * C + c,
                        dst_base + c * HW + w,
                        C, HW
                    );
                }
                // Kanal Remainder (Geri kalan kanallar)
                for (; c < C; ++c) {
                    for (int64_t ww = 0; ww < 8; ++ww) {
                        dst_base[c * HW + (w + ww)] = src_base[(w + ww) * C + c];
                    }
                }
            }
            // Genişlik Remainder (Geri kalan pikseller)
            for (; w < W; ++w) {
                for (int64_t c = 0; c < C; ++c) {
                    dst_base[c * HW + w] = src_base[w * C + c];
                }
            }

            // Döngü içi mod/bölme işlemlerinden kaçınmak için manuel artırım
            h++;
            if (h == H) { h = 0; n++; }
        }
    });

    return dst;
}

/* ============================================================
 *  NCHW [N,C,H,W] → NHWC [N,H,W,C]
 *  Optimize edilmiş: AVX2 Block Tiling + Hızlı Pointer Aritmetiği
 * ============================================================ */
Tensor nchw_to_nhwc(const Tensor& src) noexcept {
    if (src.dtype() != DType::F32 || src.rank() != 4) return Tensor{};

    const int64_t N = src.dim(0), C = src.dim(1);
    const int64_t H = src.dim(2), W = src.dim(3);

    Tensor dst = Tensor::empty(Shape{N, H, W, C}, DType::F32);
    if (dst.is_empty()) return dst;

    const float* sp = src.data<float>();
    float* dp = dst.data<float>();
    
    const int64_t HW = H * W;
    const int64_t WC = W * C;
    const int64_t CHW = C * HW;

    auto& pool = runtime::GlobalPool::get();

    pool.parallel_for(0, N * H, [&](int64_t start, int64_t end) {
        int64_t n = start / H;
        int64_t h = start % H;

        for (int64_t i = start; i < end; ++i) {
            const float* src_base = sp + n * CHW + h * W;
            float* dst_base = dp + (n * H + h) * WC;

            int64_t w = 0;
            // AVX2 ile 8x8 Tiling
            for (; w <= W - 8; w += 8) {
                int64_t c = 0;
                for (; c <= C - 8; c += 8) {
                    block8x8_nchw_to_nhwc(
                        src_base + c * HW + w,
                        dst_base + w * C + c,
                        HW, C
                    );
                }
                // Kanal Remainder
                for (; c < C; ++c) {
                    for (int64_t ww = 0; ww < 8; ++ww) {
                        dst_base[(w + ww) * C + c] = src_base[c * HW + (w + ww)];
                    }
                }
            }
            // Genişlik Remainder
            for (; w < W; ++w) {
                for (int64_t c = 0; c < C; ++c) {
                    dst_base[w * C + c] = src_base[c * HW + w];
                }
            }

            h++;
            if (h == H) { h = 0; n++; }
        }
    });

    return dst;
}

Tensor to_layout(const Tensor& src, Layout target) noexcept {
    if (src.rank() != 4) return Tensor{};
    if (target == Layout::NHWC) return nchw_to_nhwc(src);
    return nhwc_to_nchw(src);
}

/* ============================================================
 *  Conv Weights: [Cout, Cin, KH, KW] -> [KH, KW, Cin, Cout]
 *  Optimize edilmiş: Pointer Chasing, Loop Unrolling & Contiguous Access
 * ============================================================ */
Tensor conv_weight_to_nhwc(const Tensor& src) noexcept {
    if (src.dtype() != DType::F32 || src.rank() != 4) return Tensor{};

    const int64_t Cout = src.dim(0);
    const int64_t Cin  = src.dim(1);
    const int64_t KH   = src.dim(2);
    const int64_t KW   = src.dim(3);

    Tensor dst = Tensor::empty(Shape{KH, KW, Cin, Cout}, DType::F32);
    if (dst.is_empty()) return dst;

    const float* sp = src.data<float>();
    float* dp = dst.data<float>();

    const int64_t KHKW = KH * KW;
    const int64_t CinKHKW = Cin * KHKW;
    const int64_t CinCout = Cin * Cout;
    const int64_t KWCinCout = KW * CinCout;

    auto& pool = runtime::GlobalPool::get();

    // Ağırlık tensörü memory transformasyonu matematiksel bir boyut değişimidir.
    // Index formülleri pointer aritmetiği (strength reduction) ile lineer hale getirildi.
    pool.parallel_for(0, Cout, [&](int64_t co_start, int64_t co_end) {
        for (int64_t co = co_start; co < co_end; ++co) {
            float* dst_co = dp + co;
            const float* src_co = sp + co * CinKHKW;
            
            for (int64_t kh = 0; kh < KH; ++kh) {
                float* dst_kh = dst_co + kh * KWCinCout;
                const float* src_kh = src_co + kh * KW;
                
                for (int64_t kw = 0; kw < KW; ++kw) {
                    float* dst_kw = dst_kh + kw * CinCout;
                    const float* src_kw = src_kh + kw;
                    
                    for (int64_t ci = 0; ci < Cin; ++ci) {
                        // En iç döngüde çarpma/toplama yok, sadece sabit stride atlaması var.
                        dst_kw[ci * Cout] = src_kw[ci * KHKW];
                    }
                }
            }
        }
    });

    return dst;
}

} /* namespace optimizer */
} /* namespace engine */