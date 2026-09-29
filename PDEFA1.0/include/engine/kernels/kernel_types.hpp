/* ============================================================================
 *  include/engine/kernels/kernel_types.hpp
 *  Engine-AI — Kernel fonksiyon tip tanımları
 *
 *  Tüm kernel'ler bu imzaları kullanır. Aynı op için birden fazla
 *  implementasyon (scalar, SSE, AVX2, AVX-512) olabilir ve runtime'da
 *  CPU'ya göre seçilir.
 *
 *  KONVANSİYON:
 *    - Row-major (C-contiguous) bellek
 *    - C = A @ B  (MxK @ KxN = MxN)
 *    - Pointer'lar 64-byte hizalı (aligned allocator ile)
 *    - Fonksiyonlar noexcept (sıcak yol)
 * ========================================================================== */

#ifndef ENGINE_KERNELS_KERNEL_TYPES_HPP
#define ENGINE_KERNELS_KERNEL_TYPES_HPP

#include "engine/dtype.hpp"
#include <cstddef>
#include <cstdint>

namespace engine {
namespace kernels {

/* ============================================================================
 *  MATMUL
 *
 *  C[M,N] = A[M,K] @ B[K,N]
 *
 *  Tüm pointer'lar row-major, eleman sayısı int64_t (büyük modeller için).
 * ========================================================================== */

using MatmulKernelFn = void (*)(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K) noexcept;

/* Batch matmul: C[b] = A[b] @ B[b]  (aynı boyutlar her batch'te) */
using BatchMatmulKernelFn = void (*)(
    const float* A, const float* B, float* C,
    int64_t batch, int64_t M, int64_t N, int64_t K) noexcept;

/* ============================================================================
 *  ELEMENTWISE (activations, arithmetic)
 *
 *  Unary: y[i] = f(x[i])
 *  Binary: z[i] = f(x[i], y[i])
 * ========================================================================== */

using UnaryKernelFn = void (*)(
    const float* x, float* y, int64_t n) noexcept;

using BinaryKernelFn = void (*)(
    const float* x, const float* y, float* z, int64_t n) noexcept;

/* ============================================================================
 *  CONV2D
 *
 *  im2col + GEMM tekniği kullanılır → matmul kernel'ini tekrar kullanır.
 *  Bu yüzden ayrı bir "kernel" tipi değil, op katmanında çözülür.
 *  Ama düşük seviye direkt conv implementasyonu için imza:
 * ========================================================================== */

using Conv2dKernelFn = void (*)(
    const float* input,   // [N, Cin, H, W]
    const float* weight,  // [Cout, Cin, KH, KW]
    const float* bias,    // [Cout] veya nullptr
    float* output,        // [N, Cout, Hout, Wout]
    int64_t N, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout, int64_t KH, int64_t KW,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w) noexcept;

} /* namespace kernels */
} /* namespace engine */

#endif