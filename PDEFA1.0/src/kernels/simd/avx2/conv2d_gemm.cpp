/* ============================================================================
 *  src/kernels/simd/avx2/conv2d_gemm.cpp
 *  Engine-AI — Parcali yapi (conv2d_gemm_parts/*.inc)
 * ========================================================================== */

#ifdef _MSC_VER
#pragma float_control(fast, on)
#endif

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <shared_mutex>

#include "engine/runtime/thread_pool.hpp"
#include "engine/runtime/cpu_info.hpp"

#if defined(_MSC_VER)
#  define ENGINE_RESTRICT __restrict
#  define ENGINE_INLINE   __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#  define ENGINE_RESTRICT __restrict__
#  define ENGINE_INLINE   inline __attribute__((always_inline))
#else
#  define ENGINE_RESTRICT
#  define ENGINE_INLINE   inline
#endif

/* ============================================================================
 *  matmul_avx2 backend — conv GEMM artık bunu çağırıyor.
 *  Bu bildirim namespace'ten ÖNCE olmalı ki 07_dispatch.inc içinde
 *  çağrıldığında derleyici görsün.
 * ========================================================================== */
extern "C" void engine_matmul_avx2(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K) noexcept;

/* ============================================================================
 *  Anonim namespace: yardimcilar (sirayla include edilir)
 * ========================================================================== */
namespace {
#include "conv2d_gemm_parts/01_math.inc"
#include "conv2d_gemm_parts/01b_fast_div_silu.inc"
#include "conv2d_gemm_parts/02_microkernels.inc"
#include "conv2d_gemm_parts/03_weight_cache.inc"
#include "conv2d_gemm_parts/04_im2col.inc"
#include "conv2d_gemm_parts/05_gemm.inc"
#include "conv2d_gemm_parts/06_output.inc"
#include "conv2d_gemm_parts/08_weight_kmn.inc"

ENGINE_INLINE float apply_act_local(float v, int fused_activation, float slope) noexcept {
    switch (fused_activation) {
        case 1: return v > 0.0f ? v : 0.0f;
        case 2: return v > 0.0f ? v : v * slope;
        case 3: return 1.0f / (1.0f + std::exp(-v));
        case 4: return std::tanh(v);
        case 5: return v / (1.0f + std::exp(-v));
        default: return v;
    }
}

ENGINE_INLINE __m256 apply_act_avx2(__m256 acc, int fused_activation,
                                     __m256 vslope, __m256 vzero) noexcept {
    switch (fused_activation) {
        case 1: /* ReLU */
            return _mm256_max_ps(acc, vzero);

        case 2: { /* LeakyReLU: branchless FMA */
            const __m256 pos = _mm256_max_ps(acc, vzero);
            const __m256 neg = _mm256_min_ps(acc, vzero);
            return _mm256_fmadd_ps(neg, vslope, pos);
        }

        case 3: /* Sigmoid */
            return sigmoid_ps_avx2(acc);

        case 4: { /* Tanh: tanh(x) = 2*sigmoid(2x) - 1 via FMA */
            const __m256 vtwo = _mm256_set1_ps(2.0f);
            const __m256 vone = _mm256_set1_ps(1.0f);
            const __m256 x2   = _mm256_mul_ps(acc, vtwo);
            const __m256 sig  = sigmoid_ps_avx2(x2);
            return _mm256_fmsub_ps(vtwo, sig, vone);
        }

        case 5: /* SiLU / Swish */
            return silu_ps_avx2(acc);

        default:
            return acc;
    }
}
} // namespace

/* ============================================================================
 *  Public C API
 * ========================================================================== */
extern "C" {
#include "conv2d_gemm_parts/07_dispatch.inc"
}