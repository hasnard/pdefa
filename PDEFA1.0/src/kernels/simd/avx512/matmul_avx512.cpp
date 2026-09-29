#include <immintrin.h>
#include <cstring>
#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#  define ENGINE_RESTRICT __restrict__
#else
#  define ENGINE_RESTRICT
#endif

namespace {

constexpr int64_t MR = 4;
constexpr int64_t NR = 16;
constexpr int64_t MC = 128;
constexpr int64_t KC = 256;
constexpr int64_t NC = 256;

inline void micro_kernel_4x16_avx512(
    const float* ENGINE_RESTRICT A,
    const float* ENGINE_RESTRICT B,
    float* ENGINE_RESTRICT C,
    int64_t K, int64_t lda, int64_t ldb, int64_t ldc) noexcept
{
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    __m512 c3 = _mm512_setzero_ps();

    for (int64_t k = 0; k < K; ++k) {
        const __m512 b  = _mm512_loadu_ps(B + k * ldb);
        const __m512 a0 = _mm512_set1_ps(A[0 * lda + k]);
        const __m512 a1 = _mm512_set1_ps(A[1 * lda + k]);
        const __m512 a2 = _mm512_set1_ps(A[2 * lda + k]);
        const __m512 a3 = _mm512_set1_ps(A[3 * lda + k]);
        c0 = _mm512_fmadd_ps(a0, b, c0);
        c1 = _mm512_fmadd_ps(a1, b, c1);
        c2 = _mm512_fmadd_ps(a2, b, c2);
        c3 = _mm512_fmadd_ps(a3, b, c3);
    }

    auto store_add = [](float* dst, __m512 v) {
        __m512 old = _mm512_loadu_ps(dst);
        _mm512_storeu_ps(dst, _mm512_add_ps(old, v));
    };
    store_add(C + 0 * ldc, c0);
    store_add(C + 1 * ldc, c1);
    store_add(C + 2 * ldc, c2);
    store_add(C + 3 * ldc, c3);
}

} /* anonymous namespace */

extern "C" void engine_matmul_avx512(
    const float* ENGINE_RESTRICT A,
    const float* ENGINE_RESTRICT B,
    float* ENGINE_RESTRICT C,
    int64_t M, int64_t N, int64_t K) noexcept
{
    std::memset(C, 0, static_cast<size_t>(M) * N * sizeof(float));

    for (int64_t ii = 0; ii < M; ii += MC) {
        const int64_t i_max = (ii + MC < M) ? ii + MC : M;
        for (int64_t jj = 0; jj < N; jj += NC) {
            const int64_t j_max = (jj + NC < N) ? jj + NC : N;
            for (int64_t kk = 0; kk < K; kk += KC) {
                const int64_t k_max = (kk + KC < K) ? kk + KC : K;
                for (int64_t i = ii; i < i_max; i += MR) {
                    const int64_t i_end = (i + MR < i_max) ? i + MR : i_max;
                    for (int64_t j = jj; j < j_max; j += NR) {
                        const int64_t j_end = (j + NR < j_max) ? j + NR : j_max;
                        if (i_end - i == MR && j_end - j == NR) {
                            micro_kernel_4x16_avx512(A + i * K + kk,
                                                     B + kk * N + j,
                                                     C + i * N + j,
                                                     k_max - kk, K, N, N);
                        } else {
                            for (int64_t i2 = i; i2 < i_end; ++i2) {
                                for (int64_t k = kk; k < k_max; ++k) {
                                    const float aik = A[i2 * K + k];
                                    for (int64_t j2 = j; j2 < j_end; ++j2) {
                                        C[i2 * N + j2] += aik * B[k * N + j2];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}