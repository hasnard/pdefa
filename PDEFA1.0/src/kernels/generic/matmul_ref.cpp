#include <cstring>
#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#  define ENGINE_RESTRICT __restrict__
#else
#  define ENGINE_RESTRICT
#endif

namespace {

constexpr int64_t TILE_M = 64;
constexpr int64_t TILE_N = 64;
constexpr int64_t TILE_K = 64;

} /* anonymous namespace */

extern "C" void engine_matmul_ref(
    const float* ENGINE_RESTRICT A,
    const float* ENGINE_RESTRICT B,
    float* ENGINE_RESTRICT C,
    int64_t M, int64_t N, int64_t K) noexcept
{
    std::memset(C, 0, static_cast<size_t>(M) * N * sizeof(float));

    for (int64_t ii = 0; ii < M; ii += TILE_M) {
        const int64_t i_end = (ii + TILE_M < M) ? ii + TILE_M : M;

        for (int64_t kk = 0; kk < K; kk += TILE_K) {
            const int64_t k_end = (kk + TILE_K < K) ? kk + TILE_K : K;

            for (int64_t jj = 0; jj < N; jj += TILE_N) {
                const int64_t j_end = (jj + TILE_N < N) ? jj + TILE_N : N;

                for (int64_t i = ii; i < i_end; ++i) {
                    const float* a_row = A + i * K;
                    float*       c_row = C + i * N;

                    for (int64_t k = kk; k < k_end; ++k) {
                        const float a_ik = a_row[k];
                        const float* b_row = B + k * N;
                        for (int64_t j = jj; j < j_end; ++j) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}