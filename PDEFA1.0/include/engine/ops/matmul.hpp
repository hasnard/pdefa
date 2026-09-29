#ifndef ENGINE_OPS_MATMUL_HPP
#define ENGINE_OPS_MATMUL_HPP

#include "engine/tensor.hpp"
#include "engine/memory/arena.hpp"

namespace engine {
namespace ops {

/* C = A @ B
 * A: [M, K], B: [K, N], C: [M, N]
 * Hepsi F32, contiguous. */
void matmul(const Tensor& A, const Tensor& B, Tensor& C) noexcept;

[[nodiscard]] Tensor matmul(const Tensor& A, const Tensor& B) noexcept;

[[nodiscard]] Tensor matmul_in_arena(memory::Arena& arena,
                                     const Tensor& A,
                                     const Tensor& B) noexcept;

/* Batch: C[b] = A[b] @ B[b]
 * A: [batch, M, K], B: [batch, K, N], C: [batch, M, N] */
void batch_matmul(const Tensor& A, const Tensor& B, Tensor& C) noexcept;

/* Çıktı şekli tahmini */
[[nodiscard]] Shape matmul_output_shape(const Shape& a, const Shape& b) noexcept;

} /* namespace ops */
} /* namespace engine */
#endif