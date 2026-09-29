/* ============================================================================
 *  src/ops/matmul.cpp
 *  Engine-AI — Matmul op katmanı
 * ========================================================================== */

#include "engine/ops/matmul.hpp"
#include "engine/kernels/kernel_registry.hpp"
#include "engine/runtime/thread_pool.hpp"

#include <cstring>

namespace engine {
namespace ops {

/* ============================================================================
 *  ÇIKTI ŞEKLİ TAHMİNİ
 * ========================================================================== */

Shape matmul_output_shape(const Shape& a, const Shape& b) noexcept {
    if (a.rank() == 2 && b.rank() == 2) {
        if (a.dim(1) != b.dim(0)) return Shape{};
        int64_t dims[2] = { a.dim(0), b.dim(1) };
        return Shape(dims, 2);
    }

    /* Batched: [..., M, K] @ [..., K, N] veya [..., M, K] @ [K, N] */
    if (a.rank() >= 2 && b.rank() >= 2) {
        const int64_t K_a = a.dim(-1);
        const int64_t K_b = b.dim(-2);
        if (K_a != K_b) return Shape{};

        if (b.rank() == 2) {
            // Linear projeksiyon: [..., M, K] @ [K, N] -> [..., M, N]
            int64_t dims[kMaxNdim];
            for (int32_t i = 0; i < a.rank() - 1; ++i) dims[i] = a.dim(i);
            dims[a.rank() - 1] = b.dim(1);
            return Shape(dims, a.rank());
        }

        int64_t dims[kMaxNdim];
        const int32_t rank = (a.rank() > b.rank()) ? a.rank() : b.rank();
        if (rank > kMaxNdim) return Shape{};

        for (int32_t i = 0; i < rank - 2; ++i) {
            dims[i] = (i < a.rank() - 2) ? a.dim(i) : b.dim(i);
        }
        dims[rank - 2] = a.dim(-2);
        dims[rank - 1] = b.dim(-1);
        return Shape(dims, rank);
    }

    return Shape{};
}

/* ============================================================================
 *  BATCH MATMUL (Çok Boyutlu & Paralel)
 * ========================================================================== */

void batch_matmul(const Tensor& A, const Tensor& B, Tensor& C) noexcept {
    if (A.dtype() != DType::F32 ||
        B.dtype() != DType::F32 ||
        C.dtype() != DType::F32) return;
    if (A.rank() < 2 || B.rank() < 2 || C.rank() < 2) return;
    if (A.dim(-1) != B.dim(-2)) return;

    const auto& ks = kernels::Kernels::active();
    if (!ks.matmul) return;

    const int64_t M = A.dim(-2);
    const int64_t K = A.dim(-1);
    const int64_t N = B.dim(-1);

    // Tüm batch boyutlarının toplam katsayısı
    int64_t total_batch = 1;
    for (int32_t i = 0; i < A.rank() - 2; ++i) {
        total_batch *= A.dim(i);
    }
    if (total_batch <= 0) return;

    const float* ap = A.data<float>();
    const float* bp = B.data<float>();
    float*       cp = C.data<float>();

    const int64_t a_stride = M * K;
    const int64_t b_stride = (B.rank() == 2) ? 0 : (K * N); // B 2D ise ağırlığı paylaş
    const int64_t c_stride = M * N;

    auto& pool = runtime::GlobalPool::get();
    if (total_batch > 1) {
        pool.parallel_for(0, total_batch, [&](int64_t b0, int64_t b1) {
            for (int64_t i = b0; i < b1; ++i) {
                ks.matmul(ap + i * a_stride,
                          bp + i * b_stride,
                          cp + i * c_stride,
                          M, N, K);
            }
        });
    } else {
        ks.matmul(ap, bp, cp, M, N, K);
    }
}

/* ============================================================================
 *  MATMUL (2D ve Dispatcher)
 * ========================================================================== */

void matmul(const Tensor& A, const Tensor& B, Tensor& C) noexcept {
    if (A.dtype() != DType::F32 ||
        B.dtype() != DType::F32 ||
        C.dtype() != DType::F32) return;
    if (!A.is_contiguous() || !B.is_contiguous() || !C.is_contiguous()) return;

    // Rank >= 3 ise batch_matmul'a devret
    if (A.rank() > 2 || B.rank() > 2) {
        batch_matmul(A, B, C);
        return;
    }

    if (A.rank() != 2 || B.rank() != 2 || C.rank() != 2) return;

    const int64_t M = A.dim(0);
    const int64_t K = A.dim(1);
    const int64_t N = B.dim(1);

    if (B.dim(0) != K) return;
    if (C.dim(0) != M || C.dim(1) != N) return;

    const auto& ks = kernels::Kernels::active();
    if (!ks.matmul) return;

    ks.matmul(A.data<float>(),
              B.data<float>(),
              C.data<float>(),
              M, N, K);
}

Tensor matmul(const Tensor& A, const Tensor& B) noexcept {
    const Shape out_shape = matmul_output_shape(A.shape(), B.shape());
    if (out_shape.rank() == 0) return Tensor{};

    Tensor C = Tensor::empty(out_shape, DType::F32);
    if (C.is_empty()) return C;

    matmul(A, B, C);
    return C;
}

Tensor matmul_in_arena(memory::Arena& arena,
                       const Tensor& A,
                       const Tensor& B) noexcept
{
    const Shape out_shape = matmul_output_shape(A.shape(), B.shape());
    if (out_shape.rank() == 0) return Tensor{};

    Tensor C = Tensor::create_in_arena(arena, out_shape, DType::F32);
    if (C.is_empty()) return C;

    matmul(A, B, C);
    return C;
}

} /* namespace ops */
} /* namespace engine */