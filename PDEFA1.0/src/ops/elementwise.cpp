/* ============================================================================
 *  src/ops/elementwise.cpp  —  Elementwise binary ve scalar op'lar (NHWC-safe)
 * ========================================================================== */

#include "engine/ops/elementwise.hpp"
#include "engine/runtime/thread_pool.hpp"
#include <cstring>

namespace engine {
namespace ops {

namespace {

template <typename Op>
void binary_inplace(Tensor& a, const Tensor& b, Op op) noexcept {
    if (a.dtype() != DType::F32 || b.dtype() != DType::F32) return;
    if (a.numel() != b.numel()) return;
    if (!a.is_contiguous() || !b.is_contiguous()) return;

    float* ap = a.data<float>();
    const float* bp = b.data<float>();
    const int64_t n = a.numel();

    constexpr int64_t GRAIN = 4096;
    if (n < 2 * GRAIN) {
        for (int64_t i = 0; i < n; ++i) op(ap[i], bp[i]);
        return;
    }
    auto& pool = runtime::GlobalPool::get();
    pool.parallel_for(0, n, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) op(ap[i], bp[i]);
    }, GRAIN);
}

} /* anonymous namespace */

void add_(Tensor& a, const Tensor& b) noexcept {
    binary_inplace(a, b, [](float& x, float y){ x += y; });
}
void sub_(Tensor& a, const Tensor& b) noexcept {
    binary_inplace(a, b, [](float& x, float y){ x -= y; });
}
void mul_(Tensor& a, const Tensor& b) noexcept {
    binary_inplace(a, b, [](float& x, float y){ x *= y; });
}

void add_scalar_(Tensor& a, float s) noexcept {
    if (a.dtype() != DType::F32) return;
    float* ap = a.data<float>();
    const int64_t n = a.numel();
    auto& pool = runtime::GlobalPool::get();
    pool.parallel_for(0, n, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) ap[i] += s;
    }, 4096);
}

void mul_scalar_(Tensor& a, float s) noexcept {
    if (a.dtype() != DType::F32) return;
    float* ap = a.data<float>();
    const int64_t n = a.numel();
    auto& pool = runtime::GlobalPool::get();
    pool.parallel_for(0, n, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) ap[i] *= s;
    }, 4096);
}

Tensor add(const Tensor& a, const Tensor& b) noexcept {
    if (a.dtype() != DType::F32 || b.dtype() != DType::F32) return Tensor{};
    if (a.shape() != b.shape()) return Tensor{};
    Tensor out = Tensor::empty(a.shape(), DType::F32);
    if (out.is_empty()) return out;
    std::memcpy(out.data(), a.data(), a.nbytes());
    add_(out, b);
    return out;
}

} /* namespace ops */
} /* namespace engine */