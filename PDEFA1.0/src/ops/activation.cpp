/* ============================================================================
 *  src/ops/activation.cpp
 *  Engine-AI — Aktivasyon op katmanı (scalar referans + AVX2 dispatch)
 *
 *  Scalar burada, AVX2 versiyonlar ayrı TU'da (activation_avx2.cpp).
 *  Runtime'da CPU'ya bakıp uygun olanı çağırır.
 * ========================================================================== */

#include "engine/ops/activation.hpp"
#include "engine/runtime/cpu_info.hpp"

#include <cmath>

namespace engine {
namespace ops {

/* ============================================================================
 *  AVX2 FONKSİYONLARI (activation_avx2.cpp'de tanımlı) — extern "C"
 * ========================================================================== */
extern "C" {
void engine_activation_relu_avx2(float* data, int64_t n) noexcept;
void engine_activation_sigmoid_avx2(float* data, int64_t n) noexcept;
void engine_activation_tanh_avx2(float* data, int64_t n) noexcept;
void engine_activation_silu_avx2(float* data, int64_t n) noexcept;
void engine_activation_gelu_avx2(float* data, int64_t n) noexcept;
void engine_activation_leaky_relu_avx2(float* data, int64_t n, float slope) noexcept;
}

/* ============================================================================
 *  SCALAR REFERANS
 * ========================================================================== */
namespace {

void relu_ref(float* d, int64_t n) noexcept {
    for (int64_t i = 0; i < n; ++i) {
        const float v = d[i];
        d[i] = (v > 0.0f) ? v : 0.0f;
    }
}

void sigmoid_ref(float* d, int64_t n) noexcept {
    for (int64_t i = 0; i < n; ++i) {
        d[i] = 1.0f / (1.0f + std::exp(-d[i]));
    }
}

void tanh_ref(float* d, int64_t n) noexcept {
    for (int64_t i = 0; i < n; ++i) {
        d[i] = std::tanh(d[i]);
    }
}

void silu_ref(float* d, int64_t n) noexcept {
    for (int64_t i = 0; i < n; ++i) {
        const float x = d[i];
        d[i] = x / (1.0f + std::exp(-x));
    }
}

void gelu_ref(float* d, int64_t n) noexcept {
    constexpr float kAlpha = 0.7978845608f;   /* sqrt(2/pi) */
    constexpr float kBeta  = 0.044715f;
    for (int64_t i = 0; i < n; ++i) {
        const float x = d[i];
        const float inner = kAlpha * (x + kBeta * x * x * x);
        d[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void leaky_relu_ref(float* d, int64_t n, float slope) noexcept {
    for (int64_t i = 0; i < n; ++i) {
        const float x = d[i];
        d[i] = (x > 0.0f) ? x : (slope * x);
    }
}

[[nodiscard]] inline bool has_avx2() noexcept {
    return runtime::Cpu::features().avx2;
}

} /* anonymous namespace */

/* ============================================================================
 *  PUBLIC API
 * ========================================================================== */

void relu_(Tensor& x) noexcept {
    if (x.dtype() != DType::F32 || x.is_empty()) return;
    if (!x.is_contiguous()) return;
    const int64_t n = x.numel();
    float* d = x.data<float>();
    if (has_avx2()) engine_activation_relu_avx2(d, n);
    else            relu_ref(d, n);
}

void sigmoid_(Tensor& x) noexcept {
    if (x.dtype() != DType::F32 || x.is_empty()) return;
    if (!x.is_contiguous()) return;
    const int64_t n = x.numel();
    float* d = x.data<float>();
    if (has_avx2()) engine_activation_sigmoid_avx2(d, n);
    else            sigmoid_ref(d, n);
}

void tanh_(Tensor& x) noexcept {
    if (x.dtype() != DType::F32 || x.is_empty()) return;
    if (!x.is_contiguous()) return;
    const int64_t n = x.numel();
    float* d = x.data<float>();
    if (has_avx2()) engine_activation_tanh_avx2(d, n);
    else            tanh_ref(d, n);
}

void silu_(Tensor& x) noexcept {
    if (x.dtype() != DType::F32 || x.is_empty()) return;
    if (!x.is_contiguous()) return;
    const int64_t n = x.numel();
    float* d = x.data<float>();
    if (has_avx2()) engine_activation_silu_avx2(d, n);
    else            silu_ref(d, n);
}

void gelu_(Tensor& x) noexcept {
    if (x.dtype() != DType::F32 || x.is_empty()) return;
    if (!x.is_contiguous()) return;
    const int64_t n = x.numel();
    float* d = x.data<float>();
    if (has_avx2()) engine_activation_gelu_avx2(d, n);
    else            gelu_ref(d, n);
}

void leaky_relu_(Tensor& x, float slope) noexcept {
    if (x.dtype() != DType::F32 || x.is_empty()) return;
    if (!x.is_contiguous()) return;
    const int64_t n = x.numel();
    float* d = x.data<float>();
    if (has_avx2()) engine_activation_leaky_relu_avx2(d, n, slope);
    else            leaky_relu_ref(d, n, slope);
}

} /* namespace ops */
} /* namespace engine */