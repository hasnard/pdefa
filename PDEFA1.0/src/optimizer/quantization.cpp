#include "engine/optimizer/quantization.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace engine {
namespace optimizer {

Tensor quantize_int8_symmetric(const Tensor& src, float& out_scale) noexcept {
    if (src.dtype() != DType::F32 || src.is_empty()) return Tensor{};

    const int64_t n = src.numel();
    const float* sp = src.data<float>();

    /* 1) Max abs bul */
    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float a = std::fabs(sp[i]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs < 1e-12f) {
        out_scale = 1.0f;
        Tensor q = Tensor::empty(src.shape(), DType::I8);
        if (!q.is_empty()) std::memset(q.data(), 0, static_cast<size_t>(n));
        return q;
    }

    const float scale = max_abs / 127.0f;
    out_scale = scale;
    const float inv_scale = 1.0f / scale;

    Tensor q = Tensor::empty(src.shape(), DType::I8);
    if (q.is_empty()) return q;

    int8_t* qp = static_cast<int8_t*>(q.data());
    for (int64_t i = 0; i < n; ++i) {
        float v = sp[i] * inv_scale;
        v = std::round(v);
        if (v > 127.0f)   v = 127.0f;
        if (v < -128.0f)  v = -128.0f;
        qp[i] = static_cast<int8_t>(v);
    }
    return q;
}

Tensor quantize_int8(const Tensor& src, QuantParams& out_params) noexcept {
    float scale = 1.0f;
    Tensor q = quantize_int8_symmetric(src, scale);
    out_params.scale = scale;
    out_params.zero_point = 0;
    return q;
}

Tensor dequantize_int8(const Tensor& q, const QuantParams& params) noexcept {
    if (q.dtype() != DType::I8 || q.is_empty()) return Tensor{};

    const int64_t n = q.numel();
    const int8_t* qp = static_cast<const int8_t*>(q.data());

    Tensor dst = Tensor::empty(q.shape(), DType::F32);
    if (dst.is_empty()) return dst;

    float* dp = dst.data<float>();
    const float scale = params.scale;
    const float zp = static_cast<float>(params.zero_point);
    for (int64_t i = 0; i < n; ++i) {
        dp[i] = (static_cast<float>(qp[i]) - zp) * scale;
    }
    return dst;
}

} /* namespace optimizer */
} /* namespace engine */