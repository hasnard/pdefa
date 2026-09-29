#ifndef ENGINE_OPTIMIZER_QUANTIZATION_HPP
#define ENGINE_OPTIMIZER_QUANTIZATION_HPP

#include "engine/tensor.hpp"
#include <cstdint>

namespace engine {
namespace optimizer {

/* Symmetric INT8 quantization parametreleri (per-tensor) */
struct QuantParams {
    float scale = 1.0f;
    int32_t zero_point = 0;
};

/* Bir tensor'ı quantize et:
 *   q = clamp(round(x / scale) + zero_point, -128, 127)
 *   scale = max(|x|) / 127  (symmetric) */
[[nodiscard]] Tensor quantize_int8(const Tensor& src, QuantParams& out_params) noexcept;

/* Dequantize:
 *   x = (q - zero_point) * scale */
[[nodiscard]] Tensor dequantize_int8(const Tensor& q, const QuantParams& params) noexcept;

/* Simetrik quantization — zero_point = 0 */
[[nodiscard]] Tensor quantize_int8_symmetric(const Tensor& src, float& out_scale) noexcept;

} /* namespace optimizer */
} /* namespace engine */

#endif