#ifndef ENGINE_OPTIMIZER_LAYOUT_TRANSFORM_HPP
#define ENGINE_OPTIMIZER_LAYOUT_TRANSFORM_HPP

#include "engine/tensor.hpp"

namespace engine {
namespace optimizer {

enum class Layout {
    NCHW,   /* PyTorch default */
    NHWC,   /* TensorFlow default, AVX2 için bazen daha iyi */
};

/* Tek tensor için layout dönüşümü.
 * Dtype F32 olmalı, 4D olmalı. */
[[nodiscard]] Tensor to_layout(const Tensor& src, Layout target) noexcept;

/* NCHW → NHWC */
[[nodiscard]] Tensor nchw_to_nhwc(const Tensor& src) noexcept;

/* NHWC → NCHW */
[[nodiscard]] Tensor nhwc_to_nchw(const Tensor& src) noexcept;

/* Model ağırlıklarını NHWC'ye çevir (Conv2d weight'leri için).
 * PyTorch: [Cout, Cin, KH, KW] → TF: [KH, KW, Cin, Cout] */
[[nodiscard]] Tensor conv_weight_to_nhwc(const Tensor& src) noexcept;

} /* namespace optimizer */
} /* namespace engine */

#endif