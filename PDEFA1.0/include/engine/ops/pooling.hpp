#ifndef ENGINE_OPS_POOLING_HPP
#define ENGINE_OPS_POOLING_HPP

#include "engine/tensor.hpp"
#include "engine/memory/arena.hpp"

namespace engine {
namespace ops {

struct Pool2dConfig {
    int64_t kernel_h = 2;
    int64_t kernel_w = 2;
    int64_t stride_h = 2;
    int64_t stride_w = 2;
    int64_t pad_h    = 0;
    int64_t pad_w    = 0;
};

[[nodiscard]] Shape maxpool2d_output_shape(const Shape& input, const Pool2dConfig& cfg) noexcept;
[[nodiscard]] Shape avgpool2d_output_shape(const Shape& input, const Pool2dConfig& cfg) noexcept;

/* input: [N, C, H, W] → output: [N, C, Hout, Wout] */
void maxpool2d(const Tensor& input, Tensor& output, const Pool2dConfig& cfg = {}) noexcept;
void avgpool2d(const Tensor& input, Tensor& output, const Pool2dConfig& cfg = {}) noexcept;

[[nodiscard]] Tensor maxpool2d(const Tensor& input, const Pool2dConfig& cfg = {}) noexcept;
[[nodiscard]] Tensor avgpool2d(const Tensor& input, const Pool2dConfig& cfg = {}) noexcept;

} /* namespace ops */
} /* namespace engine */
#endif