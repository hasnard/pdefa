#ifndef ENGINE_OPS_CONV2D_HPP
#define ENGINE_OPS_CONV2D_HPP

#include "engine/tensor.hpp"
#include "engine/memory/arena.hpp"

namespace engine {
namespace ops {

struct Conv2dConfig {
    int64_t stride_h = 1, stride_w = 1;
    int64_t pad_h    = 0, pad_w    = 0;
    int64_t dil_h    = 1, dil_w    = 1;
    int64_t groups   = 1;

    enum class FusedActivation : int {
        None      = 0,
        ReLU      = 1,
        LeakyReLU = 2,
        Sigmoid   = 3,
        Tanh      = 4,
        SiLU      = 5,   /* Swish */
        GELU      = 6,
    } activation = FusedActivation::None;
    float leaky_slope = 0.01f;
};

[[nodiscard]] int64_t conv_output_size(int64_t in_size, int64_t kernel,
                                       int64_t stride, int64_t pad,
                                       int64_t dilation) noexcept;

[[nodiscard]] Shape conv2d_output_shape(const Shape& input,
                                        const Shape& weight,
                                        const Conv2dConfig& cfg) noexcept;

/* input:  [N, Cin, H, W]
 * weight: [Cout, Cin, KH, KW]
 * bias:   [Cout]  (nullptr olabilir)
 * output: [N, Cout, Hout, Wout] */
void conv2d(const Tensor& input,
            const Tensor& weight,
            const Tensor* bias,
            Tensor& output,
            const Conv2dConfig& cfg = {}) noexcept;

[[nodiscard]] Tensor conv2d(const Tensor& input,
                            const Tensor& weight,
                            const Tensor* bias,
                            const Conv2dConfig& cfg = {}) noexcept;

[[nodiscard]] Tensor conv2d_in_arena(memory::Arena& arena,
                                     const Tensor& input,
                                     const Tensor& weight,
                                     const Tensor* bias,
                                     const Conv2dConfig& cfg = {}) noexcept;

} /* namespace ops */
} /* namespace engine */
#endif