#ifndef ENGINE_OPS_ACTIVATION_HPP
#define ENGINE_OPS_ACTIVATION_HPP

#include "engine/tensor.hpp"

namespace engine {
namespace ops {

/* Hepsi in-place: y = f(x), x'in kendisini değiştirir */
void relu_(Tensor& x) noexcept;
void sigmoid_(Tensor& x) noexcept;
void tanh_(Tensor& x) noexcept;
void silu_(Tensor& x) noexcept;                                /* x * sigmoid(x) */
void gelu_(Tensor& x) noexcept;                                /* tanh approx */
void leaky_relu_(Tensor& x, float slope = 0.01f) noexcept;

} /* namespace ops */
} /* namespace engine */
#endif