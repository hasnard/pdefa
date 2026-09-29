#ifndef ENGINE_OPS_NORM_HPP
#define ENGINE_OPS_NORM_HPP

#include "engine/tensor.hpp"

namespace engine {
namespace ops {

/* BatchNorm2d (inference mode)
 *   input:  [N, C, H, W]
 *   weight: [C]  (gamma)  — nullptr ise 1 kabul
 *   bias:   [C]  (beta)   — nullptr ise 0
 *   mean:   [C]  (running_mean)
 *   var:    [C]  (running_var)
 *   eps:    küçük sabit (1e-5 tipik)
 *
 *   y = (x - mean) / sqrt(var + eps) * gamma + beta
 *   Optimize: scale = gamma / sqrt(var+eps), shift = beta - mean*scale
 *   → y = x * scale + shift  (tek pass)
 */
void batchnorm2d_(Tensor& x,
                  const Tensor& weight,
                  const Tensor& bias,
                  const Tensor& mean,
                  const Tensor& var,
                  float eps = 1e-5f) noexcept;

/* LayerNorm — son eksen üzerinde normalize
 *   input:  [..., D]
 *   weight: [D]
 *   bias:   [D]
 */
void layernorm_(Tensor& x,
                const Tensor& weight,
                const Tensor& bias,
                float eps = 1e-5f) noexcept;

} /* namespace ops */
} /* namespace engine */
#endif