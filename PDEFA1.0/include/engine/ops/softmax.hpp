#ifndef ENGINE_OPS_SOFTMAX_HPP
#define ENGINE_OPS_SOFTMAX_HPP

#include "engine/tensor.hpp"

namespace engine {
namespace ops {

/* Softmax — son eksen üzerinde. Sayısal kararlı (max çıkarma) */
void softmax_(Tensor& x) noexcept;

/* Sigmo: 1 / (1 + exp(-x)) — activation.hpp'de zaten var
 * (sigmoid_ olarak). Burada ek bir API yok. */

} /* namespace ops */
} /* namespace engine */
#endif