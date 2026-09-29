#pragma once
#include "engine/tensor.hpp"

namespace engine {
namespace ops {

/* TopK — axis boyunca en büyük (largest=true) veya en küçük K eleman.
 * values.shape == indices.shape == input.shape (axis boyutu K olur).
 * indices dtype = I32. */
void topk(const Tensor& input, int64_t k, int32_t axis,
          bool largest, bool sorted,
          Tensor& values, Tensor& indices) noexcept;

}
}