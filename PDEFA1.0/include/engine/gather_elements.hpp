#pragma once
#include "engine/tensor.hpp"

namespace engine {
namespace ops {

/* GatherElements — ONNX semantics.
 * output.shape == indices.shape
 * output[idx] = data[..., i_along_axis, ...]  where i_along_axis = indices[idx]
 * indices dtype = I32 */
void gather_elements(const Tensor& data, const Tensor& indices,
                     int32_t axis, Tensor& output) noexcept;

}
}