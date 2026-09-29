#ifndef ENGINE_OPS_ELEMENTWISE_HPP
#define ENGINE_OPS_ELEMENTWISE_HPP

#include "engine/tensor.hpp"

namespace engine {
namespace ops {

/* Elementwise binary op'lar (aynı şekil şart — broadcast ileride) */
void add_(Tensor& a, const Tensor& b) noexcept;   /* a += b */
void sub_(Tensor& a, const Tensor& b) noexcept;   /* a -= b */
void mul_(Tensor& a, const Tensor& b) noexcept;   /* a *= b */

/* Skaler işlemler */
void add_scalar_(Tensor& a, float s) noexcept;
void mul_scalar_(Tensor& a, float s) noexcept;

/* Yeni tensor döndüren versiyonlar */
[[nodiscard]] Tensor add(const Tensor& a, const Tensor& b) noexcept;

} /* namespace ops */
} /* namespace engine */
#endif