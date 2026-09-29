#ifndef ENGINE_OPS_CONCAT_HPP
#define ENGINE_OPS_CONCAT_HPP

#include "engine/tensor.hpp"
#include <vector>

namespace engine {
namespace ops {

/* Concat: birlestirir iki veya daha fazla tensor'u bir eksende.
 *   Tum tensor'lar AYNI shape olmali (concat ekseni haric).
 *   Ayni dtype olmali.
 *
 *   Ornek: A=[1,64,80,80], B=[1,128,80,80], axis=1 -> [1,192,80,80]
 */

/* Ana imza: pointer-array. Session::run_node stack buffer kullaniyor,
 * vector allocation'indan kacinmak icin bu imza tercih edilir. */
void concat(const Tensor* const* inputs,
            int32_t n_inputs,
            Tensor& output,
            int32_t axis) noexcept;

/* Pointer-array'den output donduren versiyon */
[[nodiscard]] Tensor concat(const Tensor* const* inputs,
                            int32_t n_inputs,
                            int32_t axis) noexcept;

/* Geriye donuk uyumluluk: vector overload'lar inline, pointer-array'e delege eder. */
inline void concat(const std::vector<const Tensor*>& inputs,
                   Tensor& output,
                   int32_t axis) noexcept {
    concat(inputs.data(), static_cast<int32_t>(inputs.size()), output, axis);
}

inline Tensor concat(const std::vector<const Tensor*>& inputs,
                     int32_t axis) noexcept {
    return concat(inputs.data(), static_cast<int32_t>(inputs.size()), axis);
}

/* 2-tensor kolaylik */
[[nodiscard]] Tensor concat(const Tensor& a, const Tensor& b, int32_t axis) noexcept;

} /* namespace ops */
} /* namespace engine */
#endif