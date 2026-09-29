#ifndef ENGINE_OPS_UPSAMPLE_HPP
#define ENGINE_OPS_UPSAMPLE_HPP

#include "engine/tensor.hpp"

namespace engine {
namespace ops {

enum class UpsampleMode {
    Nearest,    /* En yakın komşu — hızlı, FPN için yaygın */
    Bilinear,   /* Bilinear interpolasyon — daha yumuşak */
};

/* input: [N, C, H, W] → output: [N, C, H*scale, W*scale] */
void upsample(const Tensor& input,
              Tensor& output,
              int64_t scale_h,
              int64_t scale_w,
              UpsampleMode mode = UpsampleMode::Nearest) noexcept;

[[nodiscard]] Tensor upsample(const Tensor& input,
                              int64_t scale_h,
                              int64_t scale_w,
                              UpsampleMode mode = UpsampleMode::Nearest) noexcept;

/* Belirli hedef boyuta örnekleme (scale yerine) */
[[nodiscard]] Tensor resize(const Tensor& input,
                            int64_t target_h,
                            int64_t target_w,
                            UpsampleMode mode = UpsampleMode::Nearest) noexcept;

} /* namespace ops */
} /* namespace engine */
#endif