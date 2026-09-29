#ifndef ENGINE_KERNELS_KERNEL_REGISTRY_HPP
#define ENGINE_KERNELS_KERNEL_REGISTRY_HPP

#include "engine/kernels/kernel_types.hpp"
#include <cstdint>

namespace engine {
namespace kernels {

struct KernelSet {
    MatmulKernelFn matmul = nullptr;
    UnaryKernelFn  relu    = nullptr;
    UnaryKernelFn  sigmoid = nullptr;
    UnaryKernelFn  tanh_   = nullptr;
    BinaryKernelFn add     = nullptr;
    BinaryKernelFn mul     = nullptr;
    BinaryKernelFn max_    = nullptr;
    Conv2dKernelFn conv2d  = nullptr;
};

class Kernels {
public:
    [[nodiscard]] static const KernelSet& active() noexcept;
    [[nodiscard]] static const char* active_level_name() noexcept;
    Kernels() = delete;
};

} /* namespace kernels */
} /* namespace engine */

#endif