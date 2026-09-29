#include "engine/kernels/kernel_registry.hpp"
#include "engine/runtime/cpu_info.hpp"

/* ---- Kernel implementasyonları (extern "C") ----
 *
 * Bunlar doğrudan referans edilir, böylece linker kernel .obj'lerini
 * static library'den atamaz. Static init hilesi YOK. */
extern "C" {
    void engine_matmul_ref(const float*, const float*, float*,
                           int64_t, int64_t, int64_t) noexcept;

#ifdef ENGINE_USE_AVX2
    void engine_matmul_avx2(const float*, const float*, float*,
                            int64_t, int64_t, int64_t) noexcept;
#endif

#ifdef ENGINE_USE_AVX512
    void engine_matmul_avx512(const float*, const float*, float*,
                              int64_t, int64_t, int64_t) noexcept;
#endif
}

namespace engine {
namespace kernels {

namespace {

KernelSet build_kernel_set() noexcept {
    KernelSet ks;
    const auto& cpu = runtime::Cpu::features();

    /* En iyiden kötüye doğru seç */
#ifdef ENGINE_USE_AVX512
    if (cpu.has_avx512()) {
        ks.matmul = &engine_matmul_avx512;
        return ks;
    }
#endif

#ifdef ENGINE_USE_AVX2
    if (cpu.avx2) {
        ks.matmul = &engine_matmul_avx2;
        return ks;
    }
#endif

    ks.matmul = &engine_matmul_ref;
    return ks;
}

} /* anonymous namespace */

const KernelSet& Kernels::active() noexcept {
    static const KernelSet g = build_kernel_set();
    return g;
}

const char* Kernels::active_level_name() noexcept {
    const auto& cpu = runtime::Cpu::features();
    if (cpu.has_avx512()) return "AVX-512";
    if (cpu.avx2)         return "AVX2";
    return "scalar";
}

} /* namespace kernels */
} /* namespace engine */