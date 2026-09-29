/* ============================================================================
 *  src/runtime/affinity.cpp
 *  Engine-AI — CPU affinity implementasyonu
 * ========================================================================== */

#include "engine/runtime/affinity.hpp"
#include "engine/runtime/cpu_info.hpp"

#include <thread>

#if defined(__linux__)
#  include <pthread.h>
#  include <sched.h>
#  include <unistd.h>
#  define ENGINE_HAS_LINUX_AFFINITY 1
#else
#  define ENGINE_HAS_LINUX_AFFINITY 0
#endif

#if defined(_WIN32)
#  include <windows.h>
#  define ENGINE_HAS_WIN_AFFINITY 1
#else
#  define ENGINE_HAS_WIN_AFFINITY 0
#endif

namespace engine {
namespace runtime {

/* ============================================================================
 *  CORE LAYOUT TESPİTİ
 * ========================================================================== */
CoreLayout detect_core_layout() noexcept {
    CoreLayout out;
    const auto& topo = Cpu::topology();
    out.num_physical = topo.physical_cores > 0 ? topo.physical_cores : Affinity::num_pinnable_cores();
    out.num_logical  = topo.logical_cores  > 0 ? topo.logical_cores : out.num_physical;
    out.smt_enabled  = topo.has_hyperthread;
    return out;
}

/* ============================================================================
 *  DESTEK KONTROLÜ
 * ========================================================================== */
bool Affinity::is_supported() noexcept {
#if ENGINE_HAS_LINUX_AFFINITY || ENGINE_HAS_WIN_AFFINITY
    return true;
#else
    return false;
#endif
}

int32_t Affinity::num_pinnable_cores() noexcept {
    const auto& topo = Cpu::topology();
    if (topo.physical_cores > 0) return topo.physical_cores;
    if (topo.logical_cores > 0)  return topo.logical_cores;
    const unsigned hw = std::thread::hardware_concurrency();
    return (hw > 0) ? static_cast<int32_t>(hw) : 1;
}

/* ============================================================================
 *  PLATFORM-SPECIFIC PIN
 * ========================================================================== */

bool Affinity::pin_current_thread(int32_t logical_cpu_id) noexcept {
    if (logical_cpu_id < 0) return false;

#if ENGINE_HAS_LINUX_AFFINITY
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(logical_cpu_id, &cpuset);
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    return rc == 0;

#elif ENGINE_HAS_WIN_AFFINITY
    if (logical_cpu_id >= 64) return false;
    const DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << logical_cpu_id);
    const DWORD_PTR rc = SetThreadAffinityMask(GetCurrentThread(), mask);
    return rc != 0;

#else
    (void)logical_cpu_id;
    return false;
#endif
}

bool Affinity::pin_to_physical_core(int32_t phys_id) noexcept {
    CoreLayout layout = detect_core_layout();
    if (phys_id < 0 || phys_id >= layout.num_physical) return false;
    
    /* Gerçek fiziksel çekirdeğe (Örn: 0, 2, 4...) pinle */
    int32_t logical_id = layout.logical_id_for_physical(phys_id);
    return pin_current_thread(logical_id);
}

bool Affinity::unpin_current_thread() noexcept {
#if ENGINE_HAS_LINUX_AFFINITY
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    const int32_t nc = static_cast<int32_t>(sysconf(_SC_NPROCESSORS_ONLN));
    for (int32_t i = 0; i < nc && i < CPU_SETSIZE; ++i) {
        CPU_SET(i, &cpuset);
    }
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    return rc == 0;

#elif ENGINE_HAS_WIN_AFFINITY
    const DWORD_PTR all = ~static_cast<DWORD_PTR>(0);
    const DWORD_PTR rc = SetThreadAffinityMask(GetCurrentThread(), all);
    return rc != 0;

#else
    return false;
#endif
}

} /* namespace runtime */
} /* namespace engine */