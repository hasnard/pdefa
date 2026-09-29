/* ============================================================================
 *  include/engine/runtime/affinity.hpp
 *  Engine-AI — CPU core affinity (thread pinning)
 * ========================================================================== */

#ifndef ENGINE_RUNTIME_AFFINITY_HPP
#define ENGINE_RUNTIME_AFFINITY_HPP

#include <cstdint>

namespace engine {
namespace runtime {

/* ============================================================================
 *  YARDIMCI: mantıksal çekirdek → fiziksel çekirdek haritası
 * ========================================================================== */
struct CoreLayout {
    int32_t num_physical = 0;
    int32_t num_logical  = 0;
    bool    smt_enabled  = false;

    /* SMT varsa OS genellikle Core0: 0,1 | Core1: 2,3 | Core2: 4,5 şeklinde atar. */
    [[nodiscard]] int32_t logical_id_for_physical(int32_t phys_id) const noexcept {
        if (num_physical <= 0 || phys_id >= num_physical) return phys_id;
        return smt_enabled ? (phys_id * 2) : phys_id;
    }
};

[[nodiscard]] CoreLayout detect_core_layout() noexcept;

class Affinity {
public:
    /* OS Mantıksal Çekirdek ID'sine göre pinleme (Örn: 0, 1, 2...) */
    [[nodiscard]] static bool pin_current_thread(int32_t logical_cpu_id) noexcept;

    /* YENİ VE KRİTİK: Doğrudan FİZİKSEL çekirdeğe pinler.
     * HyperThreading'i otomatik çözer. Ağır AVX2 işleri için BUNU kullanın. */
    [[nodiscard]] static bool pin_to_physical_core(int32_t phys_id) noexcept;

    /* Pinning'i kaldır — OS scheduler tekrar özgür */
    [[nodiscard]] static bool unpin_current_thread() noexcept;

    /* Sistemde kaç fiziksel çekirdek pinlenebilir? */
    [[nodiscard]] static int32_t num_pinnable_cores() noexcept;

    /* Bu platformda affinity destekleniyor mu? */
    [[nodiscard]] static bool is_supported() noexcept;

    Affinity() = delete;
};

} /* namespace runtime */
} /* namespace engine */

#endif