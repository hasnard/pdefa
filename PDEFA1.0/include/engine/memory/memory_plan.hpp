/* ============================================================================
 *  include/engine/memory/memory_plan.hpp
 *  Engine-AI — Statik bellek planlayıcı
 * ========================================================================== */

#ifndef ENGINE_MEMORY_MEMORY_PLAN_HPP
#define ENGINE_MEMORY_MEMORY_PLAN_HPP

#include "engine/graph.hpp"
#include "engine/memory/arena.hpp"
#include <cstdint>
#include <vector>

namespace engine {
namespace memory {

/* ============================================================================
 *  SLOT ATAMASI — her intermediate index hangi slot'a düşüyor?
 * ========================================================================== */
struct SlotAssignment {
    int32_t tensor_idx = -1;
    int32_t slot_id    = -1;
    size_t  offset     = 0;
    size_t  size       = 0;
};

struct MemoryPlan {
    std::vector<size_t> slot_sizes;
    size_t total_bytes = 0;
    std::vector<SlotAssignment> assignments;
    size_t naive_bytes = 0;
    float  savings_ratio = 0.0f;

    static MemoryPlan build(const Graph& g) noexcept;
    static MemoryPlan build_with_sizes(
        const Graph& g,
        const std::vector<size_t>& tensor_sizes) noexcept;
};

/* ============================================================================
 *  RUNTIME YARDIMCISI
 * ========================================================================== */
class PlanRunner {
public:
    PlanRunner(const MemoryPlan& plan, Arena& arena) noexcept;

    /* Plan'a göre tüm slot'ları arena'da rezerve et. */
    bool reserve_all() noexcept;

    /* Belirli bir tensor index'inin bellek adresini döner. */
    [[nodiscard]] Tensor view_tensor(int32_t tensor_idx,
                                     const Shape& shape,
                                     DType dtype) noexcept;

    /* Tüm slot'ları sıfırla (içerik = 0).
     * Non-determinizm fix'i: yeniden kullanılan slot'lar önceki run'dan
     * kalan veriyi taşımasın diye bir defa sıfırlanır. */
    void zero_all() noexcept;

    /* Tüm slot'ları iptal et (pointer'ları nullptr yap). */
    void reset() noexcept;

private:
    const MemoryPlan& plan_;
    Arena&            arena_;
    std::vector<void*> slot_ptrs_;
};

} /* namespace memory */
} /* namespace engine */

#endif