/* ============================================================================
 *  src/memory/memory_plan.cpp
 *  Engine-AI — Bellek planı implementasyonu
 * ========================================================================== */

#include "engine/memory/memory_plan.hpp"
#include <algorithm>
#include <unordered_map>
#include <cstring>

namespace engine {
namespace memory {

namespace {

/* Bir intermediate tensor'ın yaşam aralığı */
struct LiveRange {
    int32_t tensor_idx = -1;
    int32_t born_at    = 0;
    int32_t dies_at    = 0;
    size_t  size       = 0;
    int32_t slot       = -1;
    size_t  offset     = 0;
};

/* Graph'tan yaşam aralıklarını çıkar */
std::vector<LiveRange> compute_live_ranges(
    const Graph& g,
    const std::vector<size_t>& sizes) noexcept
{
    const int32_t n = static_cast<int32_t>(g.nodes.size());

    std::unordered_map<int32_t, int32_t> producer;
    std::unordered_map<int32_t, int32_t> last_use;

    for (int32_t ni = 0; ni < n; ++ni) {
        const Node& node = g.nodes[static_cast<size_t>(ni)];

        /* Fusion sonrası hayalet node'ları atla */
        if (node.op == OpType::Input || node.op == OpType::Output) continue;

        for (int32_t out_idx : node.outputs) {
            if (out_idx < 0) continue;
            if (producer.find(out_idx) == producer.end()) {
                producer[out_idx] = ni;
            }
        }
        for (int32_t in_idx : node.inputs) {
            if (in_idx < 0) continue;
            last_use[in_idx] = ni;
        }
    }

    for (int32_t out_idx : g.graph_outputs) {
        last_use[out_idx] = n;
    }

    std::vector<LiveRange> ranges;
    ranges.reserve(producer.size());

    for (const auto& [tidx, born] : producer) {
        LiveRange r;
        r.tensor_idx = tidx;
        r.born_at    = born;
        auto it = last_use.find(tidx);
        r.dies_at = (it != last_use.end()) ? it->second : born;
        r.size    = (static_cast<size_t>(tidx) < sizes.size())
                  ? sizes[static_cast<size_t>(tidx)] : 0;
        ranges.push_back(r);
    }

    std::sort(ranges.begin(), ranges.end(),
              [](const LiveRange& a, const LiveRange& b) {
                  return a.born_at < b.born_at;
              });
    return ranges;
}

/* Her tensor kendi slot'unda — overlap yok, offset yok.
 * 1 tensor = 1 slot. Basit ve doğru. */
void assign_slots(std::vector<LiveRange>& ranges,
                  std::vector<size_t>& slot_sizes) noexcept
{
    slot_sizes.clear();
    slot_sizes.reserve(ranges.size());

    for (auto& r : ranges) {
        r.slot   = static_cast<int32_t>(slot_sizes.size());
        r.offset = 0;
        slot_sizes.push_back(r.size);
    }
}

} /* anonymous namespace */

/* ============================================================================
 *  BUILD
 * ========================================================================== */

MemoryPlan MemoryPlan::build(const Graph& g) noexcept {
    std::vector<size_t> dummy;
    return build_with_sizes(g, dummy);
}

MemoryPlan MemoryPlan::build_with_sizes(
    const Graph& g, const std::vector<size_t>& tensor_sizes) noexcept
{
    MemoryPlan plan;

    auto ranges = compute_live_ranges(g, tensor_sizes);
    if (ranges.empty()) return plan;

    assign_slots(ranges, plan.slot_sizes);

    plan.assignments.reserve(ranges.size());
    for (const auto& r : ranges) {
        SlotAssignment a;
        a.tensor_idx = r.tensor_idx;
        a.slot_id    = r.slot;
        a.offset     = r.offset;
        a.size       = r.size;
        plan.assignments.push_back(a);
    }

    size_t total = 0;
    size_t naive = 0;
    for (size_t s : plan.slot_sizes) total += s;
    for (const auto& r : ranges) naive += r.size;

    plan.total_bytes = total;
    plan.naive_bytes = naive;
    plan.savings_ratio = (naive > 0)
        ? static_cast<float>(naive - total) / static_cast<float>(naive)
        : 0.0f;

    return plan;
}

/* ============================================================================
 *  PLAN RUNNER
 * ========================================================================== */

PlanRunner::PlanRunner(const MemoryPlan& plan, Arena& arena) noexcept
    : plan_(plan), arena_(arena)
{
    slot_ptrs_.assign(plan.slot_sizes.size(), nullptr);
}

bool PlanRunner::reserve_all() noexcept {
    for (size_t s = 0; s < plan_.slot_sizes.size(); ++s) {
        const size_t bytes = plan_.slot_sizes[s];
        if (bytes == 0) continue;
        void* p = arena_.allocate(bytes, kDefaultAlign);
        if (!p) return false;
        slot_ptrs_[s] = p;
    }
    return true;
}

Tensor PlanRunner::view_tensor(int32_t tensor_idx,
                               const Shape& shape,
                               DType dtype) noexcept
{
    for (const auto& a : plan_.assignments) {
        if (a.tensor_idx == tensor_idx) {
            if (a.slot_id < 0
                || static_cast<size_t>(a.slot_id) >= slot_ptrs_.size()) {
                return Tensor{};
            }
            void* base = slot_ptrs_[static_cast<size_t>(a.slot_id)];
            if (!base) return Tensor{};
            void* p = static_cast<uint8_t*>(base) + a.offset;
            return Tensor::from_external(p, shape, dtype);
        }
    }
    return Tensor{};
}

void PlanRunner::zero_all() noexcept {
    for (size_t s = 0; s < plan_.slot_sizes.size(); ++s) {
        void* p = slot_ptrs_[s];
        if (p) {
            std::memset(p, 0, plan_.slot_sizes[s]);
        }
    }
}

void PlanRunner::reset() noexcept {
    for (auto*& p : slot_ptrs_) p = nullptr;
}

} /* namespace memory */
} /* namespace engine */