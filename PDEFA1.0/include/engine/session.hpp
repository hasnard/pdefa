#ifndef ENGINE_SESSION_HPP
#define ENGINE_SESSION_HPP

#include "engine/model.hpp"
#include "engine/memory/arena.hpp"
#include "engine/memory/memory_plan.hpp"
#include "engine/runtime/thread_pool.hpp"
#include <vector>
#include <cstdint>
#include <array>
#include <memory>

namespace engine {

/* Session = bir model için inference state'i.
 * Aynı Model'den birden çok Session oluşturulabilir (paralel thread'ler için). */
class Session {
public:
    explicit Session(const Model& model) noexcept;

    ~Session() noexcept;

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) noexcept = default;
    Session& operator=(Session&&) noexcept = default;

    /* ---- Inference ---- */

    [[nodiscard]] int32_t num_inputs()  const noexcept;
    [[nodiscard]] int32_t num_outputs() const noexcept;

    void set_input(int32_t idx, const Tensor* t) noexcept;

    bool run() noexcept;

    [[nodiscard]] const Tensor* output(int32_t idx) const noexcept;

    [[nodiscard]] const Tensor* debug_intermediate(int32_t idx) const noexcept {
        if (idx < 0 || idx >= static_cast<int32_t>(intermediate_.size())) return nullptr;
        const Tensor& t = intermediate_[static_cast<size_t>(idx)];
        return t.is_empty() ? nullptr : &t;
    }

    [[nodiscard]] memory::Arena& arena() noexcept { return arena_; }

    /* İstatistik */
    struct Stats {
        double   last_run_ms = 0.0;
        uint64_t runs = 0;
    };
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

    /* ============================================================
     *  NODE PROFILING (opsiyonel)
     * ============================================================ */
    using NodeProfileFn = void(*)(const char* node_name, double ms, void* user);

    void set_profile_callback(NodeProfileFn fn, void* user) noexcept {
        profile_fn_   = fn;
        profile_user_ = user;
    }
    void clear_profile_callback() noexcept {
        profile_fn_   = nullptr;
        profile_user_ = nullptr;
    }

private:
    const Model*               model_ = nullptr;
    memory::Arena              arena_{memory::Arena::Lazy{}, 64 * 1024 * 1024};
    std::vector<const Tensor*> external_inputs_;
    std::vector<Tensor>        intermediate_;
    std::vector<Tensor>        outputs_;
    Stats                      stats_;

    /* ============================================================
     *  STATİK TOPOLOJİ & REFERANS SAYILARI
     * ============================================================ */
    std::vector<int32_t> base_ref_counts_;
    std::vector<int32_t> current_ref_counts_;
    std::vector<size_t>  tensor_sizes_;

    /* ============================================================
     *  STATİK BELLEK PLANI (memory_plan.cpp)
     * ============================================================ */
    memory::MemoryPlan                  memory_plan_;
    std::unique_ptr<memory::PlanRunner> plan_runner_;
    bool                                has_static_plan_{false};

    const Node* current_node_         = nullptr;
    size_t      current_node_out_idx_ = 0;

    /* ============================================================
     *  O(1) SEGREGATED BIN-POOL (Dinamik Katman)
     * ============================================================ */
    static constexpr size_t   kMinBinShift = 6;   // 2^6  = 64 Bayt
    static constexpr size_t   kMaxBinShift = 30;  // 2^30 = 1 GB
    static constexpr size_t   kNumBins     = kMaxBinShift - kMinBinShift + 1; // 25 bin
    static constexpr uint32_t kMagicHeader = 0xEA110C01;

    struct alignas(64) BlockHeader {
        uint32_t     magic;
        uint32_t     bin_idx;
        size_t       payload_size;
        BlockHeader* next_free;
        uint8_t      _pad[64 - (sizeof(uint32_t) * 2 + sizeof(size_t) + sizeof(BlockHeader*))];
    };
    static_assert(sizeof(BlockHeader) == 64, "BlockHeader tam 64 bayt olmalidir!");

    std::array<BlockHeader*, kNumBins> free_bins_{};
    std::vector<void*>                 raw_allocated_blocks_;

    /* Helpers */
    [[nodiscard]] static size_t size_to_bin_(size_t bytes) noexcept;
    void*  allocate_from_pool(size_t size) noexcept;
    void   free_to_pool(void* ptr) noexcept;
    Tensor acquire_output_buffer(const Shape& shape, DType dtype) noexcept;
    void   reset_pool_states_() noexcept;
    void   init_static_memory_plan_() noexcept;

    bool run_node(const Node& node) noexcept;

    void fuse_conv_activations_() noexcept;

    NodeProfileFn profile_fn_   = nullptr;
    void*         profile_user_ = nullptr;
};

} /* namespace engine */
#endif