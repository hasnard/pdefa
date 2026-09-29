/* ============================================================
 *  O(1) SEGREGATED BIN-POOL & STATIK PLAN YONETIMI
 * ============================================================ */

size_t Session::size_to_bin_(size_t bytes) noexcept {
    constexpr size_t kMinSize = 1ULL << kMinBinShift;
    if (bytes <= kMinSize) return 0;
    constexpr size_t kMaxSize = 1ULL << kMaxBinShift;
    if (bytes >= kMaxSize) return kNumBins - 1;

#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanReverse64(&idx, bytes - 1);
    const size_t shift = static_cast<size_t>(idx) + 1;
#elif defined(__GNUC__) || defined(__clang__)
    const size_t shift = 64 - static_cast<size_t>(__builtin_clzll(bytes - 1));
#else
    size_t shift = 0;
    size_t temp = bytes - 1;
    while (temp > 0) { temp >>= 1; ++shift; }
#endif

    const size_t bin = (shift >= kMinBinShift) ? (shift - kMinBinShift) : 0;
    return (bin < kNumBins) ? bin : (kNumBins - 1);
}

void* Session::allocate_from_pool(size_t size) noexcept {
    if (size == 0) return nullptr;

    const size_t bin = size_to_bin_(size);

    // [O(1) LIFO]: Serbest listesinden anında al
    if (free_bins_[bin] != nullptr) {
        BlockHeader* hdr = free_bins_[bin];
        free_bins_[bin]  = hdr->next_free;
        hdr->next_free   = nullptr;
        void* payload = reinterpret_cast<void*>(
            reinterpret_cast<uint8_t*>(hdr) + sizeof(BlockHeader));
        /* Non-determinizm fix: yeniden kullanılan bloğu sıfırla */
        std::memset(payload, 0, hdr->payload_size);
        return payload;
    }

    // [Yavaş Yol]: Hizalı blok tahsis et
    const size_t class_size  = 1ULL << (bin + kMinBinShift);

    if (class_size < size) {
        return nullptr;
    }

    const size_t total_alloc = sizeof(BlockHeader) + class_size;

    void* raw = memory::AlignedAllocator::allocate(total_alloc, memory::kDefaultAlign);
    if (!raw) return nullptr;

    BlockHeader* hdr  = reinterpret_cast<BlockHeader*>(raw);
    hdr->magic        = kMagicHeader;
    hdr->bin_idx      = static_cast<uint32_t>(bin);
    hdr->payload_size = class_size;
    hdr->next_free    = nullptr;

    raw_allocated_blocks_.push_back(raw);

    void* payload = reinterpret_cast<void*>(
        reinterpret_cast<uint8_t*>(hdr) + sizeof(BlockHeader));
    /* Non-determinizm fix: yeni bloğu sıfırla */
    std::memset(payload, 0, class_size);
    return payload;
}

void Session::free_to_pool(void* ptr) noexcept {
    if (!ptr) return;

    BlockHeader* hdr = reinterpret_cast<BlockHeader*>(
        static_cast<uint8_t*>(ptr) - sizeof(BlockHeader)
    );

    if (hdr->magic != kMagicHeader) {
        return;
    }

    const uint32_t bin = hdr->bin_idx;
    if (bin < kNumBins) {
        hdr->next_free  = free_bins_[bin];
        free_bins_[bin] = hdr;
    }
}

Tensor Session::acquire_output_buffer(const Shape& shape, DType dtype) noexcept {
    const size_t need = compute_nbytes(shape, dtype);
    if (need == 0) return Tensor{};

    // 1. STATİK PLAN YOLU
    if (has_static_plan_ && plan_runner_ && current_node_ != nullptr &&
        current_node_out_idx_ < current_node_->outputs.size()) {
        const int32_t tidx = current_node_->outputs[current_node_out_idx_++];
        if (tidx >= 0) {
            Tensor t = plan_runner_->view_tensor(tidx, shape, dtype);
            if (!t.is_empty()) {
                return t;
            }
        }
    }

    // İlk çıkarım: tensör boyutunu kaydet
    if (!has_static_plan_ && current_node_ != nullptr &&
        current_node_out_idx_ < current_node_->outputs.size()) {
        const int32_t tidx = current_node_->outputs[current_node_out_idx_++];
        if (tidx >= 0 && static_cast<size_t>(tidx) < tensor_sizes_.size()) {
            if (need > tensor_sizes_[static_cast<size_t>(tidx)]) {
                tensor_sizes_[static_cast<size_t>(tidx)] = need;
            }
        }
    }

    // 2. DİNAMİK YOL
    void* ptr = allocate_from_pool(need);
    if (!ptr) return Tensor{};
    return Tensor::from_external(ptr, shape, dtype);
}

void Session::reset_pool_states_() noexcept {
    for (size_t b = 0; b < kNumBins; ++b) {
        free_bins_[b] = nullptr;
    }
    for (void* raw : raw_allocated_blocks_) {
        BlockHeader* hdr   = reinterpret_cast<BlockHeader*>(raw);
        const uint32_t bin = hdr->bin_idx;
        if (bin < kNumBins) {
            hdr->next_free  = free_bins_[bin];
            free_bins_[bin] = hdr;
        }
    }
}

void Session::init_static_memory_plan_() noexcept {
    if (!model_) return;
    const Graph& g = model_->graph();

    memory_plan_ = memory::MemoryPlan::build_with_sizes(g, tensor_sizes_);
    if (memory_plan_.total_bytes == 0) return;

    plan_runner_ = std::make_unique<memory::PlanRunner>(memory_plan_, arena_);
    if (!plan_runner_->reserve_all()) {
        plan_runner_.reset();
        return;
    }

    /* Non-determinizm fix: tüm slot'ları bir defa sıfırla */
    plan_runner_->zero_all();

    has_static_plan_ = true;
}