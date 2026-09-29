/* ============================================================================
 *  src/memory/arena.cpp
 *  Engine-AI — Arena allocator implementasyonu
 *
 *  Kritik noktalar:
 *    - allocation = pointer aritmetiği (bump pointer)
 *    - alignment için adres yuvarlama (bit maskeleme)
 *    - chunk büyüme: geometrik (her yeni chunk bir öncekinin 2 katı, max limit)
 *    - reset(): chunk'ları FREED DEĞİL, sadece "used = 0"
 * ========================================================================== */

#include "engine/memory/arena.hpp"

#include <algorithm>
#include <cstring>

namespace engine {
namespace memory {

namespace {

/* --------------------------------------------------------------------------
 *  Chunk boyutu büyüme politikası: geometrik.
 *  Yeni chunk boyutu = max(min_required, önceki_chunk * 2)
 *  Belirli bir üst sınırla (32x initial) → runaway büyümeyi önler.
 * -------------------------------------------------------------------------- */
inline size_t next_chunk_size(size_t previous, size_t min_required, size_t initial) noexcept {
    const size_t cap = initial * 32;
    size_t candidate = previous * 2;
    if (candidate < previous) candidate = min_required;     // overflow
    candidate = std::max(candidate, min_required);
    candidate = std::min(candidate, cap);
    candidate = std::max(candidate, kMinChunkSize);
    return candidate;
}

/* --------------------------------------------------------------------------
 *  Hizalama yardımcısı — verilen adresi yukarı yuvarla
 * -------------------------------------------------------------------------- */
inline uintptr_t align_addr(uintptr_t addr, size_t alignment) noexcept {
    return (addr + (alignment - 1)) & ~static_cast<uintptr_t>(alignment - 1);
}

} /* anonymous namespace */

/* ============================================================================
 *  KURUCU / YIKICI
 * ========================================================================== */

Arena::Arena(size_t initial_chunk_size) noexcept
    : initial_chunk_size_(std::max(initial_chunk_size, kMinChunkSize))
{
    /* Chunk listesi için 16 slot rezerve et — nadir büyüme */
    chunks_.reserve(16);
    init_first_chunk(initial_chunk_size_);
}

Arena::Arena(Lazy, size_t initial_chunk_size) noexcept
    : initial_chunk_size_(std::max(initial_chunk_size, kMinChunkSize))
{
    chunks_.reserve(16);
    /* Lazy: chunk açma, ilk allocation'da açılır */
}

Arena::~Arena() noexcept {
    release();
}

Arena::Arena(Arena&& other) noexcept
    : chunks_(std::move(other.chunks_)),
      initial_chunk_size_(other.initial_chunk_size_),
      current_chunk_idx_(other.current_chunk_idx_),
      total_allocations_(other.total_allocations_),
      num_chunk_growths_(other.num_chunk_growths_)
{
    /* Kaynakları çalınan diğer arena'yı temizle */
    other.chunks_.clear();
    other.current_chunk_idx_ = 0;
    other.total_allocations_ = 0;
    other.num_chunk_growths_ = 0;
}

Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        release();
        chunks_             = std::move(other.chunks_);
        initial_chunk_size_ = other.initial_chunk_size_;
        current_chunk_idx_  = other.current_chunk_idx_;
        total_allocations_  = other.total_allocations_;
        num_chunk_growths_  = other.num_chunk_growths_;

        other.chunks_.clear();
        other.current_chunk_idx_ = 0;
        other.total_allocations_ = 0;
        other.num_chunk_growths_ = 0;
    }
    return *this;
}

/* ============================================================================
 *  ALLOCATION
 * ========================================================================== */

void* Arena::allocate(size_t bytes, size_t alignment) noexcept {
    if (bytes == 0) return nullptr;
    if (alignment < alignof(void*)) alignment = alignof(void*);
    if (!is_power_of_two(alignment)) return nullptr;

    total_allocations_++;

    /* Hızlı yol: mevcut chunk'ta yer var mı?
     * Not: bu, sıcak yolun %99'u. En son chunk'ı deneriz. */
    if (!chunks_.empty()) {
        Chunk& c = chunks_[current_chunk_idx_];
        const uintptr_t base = reinterpret_cast<uintptr_t>(c.base);
        const uintptr_t cur  = base + c.used;
        const uintptr_t aligned_cur = align_addr(cur, alignment);
        const size_t pad = static_cast<size_t>(aligned_cur - cur);
        const size_t needed = pad + bytes;

        if (needed <= c.capacity - c.used) {
            c.used += needed;
            return reinterpret_cast<void*>(aligned_cur);
        }
    }

    /* Yavaş yol: önce mevcut chunk'ları sırayla dene, sonra büyü */
    for (size_t ci = current_chunk_idx_ + 1; ci < chunks_.size(); ++ci) {
        Chunk& c = chunks_[ci];
        const uintptr_t base = reinterpret_cast<uintptr_t>(c.base);
        const uintptr_t cur  = base + c.used;
        const uintptr_t aligned_cur = align_addr(cur, alignment);
        const size_t pad = static_cast<size_t>(aligned_cur - cur);
        const size_t needed = pad + bytes;

        if (needed <= c.capacity - c.used) {
            c.used += needed;
            current_chunk_idx_ = ci;
            return reinterpret_cast<void*>(aligned_cur);
        }
    }

    /* Yeni chunk gerek */
    if (!grow(bytes + alignment)) {
        return nullptr;
    }

    /* Yeni chunk'a yerleştir */
    Chunk& c = chunks_[current_chunk_idx_];
    const uintptr_t base = reinterpret_cast<uintptr_t>(c.base);
    const uintptr_t cur  = base + c.used;
    const uintptr_t aligned_cur = align_addr(cur, alignment);
    const size_t pad = static_cast<size_t>(aligned_cur - cur);
    c.used += pad + bytes;
    return reinterpret_cast<void*>(aligned_cur);
}

/* ============================================================================
 *  RESET / RELEASE
 * ========================================================================== */

void Arena::reset() noexcept {
    /* KRİTİK: chunk'ları FREED ETMİYORUZ.
     * Sadece "used = 0" → sonraki allocation baştan başlar.
     * Bu, reset'i O(nchunks) yapar, tipik olarak 1-3 chunk → ~10 nanosaniye. */
    for (auto& c : chunks_) {
        c.used = 0;
    }
    current_chunk_idx_ = 0;
    /* İstatistikleri sıfırla (total_allocations, num_chunk_growths) */
    total_allocations_ = 0;
    num_chunk_growths_ = 0;
}

void Arena::release() noexcept {
    for (auto& c : chunks_) {
        if (c.base) {
            AlignedAllocator::deallocate(c.base);
            c.base = nullptr;
        }
    }
    chunks_.clear();
    current_chunk_idx_ = 0;
}

void Arena::reserve(size_t bytes) noexcept {
    /* Mevcut kapasite yeterli mi? */
    if (!chunks_.empty()) {
        const Chunk& c = chunks_[current_chunk_idx_];
        if (c.capacity - c.used >= bytes) return;
    }
    /* Yeterli değilse yeni chunk aç */
    (void)grow(bytes);
}

/* ============================================================================
 *  İÇ YARDIMCILAR
 * ========================================================================== */

void Arena::init_first_chunk(size_t size) noexcept {
    if (chunks_.size() >= kMaxChunks) return;

    void* base = AlignedAllocator::allocate(size, kDefaultAlign);
    if (!base) return;   // sessizce başarısız — sonraki allocation retry eder

    Chunk c;
    c.base     = base;
    c.capacity = size;
    c.used     = 0;

    chunks_.push_back(c);
    current_chunk_idx_ = chunks_.size() - 1;
}

bool Arena::grow(size_t min_bytes) noexcept {
    if (chunks_.size() >= kMaxChunks) return false;

    /* Yeni chunk boyutunu hesapla */
    size_t prev_size = chunks_.empty() ? initial_chunk_size_
                                       : chunks_.back().capacity;
    size_t new_size = next_chunk_size(prev_size, min_bytes, initial_chunk_size_);

    /* Yeni chunk'ı ayır */
    void* base = AlignedAllocator::allocate(new_size, kDefaultAlign);
    if (!base) {
        /* Retry: tam olarak min_bytes kadar dene */
        if (new_size > min_bytes) {
            base = AlignedAllocator::allocate(min_bytes, kDefaultAlign);
            if (base) new_size = min_bytes;
        }
        if (!base) return false;
    }

    Chunk c;
    c.base     = base;
    c.capacity = new_size;
    c.used     = 0;

    chunks_.push_back(c);
    current_chunk_idx_ = chunks_.size() - 1;
    num_chunk_growths_++;
    return true;
}

/* ============================================================================
 *  SORGULAR
 * ========================================================================== */

size_t Arena::used_bytes() const noexcept {
    size_t total = 0;
    for (const auto& c : chunks_) total += c.used;
    return total;
}

size_t Arena::capacity_bytes() const noexcept {
    size_t total = 0;
    for (const auto& c : chunks_) total += c.capacity;
    return total;
}

size_t Arena::num_chunks() const noexcept {
    return chunks_.size();
}

Arena::Stats Arena::stats() const noexcept {
    Stats s;
    s.used_bytes         = used_bytes();
    s.capacity_bytes     = capacity_bytes();
    s.num_chunks         = chunks_.size();
    s.num_allocations    = total_allocations_;
    s.num_chunk_growths  = num_chunk_growths_;
    return s;
}

/* ============================================================================
 *  THREAD-LOCAL ARENA
 * ========================================================================== */

namespace {

/* Her thread için özel arena. İlk get() çağrısında oluşturulur. */
struct TLS {
    Arena  arena{Arena::Lazy{}, kDefaultChunkSize};
    size_t configured_chunk_size = kDefaultChunkSize;

    ~TLS() noexcept {
        arena.release();
    }
};

thread_local std::unique_ptr<TLS> t_tls;

inline TLS& get_tls() {
    if (!t_tls) {
        t_tls = std::make_unique<TLS>();
    }
    return *t_tls;
}

} /* anonymous namespace */

Arena& ThreadLocalArena::get() noexcept {
    return get_tls().arena;
}

void ThreadLocalArena::reset() noexcept {
    if (t_tls) t_tls->arena.reset();
}

void ThreadLocalArena::release() noexcept {
    if (t_tls) t_tls->arena.release();
}

void ThreadLocalArena::set_chunk_size(size_t bytes) noexcept {
    TLS& tls = get_tls();
    tls.configured_chunk_size = std::max(bytes, kMinChunkSize);
    /* Not: mevcut arena'nın chunk boyutunu değiştirmek için
     * release + yeni constructor gerekir; basitlik için sadece
     * sonraki arena için geçerli. */
}

} /* namespace memory */
} /* namespace engine */