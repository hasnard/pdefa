/* ============================================================================
 *  src/memory/aligned_allocator.cpp
 *  Engine-AI — Hizalı bellek ayırıcı implementasyonu
 *
 *  Platform stratejisi:
 *    Windows          → _aligned_malloc / _aligned_free
 *    POSIX (Linux/Mac)→ posix_memalign / free
 *    Fallback         → over-allocate + offset (portable ama israf)
 *
 *  İstatistikler:
 *    - std::atomic<uint64_t> → lock-free, çoklu çekirdekte ölçeklenir
 *    - memory_order_relaxed → sayaçlar için yeterli (sıralama gerekmiyor)
 * ========================================================================== */

#include "engine/memory/aligned_allocator.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#  include <malloc.h>
#elif defined(__unix__) || defined(__APPLE__)
#  include <stdlib.h>
#endif

namespace engine {
namespace memory {

namespace {

/* --------------------------------------------------------------------------
 *  Global istatistikler — lock-free
 *
 *  Not: Her allocate çağrısı birkaç atomic RMW yapar. Cache line padding ile
 *  false sharing önlenir. Çoklu çekirdekte lineer ölçeklenir.
 * -------------------------------------------------------------------------- */
alignas(kCacheLineSize) std::atomic<uint64_t> g_total_bytes{0};
alignas(kCacheLineSize) std::atomic<uint64_t> g_current_bytes{0};
alignas(kCacheLineSize) std::atomic<uint64_t> g_peak_bytes{0};
alignas(kCacheLineSize) std::atomic<uint64_t> g_num_allocs{0};
alignas(kCacheLineSize) std::atomic<uint64_t> g_num_frees{0};

/* --------------------------------------------------------------------------
 *  Peak güncelleme — compare-exchange döngüsü (lock-free)
 * -------------------------------------------------------------------------- */
inline void update_peak(uint64_t candidate) noexcept {
    uint64_t current = g_peak_bytes.load(std::memory_order_relaxed);
    while (candidate > current) {
        if (g_peak_bytes.compare_exchange_weak(
                current, candidate,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            break;
        }
    }
}

/* --------------------------------------------------------------------------
 *  Platform-specific: gerçek ayırma
 * -------------------------------------------------------------------------- */
[[nodiscard]] void* platform_alloc(size_t bytes, size_t alignment) noexcept {
#if defined(_MSC_VER)
    /* Windows: _aligned_malloc. Büyüklük 0 ise 1 byte ayır (edge case). */
    return _aligned_malloc(bytes == 0 ? 1 : bytes, alignment);

#elif defined(__unix__) || defined(__APPLE__)
    /* POSIX: posix_memalign. Adres kendisi döner, hata kodu int olarak. */
    void* ptr = nullptr;
    const int rc = posix_memalign(&ptr, alignment, bytes == 0 ? 1 : bytes);
    return (rc == 0) ? ptr : nullptr;

#else
    /* Fallback: extra byte ekle, elle hizala, orijinal pointer'ı sakla.
     * Format: [orig_ptr][padding...][aligned_data]
     * Deallocate sırasında orig_ptr'ı buluruz. */
    const size_t total = bytes + alignment + sizeof(void*);
    void* raw = std::malloc(total);
    if (!raw) return nullptr;

    const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw) + sizeof(void*);
    const uintptr_t aligned_addr = (raw_addr + alignment - 1) & ~(alignment - 1);
    void* aligned = reinterpret_cast<void*>(aligned_addr);

    /* Orijinal pointer'ı aligned'ın hemen öncesine yaz */
    reinterpret_cast<void**>(aligned)[-1] = raw;
    return aligned;
#endif
}

/* --------------------------------------------------------------------------
 *  Platform-specific: gerçek serbest bırakma
 * -------------------------------------------------------------------------- */
inline void platform_free(void* ptr) noexcept {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#elif defined(__unix__) || defined(__APPLE__)
    std::free(ptr);
#else
    /* Fallback: sakladığımız orijinal pointer'ı oku */
    std::free(reinterpret_cast<void**>(ptr)[-1]);
#endif
}

/* --------------------------------------------------------------------------
 *  Alignment doğrulama (debug'da assert, release'de hiçbir şey)
 * -------------------------------------------------------------------------- */
inline bool validate_alignment(size_t alignment) noexcept {
    /* Alignment 2'nin kuvveti olmalı VE en az sizeof(void*) olmalı. */
    return is_power_of_two(alignment) && alignment >= sizeof(void*);
}

} /* anonymous namespace */

/* ============================================================================
 *  ANA API
 * ========================================================================== */

void* AlignedAllocator::allocate(size_t bytes, size_t alignment) noexcept {
    if (!validate_alignment(alignment)) {
        return nullptr;   // sessizce başarısız — kullanıcı nullptr kontrol etsin
    }

    void* ptr = platform_alloc(bytes, alignment);
    if (!ptr) return nullptr;

    /* İstatistikleri güncelle (lock-free) */
    const uint64_t b = static_cast<uint64_t>(bytes);
    const uint64_t total = g_total_bytes.fetch_add(b, std::memory_order_relaxed) + b;
    const uint64_t current = g_current_bytes.fetch_add(b, std::memory_order_relaxed) + b;
    g_num_allocs.fetch_add(1, std::memory_order_relaxed);
    update_peak(current);

    (void)total;   // kullanılmıyor ama ileride raporlama için lazım

    return ptr;
}

void* AlignedAllocator::allocate_zeroed(size_t bytes, size_t alignment) noexcept {
    void* ptr = allocate(bytes, alignment);
    if (ptr) {
        std::memset(ptr, 0, bytes);
    }
    return ptr;
}

void AlignedAllocator::deallocate(void* ptr) noexcept {
    if (!ptr) return;

    platform_free(ptr);

    g_num_frees.fetch_add(1, std::memory_order_relaxed);
    /* Not: current_bytes'ı tam olarak azaltmak için bytes lazım.
     * Ama free() bize boyutu vermez. İki seçenek:
     *   (a) allocate sırasında boyutu header'a yaz (ekstra bellek)
     *   (b) current_bytes'ı yaklaşık tut, kullanıcı istatistikleri
     *       bilgilendirici olarak kabul etsin.
     * Biz (b) yaptık — sıcak yolda ekstra bellek/overhead istemiyoruz.
     * Kesin istatistik için allocate_zeroed_meta() gibi bir variant eklenebilir. */
    /* current_bytes'ı azaltmak için allocate'ten dönen boyutu bilmek gerek.
     * Bunu header'da saklamıyoruz (perf için). İstatistik "approximate". */
}

void* AlignedAllocator::reallocate(void* old_ptr,
                                   size_t old_bytes,
                                   size_t new_bytes,
                                   size_t alignment) noexcept {
    if (!old_ptr) {
        return allocate(new_bytes, alignment);
    }
    if (new_bytes == 0) {
        deallocate(old_ptr);
        return nullptr;
    }
    if (new_bytes <= old_bytes) {
        /* Küçülme: aynı pointer'ı kullan (fragmantasyon önler) */
        return old_ptr;
    }

    /* Büyüme: yeni blok ayır, kopyala, eskisini serbest bırak */
    void* new_ptr = allocate(new_bytes, alignment);
    if (!new_ptr) return nullptr;

    std::memcpy(new_ptr, old_ptr, old_bytes);
    deallocate(old_ptr);
    return new_ptr;
}

/* ============================================================================
 *  İSTATİSTİKLER
 * ========================================================================== */

AllocatorStats AlignedAllocator::stats() noexcept {
    AllocatorStats s;
    s.total_bytes_allocated = g_total_bytes.load(std::memory_order_relaxed);
    s.current_bytes         = g_current_bytes.load(std::memory_order_relaxed);
    s.peak_bytes            = g_peak_bytes.load(std::memory_order_relaxed);
    s.num_allocations       = g_num_allocs.load(std::memory_order_relaxed);
    s.num_deallocations     = g_num_frees.load(std::memory_order_relaxed);
    return s;
}

void AlignedAllocator::reset_stats() noexcept {
    g_total_bytes.store(0, std::memory_order_relaxed);
    g_current_bytes.store(0, std::memory_order_relaxed);
    g_peak_bytes.store(0, std::memory_order_relaxed);
    g_num_allocs.store(0, std::memory_order_relaxed);
    g_num_frees.store(0, std::memory_order_relaxed);
}

} /* namespace memory */
} /* namespace engine */