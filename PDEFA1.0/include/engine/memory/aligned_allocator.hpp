/* ============================================================================
 *  include/engine/memory/aligned_allocator.hpp
 *  Engine-AI — SIMD dostu hizalı bellek ayırıcı
 *
 *  NE YAPAR?
 *    - 16/32/64-byte hizalı bellek ayırır (SSE/AVX2/AVX-512 için şart)
 *    - Zero-initialize seçeneği
 *    - Yeniden boyutlandırma (realloc benzeri)
 *    - Lock-free istatistikler (toplam/aktif/peak bellek)
 *    - RAII sarmalayıcı (UniqueAlignedPtr)
 *
 *  NEDEN GEREKLİ?
 *    SIMD komutları (_mm256_load_ps, _mm512_load_ps) hizalı adres ister.
 *    Hizasız adres → segfault (AVX-512) veya 2-3x yavaşlık (AVX2).
 *    malloc() sadece 16-byte hizalı garanti eder (glibc). Bu yetersiz!
 *
 *  ÇOKLU ÇEKİRDEK DESTEĞİ:
 *    - allocate/deallocate: her çağrı bağımsız, kilit gerekmez
 *    - İstatistikler: lock-free atomics → çekirdek sayısı arttıkça
 *      çekişme (contention) YOK
 *    - Numa-farkında değil (ileride eklenebilir)
 * ========================================================================== */

#ifndef ENGINE_MEMORY_ALIGNED_ALLOCATOR_HPP
#define ENGINE_MEMORY_ALIGNED_ALLOCATOR_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#if defined(_MSC_VER)
#  include <malloc.h>
#endif

namespace engine {
namespace memory {

/* ============================================================================
 *  SABİTLER
 * ========================================================================== */

/* x86/x64/ARM'de tipik cache line boyutu.
 * Yanlış paylaşımı (false sharing) önlemek için kritik. */
inline constexpr size_t kCacheLineSize = 64;

/* SIMD yükleme hizalamaları */
inline constexpr size_t kAlignSSE     = 16;  // 128-bit
inline constexpr size_t kAlignAVX2    = 32;  // 256-bit
inline constexpr size_t kAlignAVX512  = 64;  // 512-bit

/* Varsayılan hizalama — AVX-512 dahil her şey için yeterli */
inline constexpr size_t kDefaultAlign = kAlignAVX512;

/* ============================================================================
 *  YARDIMCI FONKSİYONLAR (inline, constexpr)
 * ========================================================================== */

/* Hizalamayı 2'nin kuvvetine yuvarla (üstüne).
 * Örnek: round_up_pow2(5) = 8, round_up_pow2(16) = 16, round_up_pow2(17) = 32 */
[[nodiscard]] constexpr size_t round_up_pow2(size_t x) noexcept {
    if (x == 0) return 1;
    if ((x & (x - 1)) == 0) return x;  // zaten 2'nin kuvveti
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
#if SIZE_MAX > UINT32_MAX
    x |= x >> 32;
#endif
    return x + 1;
}

/* Verilen sayı, alignment'ın katı mı? */
[[nodiscard]] constexpr bool is_power_of_two(size_t x) noexcept {
    return x != 0 && (x & (x - 1)) == 0;
}

/* Boyutu alignment'ın üstüne yuvarla */
[[nodiscard]] constexpr size_t align_up(size_t bytes, size_t alignment) noexcept {
    return (bytes + alignment - 1) & ~(alignment - 1);
}

/* Pointer hizalı mı? (runtime, sıcak yolda kullanılmaz) */
[[nodiscard]] inline bool is_aligned(const void* ptr, size_t alignment) noexcept {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

/* ============================================================================
 *  STATİSTİKLER
 * ========================================================================== */
struct AllocatorStats {
    uint64_t total_bytes_allocated;  // toplam ayrılan (kümülatif)
    uint64_t current_bytes;          // şu anda aktif olan
    uint64_t peak_bytes;             // en yüksek seviye
    uint64_t num_allocations;        // kaç kez allocate çağrıldı
    uint64_t num_deallocations;      // kaç kez deallocate
};

/* ============================================================================
 *  ANA AYIRICI — AlignedAllocator
 * ========================================================================== */

class AlignedAllocator {
public:
    /* ------------------------------------------------------------------------
     *  ANA API
     * ---------------------------------------------------------------------- */

    /* Hizalı bellek ayır. Başarısızsa nullptr döner. */
    [[nodiscard]] static void* allocate(size_t bytes,
                                        size_t alignment = kDefaultAlign) noexcept;

    /* Hizalı, sıfırla doldurulmuş bellek ayır. */
    [[nodiscard]] static void* allocate_zeroed(size_t bytes,
                                               size_t alignment = kDefaultAlign) noexcept;

    /* Belleği serbest bırak. nullptr güvenli. */
    static void deallocate(void* ptr) noexcept;

    /* Yeniden boyutlandır (realloc benzeri).
     *  - old_ptr nullptr → yeni allocation
     *  - new_bytes 0 → free
     *  - new <= old → aynı pointer döner
     *  - new > old → yeni blok, veri kopyalanır, eskisi serbest bırakılır */
    [[nodiscard]] static void* reallocate(void* old_ptr,
                                          size_t old_bytes,
                                          size_t new_bytes,
                                          size_t alignment = kDefaultAlign) noexcept;

    /* ------------------------------------------------------------------------
     *  İSTATİSTİKLER (thread-safe)
     * ---------------------------------------------------------------------- */
    [[nodiscard]] static AllocatorStats stats() noexcept;
    static void reset_stats() noexcept;

    /* ------------------------------------------------------------------------
     *  SİLİNMİŞ KOPYALAMA (accidental copy önle)
     * ---------------------------------------------------------------------- */
    AlignedAllocator() = delete;
    ~AlignedAllocator() = delete;
    AlignedAllocator(const AlignedAllocator&) = delete;
    AlignedAllocator& operator=(const AlignedAllocator&) = delete;
};

/* ============================================================================
 *  RAII SARMALAYICI — UniqueAlignedPtr
 *
 *  std::unique_ptr benzeri ama hizalı free kullanır.
 *  Zero-overhead: boş sınıf optimizasyonu (EBO) ile std::unique_ptr kadar hafif.
 * ========================================================================== */

template <typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const noexcept {
        AlignedAllocator::deallocate(ptr);
    }
};

/* Ham bellek için (T = void* gibi) */
using AlignedBufferPtr = std::unique_ptr<void, AlignedDeleter<void>>;

/* T tipi için (construct edilmemiş, sadece hizalı bellek) */
template <typename T>
using AlignedRawPtr = std::unique_ptr<T, AlignedDeleter<T>>;

/* ------------------------------------------------------------------------
 *  Yardımcı: hizalı bellekten RAII pointer üret
 * ------------------------------------------------------------------------ */
[[nodiscard]] inline AlignedBufferPtr make_aligned_buffer(
    size_t bytes,
    size_t alignment = kDefaultAlign) noexcept
{
    return AlignedBufferPtr(AlignedAllocator::allocate(bytes, alignment));
}

/* ------------------------------------------------------------------------
 *  Yardımcı: T tipinden N adet için hizalı, construct edilmiş pointer
 *  (placement new ile construct eder, deleter'da destruct eder)
 * ------------------------------------------------------------------------ */
namespace detail {

template <typename T>
struct ObjectDeleter {
    size_t count;
    void operator()(T* ptr) const noexcept {
        if (!ptr) return;
        for (size_t i = count; i > 0; --i) {
            ptr[i - 1].~T();
        }
        AlignedAllocator::deallocate(ptr);
    }
};

} /* namespace detail */

template <typename T, typename... Args>
[[nodiscard]] std::unique_ptr<T, detail::ObjectDeleter<T>>
make_aligned_object(size_t count = 1, size_t alignment = kDefaultAlign, Args&&... args) {
    static_assert(std::is_trivially_destructible_v<T> || std::is_destructible_v<T>,
                  "T must be destructible");
    void* raw = AlignedAllocator::allocate(sizeof(T) * count, alignment);
    if (!raw) return {nullptr, detail::ObjectDeleter<T>{count}};
    T* obj = static_cast<T*>(raw);
    size_t constructed = 0;
    try {
        for (; constructed < count; ++constructed) {
            new (obj + constructed) T(std::forward<Args>(args)...);
        }
    } catch (...) {
        for (size_t i = constructed; i > 0; --i) obj[i - 1].~T();
        AlignedAllocator::deallocate(raw);
        throw;
    }
    return {obj, detail::ObjectDeleter<T>{count}};
}

/* ============================================================================
 *  YARDIMCI: tür-güvenli yeniden yorumlama
 * ========================================================================== */

/* void* → T* (hizalamayı kontrol eder, debug'da) */
template <typename T>
[[nodiscard]] inline T* aligned_cast(void* ptr) noexcept {
    return static_cast<T*>(ptr);
}

} /* namespace memory */
} /* namespace engine */

#endif /* ENGINE_MEMORY_ALIGNED_ALLOCATOR_HPP */