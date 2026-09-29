/* ============================================================================
 *  include/engine/memory/arena.hpp
 *  Engine-AI — Arena (Bump) Allocator
 *
 *  NE YAPAR?
 *    - Büyük bir bellek bloğu alır, içinden sırayla dağıtır (bump pointer)
 *    - Serbest bırakma YOK → hepsi tek seferde reset edilir
 *    - Allocation = pointer aritmetiği → ~5 nanosaniye
 *    - Chunk-based büyüme → ilk blok yetmezse yeni chunk ekler
 *    - Thread-safe tasarım: her thread kendi arena'sına sahip
 *
 *  NE ZAMAN KULLANILIR?
 *    ✅ Inference sırasında (her op için geçici tensor)
 *    ✅ Batch processing (aynı boyutlar tekrar tekrar)
 *    ✅ Pipeline: input → proc → output → reset → tekrar
 *
 *  NE ZAMAN KULLANILMAZ?
 *    ❌ Uzun ömürlü nesneler (ağırlıklar, kalıcı state)
 *    ❌ Tek tek free gerektiren senaryolar
 *    ❌ Farklı boyutlarda sürekli değişen workload (arena'yı sürekli resetlemek gerekir)
 *
 *  ÇOKLU ÇEKİRDEK DESTEĞİ:
 *    - Her thread kendi `Arena` instance'ına sahip olmalı
 *    - `ThreadLocalArena` sınıfı bunu otomatik yönetir
 *    - Global arena YOK → kilit (mutex) YOK → lineer ölçeklenme
 *
 *  BELLEK DÜZENİ:
 *
 *      ┌──────────────────────────────────────────────────────┐
 *      │ Chunk 0 (16 MB)  │  Chunk 1 (16 MB)  │ Chunk 2 ...   │
 *      └──────────────────────────────────────────────────────┘
 *              ↑                    ↑
 *           bump_ptr            bump_ptr
 *      (her chunk kendi içinde sırayla dolar)
 *
 *  TASARIM İLKELERİ:
 *    1) Zero-cost: allocation = 3-4 makine komutu
 *    2) Cache-friendly: ardışık allocation'lar ardışık adreslerde
 *    3) Exception-safe: bellek yetmezse yeni chunk dener, olmazsa nullptr
 *    4) Basit: reset() = O(1), tüm chunk'ları "boşalt" (deallocate değil!)
 * ========================================================================== */

#ifndef ENGINE_MEMORY_ARENA_HPP
#define ENGINE_MEMORY_ARENA_HPP

#include "engine/memory/aligned_allocator.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace engine {
namespace memory {

/* ============================================================================
 *  SABİTLER
 * ========================================================================== */

/* Varsayılan chunk boyutu: 16 MB
 * Neden 16?
 *   - Küçük workload'da 16 MB tek seferde alınır, yeter (çoğu inference < 16 MB)
 *   - Büyük workload'da birden çok chunk alınır, amortize olur
 *   - 2 MB sayfa boyutunun 8 katı → TLB dostu
 *   - OS page fault overhead'i amortize edilir */
inline constexpr size_t kDefaultChunkSize = 16ull * 1024 * 1024;

/* Minimum chunk boyutu: 64 KB */
inline constexpr size_t kMinChunkSize = 64ull * 1024;

/* Maksimum chunk sayısı (sanity check) */
inline constexpr size_t kMaxChunks = 1024;

/* ============================================================================
 *  ARENA SINIFI
 * ========================================================================== */

class Arena {
public:
    /* ------------------------------------------------------------------------
     *  KURUCU / YIKICI
     * ---------------------------------------------------------------------- */

    /* Varsayılan: ilk chunk'ı hemen ayırır. */
    explicit Arena(size_t initial_chunk_size = kDefaultChunkSize) noexcept;

    /* İlk chunk'ı ayırmadan boş arena oluştur (lazy). */
    struct Lazy {};
    explicit Arena(Lazy, size_t initial_chunk_size = kDefaultChunkSize) noexcept;

    ~Arena() noexcept;

    /* Kopyalama YOK (her thread kendi arena'sına sahip) */
    Arena(const Arena&)            = delete;
    Arena& operator=(const Arena&) = delete;

    /* Taşıma var (vector<Arena> için gerekebilir) */
    Arena(Arena&& other) noexcept;
    Arena& operator=(Arena&& other) noexcept;

    /* ------------------------------------------------------------------------
     *  ANA API
     * ---------------------------------------------------------------------- */

    /* Hizalı bellek ayır. Yer yoksa yeni chunk açar. Başarısızsa nullptr. */
    [[nodiscard]] void* allocate(size_t bytes,
                                 size_t alignment = kDefaultAlign) noexcept;

    /* Şablon versiyonu: T tipinden N adet */
    template <typename T>
    [[nodiscard]] T* allocate(size_t count = 1) noexcept {
        static_assert(alignof(T) <= kDefaultAlign,
                      "T requires alignment larger than kDefaultAlign");
        return static_cast<T*>(allocate(sizeof(T) * count, alignof(T)));
    }

    /* Constructor'ı çağırıp nesne döndür */
    template <typename T, typename... Args>
    [[nodiscard]] T* create(Args&&... args) {
        void* mem = allocate(sizeof(T), alignof(T));
        if (!mem) return nullptr;
        return new (mem) T(std::forward<Args>(args)...);
    }

    /* T dizisi oluştur (construct eder, destructor YOK — arena sahibi değil) */
    template <typename T>
    [[nodiscard]] T* create_array(size_t count) {
        void* mem = allocate(sizeof(T) * count, alignof(T));
        if (!mem) return nullptr;
        T* arr = static_cast<T*>(mem);
        for (size_t i = 0; i < count; ++i) new (arr + i) T();
        return arr;
    }

    /* Tüm bellek bloğunu sıfırla — sonraki allocation'lar baştan başlar.
     * O(1). Chunk'lar FREED DEĞİL, yeniden kullanılır. */
    void reset() noexcept;

    /* Tüm chunk'ları gerçekten serbest bırak (OS'a geri ver). */
    void release() noexcept;

    /* Reserve: bir sonraki büyük allocation için önceden yer aç */
    void reserve(size_t bytes) noexcept;

    /* ------------------------------------------------------------------------
     *  SORGULAR
     * ---------------------------------------------------------------------- */

    [[nodiscard]] size_t used_bytes()      const noexcept;  // aktif kullanılan
    [[nodiscard]] size_t capacity_bytes()  const noexcept;  // toplam ayrılan
    [[nodiscard]] size_t num_chunks()      const noexcept;  // chunk sayısı
    [[nodiscard]] size_t initial_chunk_size() const noexcept { return initial_chunk_size_; }

    /* İstatistik snapshot */
    struct Stats {
        size_t used_bytes;
        size_t capacity_bytes;
        size_t num_chunks;
        size_t num_allocations;
        size_t num_chunk_growths;
    };
    [[nodiscard]] Stats stats() const noexcept;

private:
    /* ------------------------------------------------------------------------
     *  İÇ YAPI: Chunk
     * ---------------------------------------------------------------------- */
    struct Chunk {
        void*  base      = nullptr;   // hizalı başlangıç adresi
        size_t capacity  = 0;         // toplam byte
        size_t used      = 0;         // aktif kullanılan byte
        // Not: base, AlignedAllocator ile alınır → destructor'da free edilir.
    };

    /* Chunk listesi. std::vector ama reserve ile önceden boyut verilir,
     * böylece büyüme nadir olur. */
    std::vector<Chunk> chunks_;
    size_t             initial_chunk_size_;
    size_t             current_chunk_idx_ = 0;  // aktif chunk (sonuncu genelde)
    size_t             total_allocations_ = 0;
    size_t             num_chunk_growths_ = 0;

    /* Yeni chunk ekle. Boyut önerisi verilebilir (0 = otomatik). */
    bool grow(size_t min_bytes) noexcept;

    /* İlk chunk'ı ayır */
    void init_first_chunk(size_t size) noexcept;
};

/* ============================================================================
 *  THREAD-LOCAL ARENA (çoklu çekirdek için ideal)
 *
 *  Her thread kendi arena'sına sahip → kilit YOK.
 *  `thread_local` anahtar kelimesiyle otomatik yönetim.
 * ========================================================================== */

class ThreadLocalArena {
public:
    /* Bu thread'in arena'sına eriş. İlk çağrıda otomatik oluşturulur. */
    [[nodiscard]] static Arena& get() noexcept;

    /* Bu thread'in arena'sını sıfırla (inference sonu gibi). */
    static void reset() noexcept;

    /* Bu thread'in arena'sını serbest bırak. */
    static void release() noexcept;

    /* Chunk boyutunu bu thread için ayarla (ilk get() çağrısından önce). */
    static void set_chunk_size(size_t bytes) noexcept;

    ThreadLocalArena() = delete;
};

/* ============================================================================
 *  SCOPED ARENA GUARD
 *
 *  RAII: scope'tan çıkınca arena'yı otomatik reset eder.
 *  Bir inference bloğunun başında oluştur, sonunda otomatik temizlenir.
 *
 *  Kullanım:
 *      {
 *          ScopedArenaGuard guard(arena);
 *          // ... tüm geçici tensorlar bu arena'dan
 *      }   // ← otomatik reset
 * ========================================================================== */

class ScopedArenaGuard {
public:
    explicit ScopedArenaGuard(Arena& arena) noexcept : arena_(&arena) {}
    ~ScopedArenaGuard() noexcept { if (arena_) arena_->reset(); }

    ScopedArenaGuard(const ScopedArenaGuard&)            = delete;
    ScopedArenaGuard& operator=(const ScopedArenaGuard&) = delete;

    ScopedArenaGuard(ScopedArenaGuard&& o) noexcept : arena_(o.arena_) { o.arena_ = nullptr; }

private:
    Arena* arena_;
};

/* ============================================================================
 *  ARENA ALLOCATOR (STL uyumlu)
 *
 *  std::vector<T, ArenaAllocator<T>> gibi kullanılabilir.
 *  Arena ömrü, container ömrünü kapsamalıdır.
 * ========================================================================== */

template <typename T>
class ArenaAllocator {
public:
    using value_type = T;

    ArenaAllocator() noexcept = default;
    explicit ArenaAllocator(Arena* arena) noexcept : arena_(arena) {}

    template <typename U>
    ArenaAllocator(const ArenaAllocator<U>& o) noexcept : arena_(o.arena()) {}

    [[nodiscard]] T* allocate(size_t n) {
        if (n > (SIZE_MAX / sizeof(T))) throw std::bad_alloc();
        void* mem = arena_ ? arena_->allocate(n * sizeof(T), alignof(T)) : nullptr;
        if (!mem) throw std::bad_alloc();
        return static_cast<T*>(mem);
    }

    /* Deallocate = no-op. Arena kendi reset'iyle toplu temizler. */
    void deallocate(T* /*ptr*/, size_t /*n*/) noexcept { /* no-op */ }

    [[nodiscard]] Arena* arena() const noexcept { return arena_; }

    template <typename U>
    struct rebind { using other = ArenaAllocator<U>; };

    bool operator==(const ArenaAllocator& o) const noexcept { return arena_ == o.arena_; }
    bool operator!=(const ArenaAllocator& o) const noexcept { return arena_ != o.arena_; }

private:
    Arena* arena_ = nullptr;
};

} /* namespace memory */
} /* namespace engine */

#endif /* ENGINE_MEMORY_ARENA_HPP */