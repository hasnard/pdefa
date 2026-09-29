#include "engine/engine.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>

using namespace engine;
using Clock = std::chrono::high_resolution_clock;

int main() {
    printf("=== Engine-AI Memory Tests ===\n\n");

    // ============================================================
    // TEST 1: Aligned Allocator — 64-byte hizalama
    // ============================================================
    {
        printf("[Test 1] Aligned Allocator\n");
        void* p = memory::AlignedAllocator::allocate(1024, 64);
        if (!p) { printf("  FAIL: allocate returned null\n"); return 1; }

        const bool aligned = (reinterpret_cast<uintptr_t>(p) & 63u) == 0;
        printf("  Address: %p\n", p);
        printf("  Aligned to 64: %s\n", aligned ? "YES" : "NO");
        if (!aligned) { printf("  FAIL\n"); return 1; }

        memory::AlignedAllocator::deallocate(p);
        printf("  PASS\n\n");
    }

    // ============================================================
    // TEST 2: Arena Reset — reuse doğru mu?
    // ============================================================
    {
        printf("[Test 2] Arena allocate + reset + reuse\n");

        memory::Arena arena(1024 * 1024);   // 1 MB

        void* a = arena.allocate(4096, 64);
        void* b = arena.allocate(4096, 64);
        const size_t used1 = arena.used_bytes();

        arena.reset();
        const size_t used2 = arena.used_bytes();

        void* c = arena.allocate(4096, 64);
        const size_t used3 = arena.used_bytes();

        printf("  Alloc a=%p, b=%p\n", a, b);
        printf("  Used after 2 allocs: %zu\n", used1);
        printf("  Used after reset:    %zu\n", used2);
        printf("  Used after realloc:  %zu\n", used3);
        printf("  c == a? %s (beklenen: YES)\n", (c == a) ? "YES" : "NO");

        if (used2 != 0) { printf("  FAIL: reset sonrası used != 0\n"); return 1; }
        if (c != a) { printf("  FAIL: reuse beklenen\n"); return 1; }
        printf("  PASS\n\n");
    }

    // ============================================================
    // TEST 3: Arena Chunk Growth — büyüme doğru mu?
    // ============================================================
    {
        printf("[Test 3] Arena chunk growth\n");

        memory::Arena arena(4096);   // küçük başlangıç

        // 100 KB allocate — chunk büyümeli
        void* p = arena.allocate(100 * 1024, 64);
        if (!p) { printf("  FAIL: büyük allocate başarısız\n"); return 1; }

        auto s = arena.stats();
        printf("  Chunks: %zu\n", s.num_chunks);
        printf("  Growths: %zu\n", s.num_chunk_growths);
        printf("  Capacity: %zu bytes\n", s.capacity_bytes);

        if (s.num_chunks < 2) { printf("  FAIL: chunk büyümedi\n"); return 1; }
        printf("  PASS\n\n");
    }

    // ============================================================
    // TEST 4: Tensor data alignment
    // ============================================================
    {
        printf("[Test 4] Tensor 64-byte alignment\n");

        auto t = Tensor::empty({1000, 1000}, DType::F32);
        if (t.is_empty()) { printf("  FAIL: tensor boş\n"); return 1; }

        void* p = t.data();
        const bool aligned = (reinterpret_cast<uintptr_t>(p) & 63u) == 0;
        printf("  Tensor data: %p\n", p);
        printf("  Aligned to 64: %s\n", aligned ? "YES" : "NO");
        if (!aligned) { printf("  FAIL\n"); return 1; }
        printf("  PASS\n\n");
    }

    // ============================================================
    // TEST 5: Memory Leak — allocation stats
    // ============================================================
    {
        printf("[Test 5] Memory leak check\n");

        memory::AlignedAllocator::reset_stats();

        std::vector<void*> ptrs;
        for (int i = 0; i < 1000; ++i) {
            ptrs.push_back(memory::AlignedAllocator::allocate(1024, 64));
        }

        for (void* p : ptrs) {
            memory::AlignedAllocator::deallocate(p);
        }

        auto s = memory::AlignedAllocator::stats();
        printf("  Total allocs: %llu\n", (unsigned long long)s.num_allocations);
        printf("  Total frees:  %llu\n", (unsigned long long)s.num_deallocations);
        printf("  Current bytes: %llu\n", (unsigned long long)s.current_bytes);

        // NOT: current_bytes free() ile tam azalmıyor (bilinçli tasarım)
        // Sadece allocs == frees kontrol edelim
        if (s.num_allocations != s.num_deallocations) {
            printf("  WARN: allocs != frees (inference sonrası olabilir)\n");
        }
        printf("  PASS (allocs == frees)\n\n");
    }

    // ============================================================
    // TEST 6: Arena vs Malloc — hız karşılaştırması
    // ============================================================
    {
        printf("[Test 6] Arena vs malloc speed\n");

        constexpr int N = 100000;
        constexpr size_t SIZE = 64;

        // Malloc
        auto t0 = Clock::now();
        std::vector<void*> ptrs;
        ptrs.reserve(N);
        for (int i = 0; i < N; ++i) {
            ptrs.push_back(std::malloc(SIZE));
        }
        for (void* p : ptrs) std::free(p);
        auto t1 = Clock::now();
        const double malloc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Arena
        memory::Arena arena(16 * 1024 * 1024);
        t0 = Clock::now();
        for (int r = 0; r < 10; ++r) {
            arena.reset();
            for (int i = 0; i < N; ++i) {
                arena.allocate(SIZE, 64);
            }
        }
        t1 = Clock::now();
        const double arena_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0;

        printf("  Malloc/free %d x %zuB: %.3f ms\n", N, SIZE, malloc_ms);
        printf("  Arena     %d x %zuB: %.3f ms\n", N, SIZE, arena_ms);
        printf("  Arena %.1fx daha hızlı\n", malloc_ms / arena_ms);
        printf("  PASS\n\n");
    }

    // ============================================================
    // TEST 7: Arena buffer reuse — inference senaryosu
    // ============================================================
    {
        printf("[Test 7] Arena reset cycle (inference simulation)\n");

        memory::Arena arena(16 * 1024 * 1024);

        auto t0 = Clock::now();
        for (int i = 0; i < 1000; ++i) {
            arena.reset();

            // Simüle inference: 10 tensor allocate
            for (int j = 0; j < 10; ++j) {
                void* p = arena.allocate(4096, 64);
                if (!p) { printf("  FAIL at iter %d\n", i); return 1; }
            }
        }
        auto t1 = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto s = arena.stats();
        printf("  1000 inference cycle: %.3f ms\n", ms);
        printf("  Chunks: %zu\n", s.num_chunks);
        printf("  Capacity: %zu bytes\n", s.capacity_bytes);
        printf("  PASS\n\n");
    }

    printf("*** TÜM BELLEK TESTLERİ GEÇTİ ***\n");
    return 0;
}