    /* ============================================================================
    *  src/kernels/simd/avx2/matmul_avx2.cpp
    *  Engine-AI — Extreme Low-Latency AVX2 GEMM (Zero-Overhead Edition v5)
    *  Geliştirmeler:
    *   - %100 CPU Kullanımı: parallel_for range dağıtımı total_mr olarak düzeltildi.
    *   - Direct Path K-Fusion: C matrisi RAM okumaları %100 iptal edildi, tam K döngüsü registerda.
    *   - MC=144 Tiling: 256^3 ve 2048^3 gerilemelerini çözen BLIS-Optimal Cache bloklaması.
    * ========================================================================== */
    #ifndef NOMINMAX
    #  define NOMINMAX
    #endif

    #include <immintrin.h>
    #include <algorithm>
    #include <cstdint>
    #include <cstdlib>
    #include <unordered_map>
    #include <shared_mutex>
    #include <memory>
    #include <vector>
    #include <array>
    #include <type_traits>
    #include <atomic>



    #if defined(_MSC_VER) || defined(__MINGW32__)
    #  include <malloc.h>
    #endif

    #include "engine/runtime/thread_pool.hpp"
    #include "engine/runtime/cpu_info.hpp"

    #if defined(__GNUC__) || defined(__clang__)
    #  define ENGINE_RESTRICT __restrict__
    #else
    #  define ENGINE_RESTRICT
    #endif

    namespace {

    constexpr int64_t MR = 6; 
    constexpr int64_t NR = 16;
    constexpr int64_t SINGLE_THREAD_THRESHOLD = 512 * 1024;

    /* ============================================================================
    *  DONANIM PROFİLİ
    * ========================================================================== */
    struct DynamicGemmConfig {
        int64_t mc = 144; // L2 Cache Optimizasyonu (144 * 512 * 4 = 294 KB)
        int64_t kc = 512;
        int64_t nc = 256;
    };

    inline const DynamicGemmConfig& get_gemm_config() noexcept {
        static const DynamicGemmConfig cfg = []() {
            DynamicGemmConfig c;
            const auto& cache = engine::runtime::Cpu::cache();
            const size_t l2 = cache.l2 > 0 ? cache.l2 : 524288;
            
            c.mc = 144;
            c.kc = 512; // L2 bütçesi ve MC=144 ile tam hizalı (294 KB A-Panel)
            
            const size_t l3_budget = static_cast<size_t>(2 * 1024 * 1024); // Varsayılan paylaşımlı L3 toleransı
            int64_t target_nc = l3_budget / (c.kc * sizeof(float));
            target_nc = (target_nc / NR) * NR;
            c.nc = std::clamp<int64_t>(target_nc, 256, 1024);

            return c;
        }();
        return cfg;
    }

    inline void* engine_aligned_alloc(size_t size) noexcept {
        if (size == 0) return nullptr;
    #if defined(_MSC_VER) || defined(__MINGW32__)
        return _aligned_malloc(size, 64);
    #else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, size) != 0) return nullptr;
        return ptr;
    #endif
    }

    inline void engine_aligned_free(void* ptr) noexcept {
    #if defined(_MSC_VER) || defined(__MINGW32__)
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
    }

    struct alignas(64) AlignedBuffer {
        float* ptr = nullptr;
        size_t capacity = 0;
        char padding[64]; 

        AlignedBuffer() {
            capacity = 65536;
            ptr = static_cast<float*>(engine_aligned_alloc(65536 * sizeof(float)));
        }
        explicit AlignedBuffer(size_t elements) {
            if (elements > 0) {
                capacity = elements;
                ptr = static_cast<float*>(engine_aligned_alloc(elements * sizeof(float)));
            }
        }
        void ensure(size_t elements) noexcept {
            if (elements > capacity) {
                if (ptr) engine_aligned_free(ptr);
                capacity = elements;
                ptr = static_cast<float*>(engine_aligned_alloc(elements * sizeof(float)));
            }
        }
        ~AlignedBuffer() {
            if (ptr) engine_aligned_free(ptr);
        }
        AlignedBuffer(const AlignedBuffer&) = delete;
        AlignedBuffer& operator=(const AlignedBuffer&) = delete;
        AlignedBuffer(AlignedBuffer&& o) noexcept : ptr(o.ptr), capacity(o.capacity) {
            o.ptr = nullptr;
            o.capacity = 0;
        }
        AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
            if (this != &o) {
                if (ptr) engine_aligned_free(ptr);
                ptr = o.ptr;
                capacity = o.capacity;
                o.ptr = nullptr;
                o.capacity = 0;
            }
            return *this;
        }
    };

    /* ============================================================================
    *  B MATRIX PACKING & ÖNBELLEK
    * ========================================================================== */
    inline void pack_B_global(int64_t K, int64_t N,
                            int64_t KC, int64_t NC,
                            const float* ENGINE_RESTRICT B,
                            float* ENGINE_RESTRICT pB) noexcept
    {
        for (int64_t k = 0; k < K; k += KC) {
            const int64_t kc = std::min<int64_t>(KC, K - k);
            for (int64_t j = 0; j < N; j += NC) {
                const int64_t nc = std::min<int64_t>(NC, N - j);
                for (int64_t jr = 0; jr < nc; jr += NR) {
                    const int64_t remain = std::min<int64_t>(NR, nc - jr);
                    if (remain == NR) {
                        for (int64_t ki = 0; ki < kc; ++ki) {
                            const float* src = B + (k + ki) * N + (j + jr);
                            _mm_prefetch(reinterpret_cast<const char*>(src + 64), _MM_HINT_T0);
                            __m256 v0 = _mm256_loadu_ps(src + 0);
                            __m256 v1 = _mm256_loadu_ps(src + 8);
                            _mm256_store_ps(pB + 0, v0);
                            _mm256_store_ps(pB + 8, v1);
                            pB += 16;
                        }
                    } else {
                        for (int64_t ki = 0; ki < kc; ++ki) {
                            const float* src = B + (k + ki) * N + (j + jr);
                            for (int64_t ji = 0; ji < remain; ++ji) {
                                *pB++ = src[ji];
                            }
                            for (int64_t ji = remain; ji < NR; ++ji) *pB++ = 0.0f;
                        }
                    }
                }
            }
        }
    }

    struct CacheKey {
        const float* ptr;
        int64_t K;
        int64_t N;
        bool operator==(const CacheKey& o) const noexcept {
            return ptr == o.ptr && K == o.K && N == o.N;
        }
    };

    struct alignas(64) CacheKeyHash {
        size_t operator()(const CacheKey& k) const noexcept {
            size_t h = std::hash<size_t>{}(reinterpret_cast<size_t>(k.ptr) >> 6);
            h ^= std::hash<int64_t>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int64_t>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::shared_mutex g_matmul_wcache_mtx;
    std::unordered_map<CacheKey, std::shared_ptr<AlignedBuffer>, CacheKeyHash> g_matmul_wcache;

    /* Cache generation counter — cache temizlendiğinde artar.
    * thread_local cache bu sayacı kontrol eder; değiştiyse kendi
    * cache'ini geçersiz sayar. Bu, dangling pointer riskini kaldırır. */
    std::atomic<uint64_t> g_matmul_cache_generation{0};

    inline const float* get_packed_B_fast(int64_t K, int64_t N, const float* B, int64_t KC, int64_t NC) {
        thread_local const float* tl_last_B = nullptr;
        thread_local int64_t tl_last_K = 0;
        thread_local int64_t tl_last_N = 0;
        thread_local const float* tl_last_pB = nullptr;
        thread_local uint64_t tl_last_gen = 0;

        /* Cache generation kontrolü — değiştiyse thread_local cache
         * geçersiz sayılır (dangling pointer riskini ortadan kaldırır). */
        const uint64_t cur_gen = g_matmul_cache_generation.load(std::memory_order_acquire);
        if (tl_last_gen != cur_gen) {
            tl_last_B   = nullptr;
            tl_last_pB  = nullptr;
            tl_last_gen = cur_gen;
        }

        if (B == tl_last_B && K == tl_last_K && N == tl_last_N && tl_last_pB != nullptr) {
            return tl_last_pB;
        }

        CacheKey key{B, K, N};
        std::shared_ptr<AlignedBuffer> buf;
        
        {
            std::shared_lock<std::shared_mutex> read_lock(g_matmul_wcache_mtx);
            auto it = g_matmul_wcache.find(key);
            if (it != g_matmul_wcache.end()) buf = it->second;
        }

        if (!buf) {
            std::unique_lock<std::shared_mutex> write_lock(g_matmul_wcache_mtx);
            auto it = g_matmul_wcache.find(key);
            if (it != g_matmul_wcache.end()) {
                buf = it->second;
            } else {
                const int64_t padded_N = ((N + NR - 1) / NR) * NR;
                buf = std::make_shared<AlignedBuffer>(static_cast<size_t>(K) * padded_N);
                pack_B_global(K, N, KC, NC, B, buf->ptr);
                g_matmul_wcache.emplace(key, buf);
            }
        }

        tl_last_B = B;
        tl_last_K = K;
        tl_last_N = N;
        tl_last_pB = buf->ptr;
        return buf->ptr;
    }

    /* ============================================================================
    *  1. ZERO-COPY DIRECT STREAMING MICROKERNEL (FULL K-FUSION EKLENDİ)
    * ========================================================================== */
    #define DIRECT_STEP_6x16(K_ABS) do { \
        __m256 b0 = _mm256_load_ps(pB + 0); \
        __m256 b1 = _mm256_load_ps(pB + 8); \
        __m256 a; \
        a = _mm256_broadcast_ss(a0 + (K_ABS)); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01); \
        a = _mm256_broadcast_ss(a1 + (K_ABS)); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11); \
        a = _mm256_broadcast_ss(a2 + (K_ABS)); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21); \
        a = _mm256_broadcast_ss(a3 + (K_ABS)); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31); \
        a = _mm256_broadcast_ss(a4 + (K_ABS)); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41); \
        a = _mm256_broadcast_ss(a5 + (K_ABS)); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51); \
        pB += 16; \
    } while(0)

    inline void compute_6x16_direct_full_K(
        const float* ENGINE_RESTRICT A_row, int64_t lda,
        const float* ENGINE_RESTRICT pB_global,
        const int64_t* b_offsets, int64_t grid_K, int64_t grid_N,
        int64_t KC, int64_t K, int64_t n_panel_idx, int64_t jr,
        float* ENGINE_RESTRICT C, int64_t ldc) noexcept
    {
        __m256 c00 = _mm256_setzero_ps(); __m256 c01 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps(); __m256 c11 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps(); __m256 c21 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps(); __m256 c31 = _mm256_setzero_ps();
        __m256 c40 = _mm256_setzero_ps(); __m256 c41 = _mm256_setzero_ps();
        __m256 c50 = _mm256_setzero_ps(); __m256 c51 = _mm256_setzero_ps();

        const float* a0 = A_row + 0 * lda;
        const float* a1 = A_row + 1 * lda;
        const float* a2 = A_row + 2 * lda;
        const float* a3 = A_row + 3 * lda;
        const float* a4 = A_row + 4 * lda;
        const float* a5 = A_row + 5 * lda;

        for (int64_t k_idx = 0; k_idx < grid_K; ++k_idx) {
            const int64_t k_base = k_idx * KC;
            const int64_t kc = std::min<int64_t>(KC, K - k_base);
            const float* pB = pB_global + b_offsets[k_idx * grid_N + n_panel_idx] + jr * kc;
            
            int64_t k = 0;
            for (; k <= kc - 4; k += 4) {
                DIRECT_STEP_6x16(k_base + k + 0);
                DIRECT_STEP_6x16(k_base + k + 1);
                DIRECT_STEP_6x16(k_base + k + 2);
                DIRECT_STEP_6x16(k_base + k + 3);
            }
            for (; k < kc; ++k) {
                DIRECT_STEP_6x16(k_base + k);
            }
        }

        _mm256_storeu_ps(C + 0 * ldc + 0, c00); _mm256_storeu_ps(C + 0 * ldc + 8, c01);
        _mm256_storeu_ps(C + 1 * ldc + 0, c10); _mm256_storeu_ps(C + 1 * ldc + 8, c11);
        _mm256_storeu_ps(C + 2 * ldc + 0, c20); _mm256_storeu_ps(C + 2 * ldc + 8, c21);
        _mm256_storeu_ps(C + 3 * ldc + 0, c30); _mm256_storeu_ps(C + 3 * ldc + 8, c31);
        _mm256_storeu_ps(C + 4 * ldc + 0, c40); _mm256_storeu_ps(C + 4 * ldc + 8, c41);
        _mm256_storeu_ps(C + 5 * ldc + 0, c50); _mm256_storeu_ps(C + 5 * ldc + 8, c51);
    }
    #undef DIRECT_STEP_6x16

    inline void compute_6x16_direct_full_K_edge(
        const float* ENGINE_RESTRICT A_row, int64_t lda,
        const float* ENGINE_RESTRICT pB_global,
        const int64_t* b_offsets, int64_t grid_K, int64_t grid_N,
        int64_t KC, int64_t K, int64_t n_panel_idx, int64_t jr,
        float* ENGINE_RESTRICT C, int64_t ldc, int64_t mr, int64_t nr) noexcept
    {
        __m256 c00 = _mm256_setzero_ps(); __m256 c01 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps(); __m256 c11 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps(); __m256 c21 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps(); __m256 c31 = _mm256_setzero_ps();
        __m256 c40 = _mm256_setzero_ps(); __m256 c41 = _mm256_setzero_ps();
        __m256 c50 = _mm256_setzero_ps(); __m256 c51 = _mm256_setzero_ps();

        const float* a0 = (mr > 0) ? A_row + 0 * lda : nullptr;
        const float* a1 = (mr > 1) ? A_row + 1 * lda : nullptr;
        const float* a2 = (mr > 2) ? A_row + 2 * lda : nullptr;
        const float* a3 = (mr > 3) ? A_row + 3 * lda : nullptr;
        const float* a4 = (mr > 4) ? A_row + 4 * lda : nullptr;
        const float* a5 = (mr > 5) ? A_row + 5 * lda : nullptr;

        for (int64_t k_idx = 0; k_idx < grid_K; ++k_idx) {
            const int64_t k_base = k_idx * KC;
            const int64_t kc = std::min<int64_t>(KC, K - k_base);
            const float* pB = pB_global + b_offsets[k_idx * grid_N + n_panel_idx] + jr * kc;

            for (int64_t k = 0; k < kc; ++k) {
                __m256 b0 = _mm256_load_ps(pB + 0);
                __m256 b1 = _mm256_load_ps(pB + 8);
                __m256 a;
                if (a0) { a = _mm256_broadcast_ss(a0 + k_base + k); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01); }
                if (a1) { a = _mm256_broadcast_ss(a1 + k_base + k); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11); }
                if (a2) { a = _mm256_broadcast_ss(a2 + k_base + k); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21); }
                if (a3) { a = _mm256_broadcast_ss(a3 + k_base + k); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31); }
                if (a4) { a = _mm256_broadcast_ss(a4 + k_base + k); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41); }
                if (a5) { a = _mm256_broadcast_ss(a5 + k_base + k); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51); }
                pB += 16;
            }
        }

        if (nr == 16) {
            if (mr > 0) { _mm256_storeu_ps(C + 0 * ldc + 0, c00); _mm256_storeu_ps(C + 0 * ldc + 8, c01); }
            if (mr > 1) { _mm256_storeu_ps(C + 1 * ldc + 0, c10); _mm256_storeu_ps(C + 1 * ldc + 8, c11); }
            if (mr > 2) { _mm256_storeu_ps(C + 2 * ldc + 0, c20); _mm256_storeu_ps(C + 2 * ldc + 8, c21); }
            if (mr > 3) { _mm256_storeu_ps(C + 3 * ldc + 0, c30); _mm256_storeu_ps(C + 3 * ldc + 8, c31); }
            if (mr > 4) { _mm256_storeu_ps(C + 4 * ldc + 0, c40); _mm256_storeu_ps(C + 4 * ldc + 8, c41); }
            if (mr > 5) { _mm256_storeu_ps(C + 5 * ldc + 0, c50); _mm256_storeu_ps(C + 5 * ldc + 8, c51); }
            return;
        }

        alignas(64) float tempC[96];
        _mm256_store_ps(tempC + 0,  c00); _mm256_store_ps(tempC + 8,  c01);
        _mm256_store_ps(tempC + 16, c10); _mm256_store_ps(tempC + 24, c11);
        _mm256_store_ps(tempC + 32, c20); _mm256_store_ps(tempC + 40, c21);
        _mm256_store_ps(tempC + 48, c30); _mm256_store_ps(tempC + 56, c31);
        _mm256_store_ps(tempC + 64, c40); _mm256_store_ps(tempC + 72, c41);
        _mm256_store_ps(tempC + 80, c50); _mm256_store_ps(tempC + 88, c51);

        for (int64_t i = 0; i < mr; ++i) {
            for (int64_t j = 0; j < nr; ++j) {
                C[i * ldc + j] = tempC[i * 16 + j];
            }
        }
    }


    /* ============================================================================
    *  2. PAKETLİ MICROKERNEL (BÜYÜK KATMANLAR İÇİN: CONV5, 3x3 128->128)
    * ========================================================================== */
    inline void pack_A_fast(int64_t mc, int64_t kc,
                            const float* ENGINE_RESTRICT A, int64_t lda,
                            float* ENGINE_RESTRICT pA) noexcept
    {
        for (int64_t ir_block = 0; ir_block < mc; ir_block += MR) {
            const int64_t remain = std::min<int64_t>(MR, mc - ir_block);
            if (remain == MR) {
                for (int64_t k = 0; k < kc; ++k) {
                    _mm_prefetch(reinterpret_cast<const char*>(A + (ir_block)*lda + k + 32), _MM_HINT_T0);
                    *pA++ = A[(ir_block + 0) * lda + k];
                    *pA++ = A[(ir_block + 1) * lda + k];
                    *pA++ = A[(ir_block + 2) * lda + k];
                    *pA++ = A[(ir_block + 3) * lda + k];
                    *pA++ = A[(ir_block + 4) * lda + k];
                    *pA++ = A[(ir_block + 5) * lda + k];
                }
            } else {
                for (int64_t k = 0; k < kc; ++k) {
                    for (int64_t ir = 0; ir < remain; ++ir) {
                        *pA++ = A[(ir_block + ir) * lda + k];
                    }
                    for (int64_t ir = remain; ir < MR; ++ir) {
                        *pA++ = 0.0f;
                    }
                }
            }
        }
    }

    #define KERNEL_6x16_STEP(P_A, P_B) do { \
        __m256 b0 = _mm256_load_ps((P_B) + 0); \
        __m256 b1 = _mm256_load_ps((P_B) + 8); \
        __m256 a; \
        a = _mm256_broadcast_ss((P_A) + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01); \
        a = _mm256_broadcast_ss((P_A) + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11); \
        a = _mm256_broadcast_ss((P_A) + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21); \
        a = _mm256_broadcast_ss((P_A) + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31); \
        a = _mm256_broadcast_ss((P_A) + 4); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41); \
        a = _mm256_broadcast_ss((P_A) + 5); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51); \
    } while(0)

    inline void micro_kernel_6x16(
        const float* ENGINE_RESTRICT pA,
        const float* ENGINE_RESTRICT pB,
        float* ENGINE_RESTRICT C,
        int64_t kc, int64_t ldc, bool first_time) noexcept
    {
        __m256 c00 = _mm256_setzero_ps(); __m256 c01 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps(); __m256 c11 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps(); __m256 c21 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps(); __m256 c31 = _mm256_setzero_ps();
        __m256 c40 = _mm256_setzero_ps(); __m256 c41 = _mm256_setzero_ps();
        __m256 c50 = _mm256_setzero_ps(); __m256 c51 = _mm256_setzero_ps();

        int64_t k = 0;
        for (; k <= kc - 4; k += 4) {
            _mm_prefetch((const char*)(pB + 256), _MM_HINT_T0);
            _mm_prefetch((const char*)(pA + 96), _MM_HINT_T0);
            KERNEL_6x16_STEP(pA + 0,  pB + 0);
            KERNEL_6x16_STEP(pA + 6,  pB + 16);
            KERNEL_6x16_STEP(pA + 12, pB + 32);
            KERNEL_6x16_STEP(pA + 18, pB + 48);
            pA += 24; pB += 64;
        }
        for (; k < kc; ++k) {
            KERNEL_6x16_STEP(pA + 0, pB + 0);
            pA += 6; pB += 16;
        }

        if (first_time) {
            _mm256_storeu_ps(C + 0 * ldc + 0, c00); _mm256_storeu_ps(C + 0 * ldc + 8, c01);
            _mm256_storeu_ps(C + 1 * ldc + 0, c10); _mm256_storeu_ps(C + 1 * ldc + 8, c11);
            _mm256_storeu_ps(C + 2 * ldc + 0, c20); _mm256_storeu_ps(C + 2 * ldc + 8, c21);
            _mm256_storeu_ps(C + 3 * ldc + 0, c30); _mm256_storeu_ps(C + 3 * ldc + 8, c31);
            _mm256_storeu_ps(C + 4 * ldc + 0, c40); _mm256_storeu_ps(C + 4 * ldc + 8, c41);
            _mm256_storeu_ps(C + 5 * ldc + 0, c50); _mm256_storeu_ps(C + 5 * ldc + 8, c51);
        } else {
            auto add_store = [](float* dst, __m256 v) {
                _mm256_storeu_ps(dst, _mm256_add_ps(_mm256_loadu_ps(dst), v));
            };
            add_store(C + 0 * ldc + 0, c00); add_store(C + 0 * ldc + 8, c01);
            add_store(C + 1 * ldc + 0, c10); add_store(C + 1 * ldc + 8, c11);
            add_store(C + 2 * ldc + 0, c20); add_store(C + 2 * ldc + 8, c21);
            add_store(C + 3 * ldc + 0, c30); add_store(C + 3 * ldc + 8, c31);
            add_store(C + 4 * ldc + 0, c40); add_store(C + 4 * ldc + 8, c41);
            add_store(C + 5 * ldc + 0, c50); add_store(C + 5 * ldc + 8, c51);
        }
    }

    inline void micro_kernel_6x16_edge(
        const float* ENGINE_RESTRICT pA,
        const float* ENGINE_RESTRICT pB,
        float* ENGINE_RESTRICT C,
        int64_t kc, int64_t ldc, bool first_time,
        int64_t mr, int64_t nr) noexcept
    {
        __m256 c00 = _mm256_setzero_ps(); __m256 c01 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps(); __m256 c11 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps(); __m256 c21 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps(); __m256 c31 = _mm256_setzero_ps();
        __m256 c40 = _mm256_setzero_ps(); __m256 c41 = _mm256_setzero_ps();
        __m256 c50 = _mm256_setzero_ps(); __m256 c51 = _mm256_setzero_ps();

        int64_t k = 0;
        for (; k <= kc - 4; k += 4) {
            KERNEL_6x16_STEP(pA + 0,  pB + 0);
            KERNEL_6x16_STEP(pA + 6,  pB + 16);
            KERNEL_6x16_STEP(pA + 12, pB + 32);
            KERNEL_6x16_STEP(pA + 18, pB + 48);
            pA += 24; pB += 64;
        }
        for (; k < kc; ++k) {
            KERNEL_6x16_STEP(pA + 0, pB + 0);
            pA += 6; pB += 16;
        }

        if (nr == 16) {
            auto write_row = [ldc, first_time](float* dst, __m256 v0, __m256 v1) {
                if (first_time) {
                    _mm256_storeu_ps(dst + 0, v0);
                    _mm256_storeu_ps(dst + 8, v1);
                } else {
                    _mm256_storeu_ps(dst + 0, _mm256_add_ps(_mm256_loadu_ps(dst + 0), v0));
                    _mm256_storeu_ps(dst + 8, _mm256_add_ps(_mm256_loadu_ps(dst + 8), v1));
                }
            };
            if (mr > 0) write_row(C + 0 * ldc, c00, c01);
            if (mr > 1) write_row(C + 1 * ldc, c10, c11);
            if (mr > 2) write_row(C + 2 * ldc, c20, c21);
            if (mr > 3) write_row(C + 3 * ldc, c30, c31);
            if (mr > 4) write_row(C + 4 * ldc, c40, c41);
            if (mr > 5) write_row(C + 5 * ldc, c50, c51);
            return;
        }

        alignas(64) float tempC[96];
        _mm256_store_ps(tempC + 0,  c00); _mm256_store_ps(tempC + 8,  c01);
        _mm256_store_ps(tempC + 16, c10); _mm256_store_ps(tempC + 24, c11);
        _mm256_store_ps(tempC + 32, c20); _mm256_store_ps(tempC + 40, c21);
        _mm256_store_ps(tempC + 48, c30); _mm256_store_ps(tempC + 56, c31);
        _mm256_store_ps(tempC + 64, c40); _mm256_store_ps(tempC + 72, c41);
        _mm256_store_ps(tempC + 80, c50); _mm256_store_ps(tempC + 88, c51);

        for (int64_t i = 0; i < mr; ++i) {
            for (int64_t j = 0; j < nr; ++j) {
                if (first_time) C[i * ldc + j]  = tempC[i * 16 + j];
                else            C[i * ldc + j] += tempC[i * 16 + j];
            }
        }
    }
    #undef KERNEL_6x16_STEP

    /* ============================================================================
    *  ANA GİRİŞ
    * ========================================================================== */
    extern "C" {

    void engine_matmul_avx2(
        const float* ENGINE_RESTRICT A,
        const float* ENGINE_RESTRICT B,
        float* ENGINE_RESTRICT C,
        int64_t M, int64_t N, int64_t K) noexcept
    {
        const auto& config = get_gemm_config();
        const int64_t MC = config.mc;
        const int64_t KC = config.kc;
        const int64_t NC = config.nc;

        const float* pB_global = get_packed_B_fast(K, N, B, KC, NC);

        const int64_t grid_K = (K + KC - 1) / KC;
        const int64_t grid_N = (N + NC - 1) / NC;
        
        constexpr size_t STACK_OFFSETS_MAX = 1024;
        std::array<int64_t, STACK_OFFSETS_MAX> stack_b_offsets;
        std::vector<int64_t> heap_b_offsets;
        int64_t* b_offsets = stack_b_offsets.data();

        if (static_cast<size_t>(grid_K * grid_N) > STACK_OFFSETS_MAX) {
            heap_b_offsets.resize(grid_K * grid_N);
            b_offsets = heap_b_offsets.data();
        }

        if (grid_K == 1 && grid_N == 1) {
            b_offsets[0] = 0;
        } else {
            int64_t current_offset = 0;
            for (int64_t k_idx = 0; k_idx < grid_K; ++k_idx) {
                const int64_t k = k_idx * KC;
                const int64_t kc = std::min<int64_t>(KC, K - k);
                for (int64_t j_idx = 0; j_idx < grid_N; ++j_idx) {
                    b_offsets[k_idx * grid_N + j_idx] = current_offset;
                    const int64_t j = j_idx * NC;
                    const int64_t nc = std::min<int64_t>(NC, N - j);
                    current_offset += kc * (((nc + NR - 1) / NR) * NR);
                }
            }
        }



        const int64_t total_mr = (M + MR - 1) / MR;
        const int64_t n_stripes = (N + NR - 1) / NR;
        const bool use_direct_A = (N <= 64 || K <= 128);

        auto& pool = engine::runtime::GlobalPool::get();

        /* BÜYÜK DEĞİŞİM: Havuz range dağıtımı n_workers değil, total_mr'ye verildi! */
        pool.parallel_for(0, total_mr, [&](int64_t mr_start, int64_t mr_end) {
            const int64_t thread_m0 = mr_start * MR;
            const int64_t thread_m1 = std::min<int64_t>(mr_end * MR, M);
            const int64_t cur_M = thread_m1 - thread_m0;
            if (cur_M <= 0) return;

            /* A) ZERO-PACK DIRECT PATH (Conv1, Conv2, 3x3 64->64) 
            K-Fusion eklendi: C matrisi RAM'den ASLA geri okunmaz. */
            if (use_direct_A) {
                for (int64_t i = 0; i + 5 < cur_M; i += 6) {
                    for (int64_t s = 0; s < n_stripes; ++s) {
                        const int64_t col = s * NR;
                        const int64_t nr_actual = std::min<int64_t>(NR, N - col);
                        const int64_t n_panel_idx = col / NC;
                        const int64_t jr = col % NC;

                        if (nr_actual == 16) {
                            compute_6x16_direct_full_K(
                                A + (thread_m0 + i) * K, K,
                                pB_global, b_offsets, grid_K, grid_N,
                                KC, K, n_panel_idx, jr,
                                C + (thread_m0 + i) * N + col, N);
                        } else {
                            compute_6x16_direct_full_K_edge(
                                A + (thread_m0 + i) * K, K,
                                pB_global, b_offsets, grid_K, grid_N,
                                KC, K, n_panel_idx, jr,
                                C + (thread_m0 + i) * N + col, N, 6, nr_actual);
                        }
                    }
                }
                if (cur_M % 6 != 0) {
                    const int64_t i = (cur_M / 6) * 6;
                    const int64_t mr_actual = cur_M - i;
                    for (int64_t s = 0; s < n_stripes; ++s) {
                        const int64_t col = s * NR;
                        const int64_t nr_actual = std::min<int64_t>(NR, N - col);
                        const int64_t n_panel_idx = col / NC;
                        const int64_t jr = col % NC;

                        compute_6x16_direct_full_K_edge(
                            A + (thread_m0 + i) * K, K,
                            pB_global, b_offsets, grid_K, grid_N,
                            KC, K, n_panel_idx, jr,
                            C + (thread_m0 + i) * N + col, N, mr_actual, nr_actual);
                    }
                }
                return;
            }

            /* B) BLIS TILED & PACKED PATH (Conv5, 3x3 128->128, Devasa Matrisler) 
            MC=144 sayesinde pack_A çağrısı minimize edildi. K -> M -> N Döngü sırası */
            thread_local AlignedBuffer t_packed_A;
            t_packed_A.ensure(static_cast<size_t>(MC) * static_cast<size_t>(KC));
            float* pA = t_packed_A.ptr;

            for (int64_t n_idx = 0; n_idx < grid_N; ++n_idx) {
                const int64_t j = n_idx * NC;
                const int64_t nc = std::min<int64_t>(NC, N - j);

                for (int64_t k_idx = 0; k_idx < grid_K; ++k_idx) {
                    const int64_t k = k_idx * KC;
                    const int64_t kc = std::min<int64_t>(KC, K - k);
                    const bool first_time = (k_idx == 0);
                    const float* pB_panel = pB_global + b_offsets[k_idx * grid_N + n_idx];

                    for (int64_t i_start = 0; i_start < cur_M; i_start += MC) {
                        const int64_t mc_actual = std::min<int64_t>(MC, cur_M - i_start);
                        
                        pack_A_fast(mc_actual, kc, A + (thread_m0 + i_start) * K + k, K, pA);

                        for (int64_t jr = 0; jr < nc; jr += NR) {
                            const int64_t nr_actual = std::min<int64_t>(NR, nc - jr);
                            const float* pB_stripe = pB_panel + jr * kc;

                            for (int64_t ir = 0; ir < mc_actual; ir += MR) {
                                const int64_t mr_actual = std::min<int64_t>(MR, mc_actual - ir);
                                if (mr_actual == MR && nr_actual == NR) {
                                    micro_kernel_6x16(pA + ir * kc, pB_stripe,
                                                    C + (thread_m0 + i_start + ir) * N + (j + jr),
                                                    kc, N, first_time);
                                } else {
                                    micro_kernel_6x16_edge(pA + ir * kc, pB_stripe,
                                                        C + (thread_m0 + i_start + ir) * N + (j + jr),
                                                        kc, N, first_time, mr_actual, nr_actual);
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    void engine_matmul_clear_cache() noexcept {
        {
            std::unique_lock<std::shared_mutex> lk(g_matmul_wcache_mtx);
            g_matmul_wcache.clear();
        }
        /* Generation'ı artır — tüm thread'lerin thread_local cache'leri
         * bir sonraki çağrıda geçersiz sayılacak. */
        g_matmul_cache_generation.fetch_add(1, std::memory_order_release);
    }

    } /* extern "C" */

    } // namespace