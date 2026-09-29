/* ============================================================================
 *  include/engine/runtime/cpu_info.hpp
 *  Engine-AI — Runtime CPU özellik tespiti
 *
 *  NE YAPAR?
 *    - CPUID komutuyla çalışma zamanında CPU özelliklerini tespit eder
 *    - SIMD desteği: SSE/AVX/AVX2/AVX-512/FMA
 *    - Cache boyutları: L1/L2/L3
 *    - Çekirdek sayısı: logical/physical
 *    - Marka ismi: "Intel Core i9-13900K" gibi
 *    - NUMA node sayısı (Linux)
 *
 *  NEDEN GEREKLİ?
 *    - Kernel dispatch: "Bu CPU AVX-512 destekliyor mu?" → runtime karar
 *    - Thread pool: "Kaç çekirdek var?" → thread sayısı
 *    - Matmul tile: "L1 cache 32 KB mı?" → blok boyutu
 *    - AVX-512 kod derlenebilir ama sadece destekleyen CPU'da çalışır
 *
 *  ÇOKLU ÇEKİRDEK DESTEĞİ:
 *    - Singleton: ilk çağrıda bir kez tespit edilir (once_flag)
 *    - Sonraki okumalar lock-free (sadece const pointer)
 *    - Her thread aynı CPU'da çalışır → tutarlı görünüm
 *
 *  PLATFORM DESTEĞİ:
 *    x86/x64  → __cpuid / __cpuidex ile tam tespit
 *    ARM64    → getauxval (Linux), sysctl (macOS)
 *    Fallback → sadece compile-time makroları kullanır
 * ========================================================================== */

#ifndef ENGINE_RUNTIME_CPU_INFO_HPP
#define ENGINE_RUNTIME_CPU_INFO_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace engine {
namespace runtime {

/* ============================================================================
 *  CPU ÖZELLİKLERİ (POD yapı — hızlı kopyalanabilir)
 * ========================================================================== */
struct CpuFeatures {
    /* ---- x86 SIMD ---- */
    bool sse      = false;
    bool sse2     = false;
    bool sse3     = false;
    bool ssse3    = false;
    bool sse41    = false;
    bool sse42    = false;
    bool avx      = false;
    bool avx2     = false;
    bool fma      = false;
    bool f16c     = false;   // half precision dönüşümü

    /* ---- AVX-512 ailesi ---- */
    bool avx512f    = false;   // Foundation (zorunlu)
    bool avx512cd   = false;   // Conflict detection
    bool avx512bw   = false;   // Byte & word
    bool avx512dq   = false;   // Doubleword & quadword
    bool avx512vl   = false;   // Vector length (128/256'da 512 op)
    bool avx512vnni = false;   // INT8 dot product (quantization için)
    bool avx512bf16 = false;   // BF16 desteği

    /* ---- ARM NEON ---- */
    bool neon     = false;
    bool neon_fp16 = false;
    bool sve      = false;

    /* ---- Yardımcılar ---- */
    bool popcnt   = false;
    bool bmi1     = false;
    bool bmi2     = false;
    bool aes      = false;
    bool rdrand   = false;

    /* AVX-512 kullanılabilir mi? (tüm temel parçalar var mı) */
    [[nodiscard]] bool has_avx512() const noexcept {
        return avx512f && avx512cd && avx512bw && avx512dq && avx512vl;
    }

    /* En iyi SIMD seviyesi (dispatch için) */
    enum class Level : uint8_t {
        Scalar  = 0,
        SSE     = 1,
        SSE42   = 2,
        AVX     = 3,
        AVX2    = 4,
        AVX512  = 5,
        NEON    = 10,   // ARM
    };

    [[nodiscard]] Level best_level() const noexcept {
        if (has_avx512()) return Level::AVX512;
        if (avx2)         return Level::AVX2;
        if (avx)          return Level::AVX;
        if (sse42)        return Level::SSE42;
        if (sse)          return Level::SSE;
        if (neon)         return Level::NEON;
        return Level::Scalar;
    }
};

/* ============================================================================
 *  CACHE BİLGİLERİ
 * ========================================================================== */
struct CacheInfo {
    size_t l1_data       = 0;   // byte
    size_t l1_instruction = 0;
    size_t l2            = 0;
    size_t l3            = 0;
    size_t cache_line    = 64;  // varsayılan 64
    int32_t l1_assoc     = 0;   // associativity
    int32_t l2_assoc     = 0;
    int32_t l3_assoc     = 0;

    /* Kaç float L1'e sığar? (matmul tile için kritik) */
    [[nodiscard]] size_t l1_floats() const noexcept {
        return l1_data / sizeof(float);
    }
    [[nodiscard]] size_t l2_floats() const noexcept {
        return l2 / sizeof(float);
    }
};

/* ============================================================================
 *  TOPOLOJİ (çekirdek bilgisi)
 * ========================================================================== */
struct TopologyInfo {
    int32_t logical_cores   = 0;   // Hyperthreading dahil
    int32_t physical_cores  = 0;   // Fiziksel çekirdek
    int32_t numa_nodes      = 1;
    bool    has_hyperthread = false;

    /* Verimli thread sayısı — fiziksel çekirdek sayısı önerilir.
     * Hyperthreading SIMD workload'da %10-20 kazandırır, %100 değil. */
    [[nodiscard]] int32_t recommended_threads() const noexcept {
        /* SIMD workload'da SMT ciddi fark yaratır (Conv/GEMM).
         * Fiziksel çekirdek sayısına bölmek yerine logical_cores kullan. */
        return logical_cores > 0 ? logical_cores : 1;
    }
};

/* ============================================================================
 *  TÜM CPU BİLGİSİ (bir arada)
 * ========================================================================== */
struct CpuInfo {
    CpuFeatures    features;
    CacheInfo      cache;
    TopologyInfo   topology;
    std::string    brand;          // "Intel Core i9-13900K"
    std::string    vendor;         // "GenuineIntel" / "AuthenticAMD"
    int32_t        family  = 0;
    int32_t        model   = 0;
    int32_t        stepping = 0;

    /* Debug/log için özet */
    [[nodiscard]] std::string summary() const;
};

/* ============================================================================
 *  ANA ERİŞİM NOKTASI (SINGLETON)
 * ========================================================================== */

class Cpu {
public:
    /* Tüm bilgiye eriş. İlk çağrıda tespit edilir (thread-safe).
     * Sonraki çağrılar: sadece const reference döner (lock-free). */
    [[nodiscard]] static const CpuInfo& info() noexcept;

    /* Kısayollar (sıcak yolda kullanılır) */
    [[nodiscard]] static const CpuFeatures& features() noexcept;
    [[nodiscard]] static const CacheInfo&   cache()    noexcept;
    [[nodiscard]] static const TopologyInfo& topology() noexcept;

    /* Tek satırda SIMD seviyesi */
    [[nodiscard]] static CpuFeatures::Level simd_level() noexcept;

    /* Feature isimleri (log için) */
    [[nodiscard]] static const char* level_name(CpuFeatures::Level lvl) noexcept;

    Cpu() = delete;
};

/* ============================================================================
 *  KOLAYLIK MAKROLARI (dispatch için)
 * ========================================================================== */

#define ENGINE_CPU_HAS_AVX2()    (::engine::runtime::Cpu::features().avx2)
#define ENGINE_CPU_HAS_AVX512()  (::engine::runtime::Cpu::features().has_avx512())
#define ENGINE_CPU_HAS_FMA()     (::engine::runtime::Cpu::features().fma)
#define ENGINE_CPU_HAS_NEON()    (::engine::runtime::Cpu::features().neon)

/* ============================================================================
 *  İLERİ BİLDİRİM (kernel dispatch ileride kullanacak)
 * ========================================================================== */

/* Aligned allocator için doğru hizalama boyutu.
 * AVX-512 varsa 64, AVX2 varsa 32, yoksa 16. */
[[nodiscard]] inline size_t recommended_alignment() noexcept {
    const auto& f = Cpu::features();
    if (f.has_avx512()) return 64;
    if (f.avx2)         return 32;
    if (f.sse)          return 16;
    return 8;
}

} /* namespace runtime */
} /* namespace engine */

#endif /* ENGINE_RUNTIME_CPU_INFO_HPP */
