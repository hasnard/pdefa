#include "engine/runtime/cpu_info.hpp"

#include <atomic>
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

/* GCC/Clang x86: __get_cpuid_count burada tanimli.
 * MSVC kullanmaz (intrin.h __cpuidex verir). */
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
#  include <cpuid.h>
#endif

/* Linux ARM64 HWCAP — sadece ARM'de gecerli */
#if defined(__linux__) && (defined(__aarch64__) || defined(__arm__))
#  include <sys/auxv.h>
#  include <asm/hwcap.h>
#endif

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#endif

namespace engine {
namespace runtime {

namespace {

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

inline void cpu_id(uint32_t leaf, uint32_t subleaf, uint32_t out[4]) noexcept {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    out[0] = static_cast<uint32_t>(regs[0]);
    out[1] = static_cast<uint32_t>(regs[1]);
    out[2] = static_cast<uint32_t>(regs[2]);
    out[3] = static_cast<uint32_t>(regs[3]);
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int a = 0, b = 0, c = 0, d = 0;
    __get_cpuid_count(leaf, subleaf, &a, &b, &c, &d);
    out[0] = a; out[1] = b; out[2] = c; out[3] = d;
#else
    (void)leaf; (void)subleaf;
    out[0] = out[1] = out[2] = out[3] = 0;
#endif
}

inline uint32_t cpu_id_max_leaf() noexcept {
    uint32_t r[4] = {0};
    cpu_id(0, 0, r);
    return r[0];
}

inline bool bit_set(uint32_t reg, uint32_t bit) noexcept {
    return (reg >> bit) & 1u;
}

void detect_x86_features(CpuFeatures& f) noexcept {
    const uint32_t max_leaf = cpu_id_max_leaf();
    if (max_leaf < 1) return;

    uint32_t regs[4] = {0};
    cpu_id(0x01, 0, regs);
    const uint32_t ecx1 = regs[2];
    const uint32_t edx1 = regs[3];

    f.sse    = bit_set(edx1, 25);
    f.sse2   = bit_set(edx1, 26);
    f.sse3   = bit_set(ecx1, 0);
    f.ssse3  = bit_set(ecx1, 9);
    f.sse41  = bit_set(ecx1, 19);
    f.sse42  = bit_set(ecx1, 20);
    f.avx    = bit_set(ecx1, 28);
    f.fma    = bit_set(ecx1, 12);
    f.f16c   = bit_set(ecx1, 29);
    f.popcnt = bit_set(ecx1, 23);
    f.aes    = bit_set(ecx1, 25);
    f.rdrand = bit_set(ecx1, 30);

    if (max_leaf >= 7) {
        cpu_id(0x07, 0, regs);
        f.avx2     = bit_set(regs[1], 5);
        f.bmi1     = bit_set(regs[1], 3);
        f.bmi2     = bit_set(regs[1], 8);
        f.avx512f  = bit_set(regs[1], 16);
        f.avx512dq = bit_set(regs[1], 17);
        f.avx512cd = bit_set(regs[1], 28);
        f.avx512bw = bit_set(regs[1], 30);
        f.avx512vl = bit_set(regs[1], 31);

        cpu_id(0x07, 1, regs);
        f.avx512bf16 = bit_set(regs[0], 5);
        f.avx512vnni = bit_set(regs[0], 4);
    }

#ifndef ENGINE_USE_AVX2
    f.avx2 = false;
#endif
#ifndef ENGINE_USE_AVX512
    f.avx512f = f.avx512cd = f.avx512bw = false;
    f.avx512dq = f.avx512vl = f.avx512vnni = f.avx512bf16 = false;
#endif
#ifndef ENGINE_USE_FMA
    f.fma = false;
#endif
}

void detect_cache_x86(CacheInfo& c) noexcept {
    const uint32_t max_leaf = cpu_id_max_leaf();
    if (max_leaf < 4) return;

    for (uint32_t i = 0; i < 16; ++i) {
        uint32_t regs[4] = {0};
        cpu_id(0x04, i, regs);
        const uint32_t type = regs[0] & 0x1F;
        if (type == 0) break;
        const uint32_t level = (regs[0] >> 5) & 0x7;
        const uint32_t ways  = ((regs[1] >> 22) & 0x3FF) + 1;
        const uint32_t parts = ((regs[1] >> 12) & 0x3FF) + 1;
        const uint32_t line  = (regs[1] & 0xFFF) + 1;
        const uint32_t sets  = regs[2] + 1;
        const size_t   size  = static_cast<size_t>(ways) * parts * line * sets;

        if (level == 1 && type == 1) { c.l1_data = size; c.l1_assoc = static_cast<int32_t>(ways); }
        if (level == 1 && type == 2) { c.l1_instruction = size; }
        if (level == 2)              { c.l2 = size; c.l2_assoc = static_cast<int32_t>(ways); }
        if (level == 3)              { c.l3 = size; c.l3_assoc = static_cast<int32_t>(ways); }
    }
}

std::string detect_brand_x86() noexcept {
    char brand[49] = {0};
    uint32_t regs[4];
    for (uint32_t i = 0; i < 3; ++i) {
        cpu_id(0x80000002u + i, 0, regs);
        std::memcpy(brand + i * 16 + 0,  &regs[0], 4);
        std::memcpy(brand + i * 16 + 4,  &regs[1], 4);
        std::memcpy(brand + i * 16 + 8,  &regs[2], 4);
        std::memcpy(brand + i * 16 + 12, &regs[3], 4);
    }
    brand[48] = '\0';
    std::string s(brand);
    const auto first = s.find_first_not_of(" \t");
    if (first != std::string::npos) s.erase(0, first);
    return s;
}

std::string detect_vendor_x86() noexcept {
    uint32_t regs[4] = {0};
    cpu_id(0, 0, regs);
    char v[13];
    std::memcpy(v + 0, &regs[1], 4);
    std::memcpy(v + 4, &regs[3], 4);
    std::memcpy(v + 8, &regs[2], 4);
    v[12] = '\0';
    return std::string(v);
}

#endif /* x86 */

TopologyInfo detect_topology() noexcept {
    TopologyInfo t;
    unsigned hw = std::thread::hardware_concurrency();
    t.logical_cores = (hw > 0) ? static_cast<int32_t>(hw) : 1;
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    uint32_t regs[4] = {0};
    cpu_id(0x01, 0, regs);
    t.has_hyperthread = bit_set(regs[3], 28);
    t.physical_cores = t.has_hyperthread ? (t.logical_cores / 2) : t.logical_cores;
#else
    t.physical_cores = t.logical_cores;
#endif
    if (t.physical_cores <= 0) t.physical_cores = t.logical_cores;
    t.numa_nodes = 1;
    return t;
}

std::once_flag g_init_flag;
CpuInfo g_cpu_info{};

void init_cpu_info() noexcept {
    auto& info = g_cpu_info;

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    detect_x86_features(info.features);
    detect_cache_x86(info.cache);
    info.brand  = detect_brand_x86();
    info.vendor = detect_vendor_x86();
#endif

    if (info.cache.l1_data == 0) info.cache.l1_data = 32 * 1024;
    if (info.cache.l2 == 0)      info.cache.l2      = 256 * 1024;
    if (info.cache.l3 == 0)      info.cache.l3      = 8 * 1024 * 1024;

    info.topology = detect_topology();
    if (info.brand.empty()) info.brand = "Unknown CPU";
}

} /* anonymous namespace */

const CpuInfo& Cpu::info() noexcept {
    std::call_once(g_init_flag, init_cpu_info);
    return g_cpu_info;
}
const CpuFeatures& Cpu::features() noexcept { return info().features; }
const CacheInfo&   Cpu::cache() noexcept    { return info().cache; }
const TopologyInfo& Cpu::topology() noexcept { return info().topology; }

CpuFeatures::Level Cpu::simd_level() noexcept {
    return info().features.best_level();
}

const char* Cpu::level_name(CpuFeatures::Level lvl) noexcept {
    switch (lvl) {
        case CpuFeatures::Level::Scalar: return "scalar";
        case CpuFeatures::Level::SSE:    return "SSE";
        case CpuFeatures::Level::SSE42:  return "SSE4.2";
        case CpuFeatures::Level::AVX:    return "AVX";
        case CpuFeatures::Level::AVX2:   return "AVX2";
        case CpuFeatures::Level::AVX512: return "AVX-512";
        case CpuFeatures::Level::NEON:   return "NEON";
    }
    return "unknown";
}

std::string CpuInfo::summary() const {
    std::ostringstream os;
    os << "CPU: " << brand << " (" << vendor << ")\n";
    os << "  SIMD: ";
    if (features.has_avx512()) os << "AVX-512 ";
    if (features.avx2)         os << "AVX2 ";
    if (features.fma)          os << "FMA ";
    os << "\n";
    os << "  Cores: " << topology.logical_cores << " logical, "
       << topology.physical_cores << " physical\n";
    return os.str();
}

} /* namespace runtime */
} /* namespace engine */