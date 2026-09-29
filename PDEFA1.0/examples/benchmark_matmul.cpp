/* ============================================================================
 *  examples/benchmark_matmul.cpp
 *  Engine-AI MatMul — 8 farklı şekil
 *  Çıktı: "  {label}   {med} ms   {gflops} GFLOPS"
 * ========================================================================== */
#include "engine/engine.hpp"
#include <cstdio>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>

using namespace engine;
using Clock = std::chrono::high_resolution_clock;

struct BenchCase { int64_t M, K, N; const char* label; };

static const BenchCase kShapes[] = {
    {128,  128,  128,  "128^3"},
    {256,  256,  256,  "256^3"},
    {512,  512,  512,  "512^3"},
    {1024, 1024, 1024, "1024^3"},
    {2048, 2048, 2048, "2048^3"},
    {1,    4096, 4096, "1x4096x4096 (LLM)"},
    {196,  768,  768,  "196x768x768 (ViT)"},
    {1024, 2048, 512,  "1024x2048x512"},
};

static void bench_one(int64_t M, int64_t K, int64_t N, const char* label)
{
    auto A = Tensor::empty({M, K}, DType::F32);
    auto B = Tensor::empty({K, N}, DType::F32);
    auto C = Tensor::empty({M, N}, DType::F32);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* ap = A.data<float>();
    float* bp = B.data<float>();
    for (int64_t i = 0; i < M * K; ++i) ap[i] = dist(rng);
    for (int64_t i = 0; i < K * N; ++i) bp[i] = dist(rng);

    constexpr int WARMUP = 5;
    for (int i = 0; i < WARMUP; ++i) ops::matmul(A, B, C);

    constexpr int RUNS = 20;
    std::vector<double> times;
    times.reserve(RUNS);
    for (int r = 0; r < RUNS; ++r) {
        auto t0 = Clock::now();
        ops::matmul(A, B, C);
        auto t1 = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    const double t_med = times[RUNS / 2];

    const double flops = 2.0 * M * K * N;
    const double gflops = flops / (t_med * 1e-3) / 1e9;

    printf("  %-22s %8.3f ms   %8.1f GFLOPS\n", label, t_med, gflops);
}

int main() {
    runtime::GlobalPool::set_num_threads(12);

    printf("=== Engine-AI MatMul Benchmark ===\n");
    printf("CPU:     %s\n", runtime::Cpu::info().brand.c_str());
    printf("SIMD:    %s\n", simd_level());
    printf("Threads: %d\n\n", runtime::GlobalPool::get().num_threads());

    const size_t n = sizeof(kShapes) / sizeof(kShapes[0]);
    for (size_t i = 0; i < n; ++i) {
        const BenchCase& s = kShapes[i];
        bench_one(s.M, s.K, s.N, s.label);
    }
    return 0;
}