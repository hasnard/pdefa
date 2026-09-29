#include "engine/engine.hpp"
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

extern "C" {
void engine_dequant_int8_avx2(const int8_t*, float*, size_t, float) noexcept;
}

using namespace engine;
using Clock = std::chrono::high_resolution_clock;

int main() {
    constexpr int64_t M = 1024, N = 1024, K = 1024;
    constexpr int RUNS = 30;

    // FP32 baseline
    auto A = Tensor::empty({M, K}, DType::F32);
    auto B = Tensor::empty({K, N}, DType::F32);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < A.numel(); ++i) A.data<float>()[i] = dist(rng);
    for (int64_t i = 0; i < B.numel(); ++i) B.data<float>()[i] = dist(rng);

    auto C = Tensor::empty({M, N}, DType::F32);

    // INT8 B
    std::vector<int8_t> B_q(K * N);
    float B_max = 0.0f;
    for (int64_t i = 0; i < B.numel(); ++i)
        B_max = std::max(B_max, std::fabs(B.data<float>()[i]));
    const float B_scale = B_max / 127.0f;
    for (int64_t i = 0; i < B.numel(); ++i)
        B_q[i] = (int8_t)std::round(B.data<float>()[i] / B_scale);

    // FP32 → FP32 baseline
    {
        for (int i = 0; i < 3; ++i) ops::matmul(A, B, C);
        std::vector<double> ts;
        for (int r = 0; r < RUNS; ++r) {
            auto t0 = Clock::now();
            ops::matmul(A, B, C);
            auto t1 = Clock::now();
            ts.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
        }
        std::sort(ts.begin(), ts.end());
        printf("FP32 matmul:  med %.3f ms  (%.1f GFLOPS)\n",
               ts[RUNS/2], 2.0*M*N*K/(ts[RUNS/2]*1e-3)/1e9);
    }

    // INT8 dequant + FP32 matmul
    std::vector<float> B_dq(K * N);
    {
        // Dequantize
        auto t0 = Clock::now();
        engine_dequant_int8_avx2(B_q.data(), B_dq.data(), K*N, B_scale);
        auto t1 = Clock::now();
        const double dq_ms = std::chrono::duration<double, std::milli>(t1-t0).count();

        auto B_dq_t = Tensor::from_external(B_dq.data(), {K, N}, DType::F32);
        for (int i = 0; i < 3; ++i) ops::matmul(A, B_dq_t, C);
        std::vector<double> ts;
        for (int r = 0; r < RUNS; ++r) {
            auto t0 = Clock::now();
            engine_dequant_int8_avx2(B_q.data(), B_dq.data(), K*N, B_scale);
            ops::matmul(A, B_dq_t, C);
            auto t1 = Clock::now();
            ts.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
        }
        std::sort(ts.begin(), ts.end());
        printf("INT8 dq+mm:   med %.3f ms  (dequant tek başına: %.3f ms)\n",
               ts[RUNS/2], dq_ms);
    }

    return 0;
}