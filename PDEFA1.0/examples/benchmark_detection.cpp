#include "engine/engine.hpp"
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

namespace {
    struct InitThreads {
        InitThreads() {
            engine::runtime::GlobalPool::set_num_threads(12);
        }
    } g_init;
}

using namespace engine;
using Clock = std::chrono::high_resolution_clock;

int main() {
    printf("=== Engine-AI Detection Model Benchmark ===\n");
    printf("Threads: %d\n\n", runtime::GlobalPool::get().num_threads());

    Model model;
    const char* kModelPath = "../../../tools/converter/mini_detector.engine";
    if (!loader::load_engine_file(kModelPath, model)) {
        const char* kAltPath = "tools/converter/mini_detector.engine";
        if (!loader::load_engine_file(kAltPath, model)) {
            printf("HATA: mini_detector.engine yuklenemedi\n");
            printf("  denenen: %s  ve  %s\n", kModelPath, kAltPath);
            return 1;
        }
    }

    printf("Model:   %s\n", model.name().c_str());
    printf("Weights: %zu\n", model.num_weights());
    printf("Nodes:   %zu\n", model.graph().nodes.size());

    auto report = optimizer::optimize(model);
    printf("Optimizer: Conv+BN=%d, Conv+ReLU=%d, dead=%d\n\n",
           (int)report.conv_bn_fused, (int)report.conv_relu_fused,
           (int)report.nodes_removed);

    Session session(model);

    auto input = Tensor::empty({1, 3, 640, 640}, DType::F32);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < input.numel(); ++i)
        input.data<float>()[i] = dist(rng);

    session.set_input(0, &input);

    printf("Warm-up...\n");
        for (int i = 0; i < 15; ++i) session.run();

    constexpr int RUNS = 100;
    std::vector<double> times;
    times.reserve(RUNS);

    printf("Running %d iterations...\n\n", RUNS);
    for (int r = 0; r < RUNS; ++r) {
        auto t0 = Clock::now();
        session.run();
        auto t1 = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(times.begin(), times.end());
    const double t_min = times.front();
    const double t_med = times[RUNS / 2];
    const double t_max = times.back();
    const double var = (t_max - t_min) / t_min * 100.0;

    const Tensor* out = session.output(0);
    if (out) printf("Output shape: %s\n\n", out->shape().to_string().c_str());

    printf("--- Results (640x640 input) ---\n");
    printf("Min:      %7.3f ms  ->  %6.1f FPS\n", t_min, 1000.0 / t_min);
    printf("Median:   %7.3f ms  ->  %6.1f FPS\n", t_med, 1000.0 / t_med);
    printf("Max:      %7.3f ms  ->  %6.1f FPS\n", t_max, 1000.0 / t_max);
    printf("Variance: %.2f%%\n\n", var);
    return 0;
}