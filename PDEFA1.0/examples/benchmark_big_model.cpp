#include "engine/engine.hpp"
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <mutex>

namespace {

    struct Profiler {
        std::mutex mtx;
        std::unordered_map<std::string, std::pair<double,int>> stats;

        static void cb(const char* name, double ms, void* user) noexcept {
            auto* self = static_cast<Profiler*>(user);
            std::lock_guard<std::mutex> lk(self->mtx);
            auto& e = self->stats[name];
            e.first  += ms;
            e.second += 1;
        }

        void dump() const {
            std::vector<std::pair<std::string, std::pair<double,int>>> v(
                stats.begin(), stats.end());
            std::sort(v.begin(), v.end(),
                      [](const auto& a, const auto& b){
                          return a.second.first > b.second.first;
                      });
            printf("\n=== PER-NODE PROFILE ===\n");
            printf("%-30s %12s %8s %12s\n", "Node", "Toplam(ms)", "Cagri", "Ort(ms)");
            printf("%-30s %12s %8s %12s\n", "----", "---------", "-----", "-------");
            double total = 0.0;
            for (const auto& kv : v) {
                printf("%-30s %12.3f %8d %12.3f\n",
                       kv.first.c_str(),
                       kv.second.first,
                       kv.second.second,
                       kv.second.first / kv.second.second);
                total += kv.second.first;
            }
            printf("%-30s %12.3f\n", "TOPLAM", total);
            printf("\n");
        }
    };

    struct InitThreads {
        InitThreads() {
            engine::runtime::GlobalPool::set_num_threads(12);
        }
    } g_init;
}

using namespace engine;
using Clock = std::chrono::high_resolution_clock;

int main() {
    printf("=== Engine-AI Big Model Benchmark ===\n");
    printf("Threads: %d\n\n", runtime::GlobalPool::get().num_threads());

    Model model;
    const char* kModelPath = "../../../tools/converter/big_model.engine";
    if (!loader::load_engine_file(kModelPath, model)) {
        const char* kAltPath = "tools/converter/big_model.engine";
        if (!loader::load_engine_file(kAltPath, model)) {
            printf("HATA: big_model.engine yuklenemedi\n");
            printf("  denenen: %s  ve  %s\n", kModelPath, kAltPath);
            return 1;
        }
    }

    printf("Weights: %zu  Nodes: %zu\n", model.num_weights(), model.graph().nodes.size());

    auto report = optimizer::optimize(model);
    printf("Optimizer: Conv+BN=%d, Conv+ReLU=%d, dead=%d\n\n",
           (int)report.conv_bn_fused, (int)report.conv_relu_fused,
           (int)report.nodes_removed);

    Session session(model);

    Profiler prof;
    session.set_profile_callback(&Profiler::cb, &prof);

    auto input = Tensor::empty({1, 3, 640, 640}, DType::F32);

    FILE* f_in = std::fopen("../../../tools/converter/test_input.f32", "rb");
    if (!f_in) f_in = std::fopen("tools/converter/test_input.f32", "rb");
    if (f_in) {
        std::fread(input.data<float>(), sizeof(float), input.numel(), f_in);
        std::fclose(f_in);
        std::printf("[correctness] test_input.f32 yuklendi\n");
    } else {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int64_t i = 0; i < input.numel(); ++i)
            input.data<float>()[i] = dist(rng);
        std::printf("[correctness] Rastgele input (test_input.f32 yok)\n");
    }

    session.set_input(0, &input);

    for (int i = 0; i < 15; ++i) session.run();

    /* ---- Profil: 1 run ---- */
    {
        std::lock_guard<std::mutex> lk(prof.mtx);
        prof.stats.clear();
    }
    session.run();
    prof.dump();

    constexpr int RUNS = 100;
    std::vector<double> times;
    times.reserve(RUNS);

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
    if (out) {
        printf("Output: %s\n\n", out->shape().to_string().c_str());

        FILE* f_out = std::fopen("test_engine_output.f32", "wb");
        if (f_out) {
            std::fwrite(out->data<float>(), sizeof(float), out->numel(), f_out);
            std::fclose(f_out);
            std::printf("[correctness] test_engine_output.f32 yazildi\n");
        }
    }

    printf("--- Engine-AI ---\n");
    printf("Min:      %8.3f ms  ->  %6.1f FPS\n", t_min, 1000.0 / t_min);
    printf("Median:   %8.3f ms  ->  %6.1f FPS\n", t_med, 1000.0 / t_med);
    printf("Max:      %8.3f ms  ->  %6.1f FPS\n", t_max, 1000.0 / t_max);
    printf("Variance: %.1f%%\n", var);
    return 0;
}