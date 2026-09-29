#include "engine/engine.hpp"
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <thread>

/* Conv dispatcher'ın hangi yolu seçtiğini gösteren thread-local string.
 * Tanımı src/ops/conv2d.cpp içindedir (anonim namespace DIŞINDA). */
extern thread_local const char* engine_conv_last_path;

/* ⬅️ GLOBAL static — main'den ÖNCE çalışır */
namespace {
    struct ThreadInit {
        ThreadInit() {
            engine::runtime::GlobalPool::set_num_threads(12);
        }
    } g_thread_init;
}

using namespace engine;
using Clock = std::chrono::high_resolution_clock;


/* Benchmark: YOLOv5n-tarzı bir conv katmanı
 *   input:  [1, H, W, Cin]  (NHWC)
 *   weight: [Cout, Cin, KH, KW]  (KCRS)
 *   output: [1, Hout, Wout, Cout] (NHWC) */
void bench_conv2d(const char* name,
                  int64_t N, int64_t Cin, int64_t H, int64_t W,
                  int64_t Cout, int64_t KH, int64_t KW,
                  int64_t stride, int64_t pad)
{
    auto input  = Tensor::empty({N, H, W, Cin}, DType::F32);
    auto weight = Tensor::empty({Cout, Cin, KH, KW}, DType::F32);
    auto bias   = Tensor::empty({Cout}, DType::F32);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < input.numel(); ++i)  input.data<float>()[i]  = dist(rng);
    for (int64_t i = 0; i < weight.numel(); ++i) weight.data<float>()[i] = dist(rng);
    for (int64_t i = 0; i < bias.numel(); ++i)   bias.data<float>()[i]   = dist(rng);

    ops::Conv2dConfig cfg;
    cfg.stride_h = stride; cfg.stride_w = stride;
    cfg.pad_h = pad; cfg.pad_w = pad;

    /* Warm-up — dispatcher yolu burada belirler */
    engine_conv_last_path = "none";
    auto out = ops::conv2d(input, weight, &bias, cfg);
    if (out.is_empty()) {
        printf("  %-30s FAILED (empty output)\n", name);
        return;
    }
    const char* path = engine_conv_last_path ? engine_conv_last_path : "?";

    const double flops = 2.0 * N * Cout * Cin * KH * KW *
                         out.dim(1) * out.dim(2) / 1e9;  /* GFLOP */

    constexpr int RUNS = 30;
    std::vector<double> times;
    times.reserve(RUNS);
    for (int r = 0; r < RUNS; ++r) {
        auto t0 = Clock::now();
        ops::conv2d(input, weight, &bias, cfg);
        auto t1 = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        /* Termal throttle'a karşı küçük nefes */
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    std::sort(times.begin(), times.end());
    const double t_med = times[RUNS / 2];
    const double t_min = times.front();
    const double t_max = times.back();
    const double variance = (t_max - t_min) / t_min * 100.0;

    /* İnsan-okur satır: hem median hem MIN hem path */
    printf("  %-30s med %7.3f  MIN %7.3f ms  (%6.2f GFLOPS @MIN)  path=%-12s  var=%.1f%%\n",
           name, t_med, t_min, flops / (t_min * 1e-3), path, variance);

    /* CSV: MIN kullanıyoruz (termal throttle'a dirençli) */
    printf("CSV,%s,%.3f,%.2f\n", name, t_min, flops / (t_min * 1e-3));

    /* Katmanlar arası soğuma */
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

int main() {
    printf("=== Engine-AI Conv2d Benchmark ===\n\n");
    printf("name,ms_min,gflops\n");
    printf("CPU:      %s\n", runtime::Cpu::info().brand.c_str());
    printf("SIMD:     %s\n", simd_level());
    printf("Threads:  %d\n\n", runtime::GlobalPool::get().num_threads());

    printf("--- YOLOv5n-style layers ---\n");
    bench_conv2d("Conv1  3->16  640x640", 1, 3, 640, 640, 16, 3, 3, 2, 1);
    bench_conv2d("Conv2  16->32 320x320", 1, 16, 320, 320, 32, 3, 3, 2, 1);
    bench_conv2d("Conv3  32->64 160x160", 1, 32, 160, 160, 64, 3, 3, 2, 1);
    bench_conv2d("Conv4  64->128 80x80",  1, 64, 80, 80, 128, 3, 3, 2, 1);
    bench_conv2d("Conv5  128->256 40x40", 1, 128, 40, 40, 256, 3, 3, 2, 1);
    bench_conv2d("Conv6  256->512 20x20", 1, 256, 20, 20, 512, 3, 3, 2, 1);

    printf("\n--- 1x1 conv (pointwise) ---\n");
    bench_conv2d("1x1   64->128 80x80", 1, 64, 80, 80, 128, 1, 1, 1, 0);
    bench_conv2d("1x1   128->256 40x40", 1, 128, 40, 40, 256, 1, 1, 1, 0);

    printf("\n--- 3x3 conv, larger channels ---\n");
    bench_conv2d("3x3   64->64 80x80",  1, 64, 80, 80, 64, 3, 3, 1, 1);
    bench_conv2d("3x3   128->128 40x40", 1, 128, 40, 40, 128, 3, 3, 1, 1);

    printf("\n");
    return 0;
}