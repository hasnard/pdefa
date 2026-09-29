#include "engine/loader/custom_loader.hpp"
#include "engine/session.hpp"
#include "engine/tensor.hpp"
#include "engine/optimizer/graph_optimizer.hpp"

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <mutex>
#include <string>

using namespace engine;

int main() {
    runtime::GlobalPool::set_num_threads(12);
    Model model;
    const char* kModelPath = "tools/converter/yolov8s.engine";
    if (!loader::load_engine_file(kModelPath, model)) {
        const char* kAltPath = "../../../tools/converter/yolov8s.engine";
        if (!loader::load_engine_file(kAltPath, model)) {
            std::printf("!!! model load FAIL\n");
            return 1;
        }
    }

    std::printf("=== Model loaded (before optimize) ===\n");
    std::printf("  nodes   = %zu\n", model.graph().nodes.size());
    std::printf("  weights = %zu\n", model.num_weights());

    /* ============ OPTİMİZER — Session'dan ÖNCE ============ */
    auto report = optimizer::optimize(model);
    std::printf("=== Optimizer ===\n");
    std::printf("  Conv+BN fused:   %d\n", (int)report.conv_bn_fused);
    std::printf("  Conv+ReLU fused: %d\n", (int)report.conv_relu_fused);
    std::printf("  Dead nodes:      %d\n", (int)report.nodes_removed);
    std::printf("  Nodes after:     %zu\n", model.graph().nodes.size());
    
        /* === WEIGHT DTYPE KONTROLÜ === */
    {
        std::printf("=== WEIGHT DTYPE CHECK (ilk 30) ===\n");
        const int32_t nw = model.num_weights();
        for (int32_t wi = 0; wi < std::min(30, nw); ++wi) {
            const Tensor& w = model.weight(wi);
            std::printf("  w[%2d] dtype=%-3d shape=[", wi, (int)w.dtype());
            for (int32_t d = 0; d < w.rank(); ++d)
                std::printf("%s%lld", (d?",":""), (long long)w.dim(d));
            std::printf("]\n");
        }
    }
        /* İlk 15 node: op, name, inputs, outputs */
    std::printf("\n=== İLK 15 NODE ===\n");
    for (size_t ni = 0; ni < std::min<size_t>(15, model.graph().nodes.size()); ++ni) {
        const auto& n = model.graph().nodes[ni];
        std::printf("  n[%2zu] op=%-3d '%-38s' in=[",
                    ni, (int)n.op, n.name.c_str());
        for (size_t ii = 0; ii < n.inputs.size(); ++ii)
            std::printf("%d%s", n.inputs[ii], (ii+1<n.inputs.size())?",":"");
        std::printf("] out=[");
        for (size_t oi = 0; oi < n.outputs.size(); ++oi)
            std::printf("%d%s", n.outputs[oi], (oi+1<n.outputs.size())?",":"");
        std::printf("]\n");
    }
    Session session(model);
        /* Model içindeki op dağılımı */
    {
        std::unordered_map<int, int> op_count;
        std::unordered_map<int, std::string> op_sample;
        for (const auto& n : model.graph().nodes) {
            op_count[(int)n.op]++;
            op_sample[(int)n.op] = n.name;
        }
        std::vector<std::pair<int,int>> sorted(op_count.begin(), op_count.end());
        std::sort(sorted.begin(), sorted.end());
        std::printf("=== NODE-OP DISTRIBUTION ===\n");
        for (auto& [op, cnt] : sorted) {
            std::printf("  op=%3d  count=%3d  sample='%s'\n",
                        op, cnt, op_sample[op].c_str());
        }
    }
    /* ---- PROFİL CALLBACK ---- */
    static std::unordered_map<std::string, std::pair<double,int>> g_stats;
    static std::mutex g_mtx;
    session.set_profile_callback(
        [](const char* name, double ms, void*) noexcept {
            std::lock_guard<std::mutex> lk(g_mtx);
            auto& e = g_stats[name];
            e.first += ms; e.second++;
        }, nullptr);

    std::vector<float> input(1 * 3 * 640 * 640);
    FILE* f = std::fopen("tools/converter/yolov8s_input.f32", "rb");
    if (!f) f = std::fopen("../../../tools/converter/yolov8s_input.f32", "rb");
    if (!f) { std::printf("!!! input f32 yok\n"); return 1; }
    std::fread(input.data(), 4, input.size(), f);
    std::fclose(f);

    int64_t dims[4] = {1, 3, 640, 640};
    Tensor t_in = Tensor::from_external(input.data(), Shape(dims, 4), DType::F32);
    session.set_input(0, &t_in);

    /* ---- 1) Isınma: 10 run ---- */
    std::printf("=== Warmup (10 run) ===\n");
    for (int i = 0; i < 10; ++i) {
        if (!session.run()) { std::printf("!!! warmup FAIL\n"); return 2; }
    }

    /* ---- 2) Stats'ı temizle ---- */
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_stats.clear();
    }

    /* ---- 3) WARM CACHE'de 1 profil run ---- */
    std::printf("=== Profil run (warm cache) ===\n");
    if (!session.run()) { std::printf("!!! profile run FAIL\n"); return 3; }

    if (auto* out = session.output(0)) {
        std::printf("  output shape=[");
        for (int i = 0; i < out->rank(); ++i)
            std::printf("%lld%s", (long long)out->dim(i),
                        (i + 1 < out->rank()) ? "," : "");
        std::printf("]\n");
    }

    /* ---- 4) PER-OP PROFILE DUMP ---- */
    {
        std::unordered_map<std::string, std::pair<double,int>> by_op;
        std::lock_guard<std::mutex> lk(g_mtx);
        for (auto& [k, v] : g_stats) {
            auto op_pos = k.find("_op");
            if (op_pos == std::string::npos) continue;
            auto sp = k.find(' ', op_pos);
            std::string op = k.substr(op_pos + 3,
                                     (sp == std::string::npos ? k.size() : sp) - op_pos - 3);
            auto& e = by_op[op];
            e.first += v.first;
            e.second += v.second;
        }
        std::vector<std::pair<std::string, std::pair<double,int>>> v(
            by_op.begin(), by_op.end());
        std::sort(v.begin(), v.end(),
                  [](auto& a, auto& b){ return a.second.first > b.second.first; });

        std::printf("\n=== PER-OP PROFILE (YOLOv8s, warm) ===\n");
        std::printf("%-8s %12s %8s %12s\n", "op_id", "Toplam(ms)", "Sayi", "Ort(ms)");
        double total = 0.0;
        for (auto& kv : v) {
            std::printf("%-8s %12.3f %8d %12.3f\n",
                        kv.first.c_str(),
                        kv.second.first, kv.second.second,
                        kv.second.first / kv.second.second);
            total += kv.second.first;
        }
        std::printf("%-8s %12.3f\n", "TOPLAM", total);

        std::printf("\n=== EN PAHALI 30 NODE ===\n");
        std::vector<std::pair<std::string, std::pair<double,int>>> all(
            g_stats.begin(), g_stats.end());
        std::sort(all.begin(), all.end(),
                  [](auto& a, auto& b){ return a.second.first > b.second.first; });
        for (size_t i = 0; i < std::min<size_t>(30, all.size()); ++i) {
            std::printf("  %-45s %10.3f ms\n",
                        all[i].first.c_str(), all[i].second.first);
        }
    }

    /* ---- 5) Benchmark ---- */
    constexpr int WARMUP2 = 5, RUNS = 50;
    for (int i = 0; i < WARMUP2; ++i) session.run();

    std::vector<double> ts;
    ts.reserve(RUNS);
    for (int i = 0; i < RUNS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        session.run();
        auto t1 = std::chrono::high_resolution_clock::now();
        ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());

    std::printf("\n=== Benchmark ===\n");
    std::printf("Engine-AI       %7.3f ms  (%6.1f FPS)\n",
                ts[RUNS / 2], 1000.0 / ts[RUNS / 2]);

    if (auto* out = session.output(0)) {
        double s = 0.0;
        float mx = -1e30f, mn = 1e30f;
        const float* p = out->data<float>();
        const int64_t n = out->numel();
        for (int64_t i = 0; i < n; ++i) {
            s += p[i];
            if (p[i] > mx) mx = p[i];
            if (p[i] < mn) mn = p[i];
        }
        std::printf("Engine out: shape=[");
        for (int i = 0; i < out->rank(); ++i)
            std::printf("%lld%s", (long long)out->dim(i),
                        (i + 1 < out->rank()) ? "," : "");
        std::printf("]  sum=%.4f  min=%.4f  max=%.4f\n", s, mn, mx);

        FILE* fo = std::fopen("tools/converter/yolov8s_eng.f32", "wb");
        if (fo) {
            std::fwrite(p, sizeof(float), (size_t)n, fo);
            std::fclose(fo);
        }
    }
    return 0;
}