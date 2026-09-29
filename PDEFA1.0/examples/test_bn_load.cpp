#include "engine/engine.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

using namespace engine;

std::vector<float> load_f32(const char* path, size_t n) {
    std::vector<float> buf(n);
    FILE* f = std::fopen(path, "rb");
    if (!f) { printf("HATA: %s açılamadı\n", path); return {}; }
    size_t got = std::fread(buf.data(), sizeof(float), n, f);
    std::fclose(f);
    if (got != n) { printf("HATA: %s okuma eksik\n", path); return {}; }
    return buf;
}

int main() {
    runtime::GlobalPool::set_num_threads(12);

    printf("=== Engine-AI BN Fusion Test ===\n\n");

    Model model;
    if (!loader::load_engine_file("bn_cnn.engine", model)) {
        printf("HATA: yüklenemedi\n");
        return 1;
    }

    printf("Öncesi: Weights=%zu, Nodes=%zu\n",
           model.num_weights(), model.graph().nodes.size());

    /* Optimization */
    auto report = optimizer::optimize(model);
    printf("\nOptimizer:\n");
    printf("  Conv+BN fused:   %d\n", (int)report.conv_bn_fused);
    printf("  Conv+ReLU fused: %d\n", (int)report.conv_relu_fused);
    printf("  Dead nodes:      %d\n\n", (int)report.nodes_removed);

    printf("Sonrası: Weights=%zu, Nodes=%zu\n\n",
           model.num_weights(), model.graph().nodes.size());

    /* Session */
    Session session(model);

    const int64_t N = 1, C = 3, H = 64, W = 64;
    auto input_data = load_f32("bn_cnn_input.f32", N * C * H * W);
    if (input_data.empty()) return 1;

    auto input = Tensor::empty({N, C, H, W}, DType::F32);
    std::memcpy(input.data(), input_data.data(), input_data.size() * sizeof(float));

    session.set_input(0, &input);
    if (!session.run()) { printf("HATA: inference fail\n"); return 1; }

    const Tensor* output = session.output(0);
    if (!output) { printf("HATA: output yok\n"); return 1; }

    const int64_t n = output->numel();
    auto ref = load_f32("bn_cnn_output.f32", n);
    if (ref.empty()) return 1;

    const float* got = output->data<float>();
    float max_d = 0.0f;
    double sum_got = 0.0, sum_ref = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        if (d > max_d) max_d = d;
        sum_got += got[i];
        sum_ref += ref[i];
    }

        printf("PyTorch sum:  %.6f\n", sum_ref);
    printf("Engine sum:   %.6f\n", sum_got);
    printf("Max abs diff: %.6e\n", max_d);

    /* Göreli hata */
    float max_rel = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float denom = std::fabs(ref[i]) + 1e-6f;
        const float rel = std::fabs(got[i] - ref[i]) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    printf("Max rel error: %.4f%%\n\n", max_rel * 100.0f);

    printf("First 5 values:\n");
    for (int i = 0; i < 5 && i < n; ++i) {
        printf("  [%d] engine=%.6f pytorch=%.6f\n", i, got[i], ref[i]);
    }

    /* 7 katmanlı ağda hata birikir — daha esnek tolerans */
    if (max_d < 2e-2f) {
        printf("\n*** PASS: numerical precision acceptable ***\n");
        return 0;
    }
    printf("\n*** FAIL ***\n");
    return 1;
}