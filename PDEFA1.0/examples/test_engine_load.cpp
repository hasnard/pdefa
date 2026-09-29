#include "engine/engine.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

using namespace engine;

/* Basit .f32 dosya okuma */
std::vector<float> load_f32(const char* path, size_t expected_count) {
    std::vector<float> buf(expected_count);
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        printf("HATA: %s açılamadı\n", path);
        return {};
    }
    size_t got = std::fread(buf.data(), sizeof(float), expected_count, f);
    std::fclose(f);
    if (got != expected_count) {
        printf("HATA: %s beklenen %zu, okunan %zu\n",
               path, expected_count, got);
        return {};
    }
    return buf;
}

int main() {
    printf("=== Engine-AI Model Load Test ===\n\n");

    /* 1) Model yükle */
    Model model;
    if (!loader::load_engine_file("simple_cnn.engine", model)) {
        printf("HATA: simple_cnn.engine yüklenemedi\n");
        return 1;
    }

    printf("Model:    %s\n", model.name().c_str());
    printf("Weights:  %zu\n", model.num_weights());
    printf("Inputs:   %d\n", model.num_inputs());
    printf("Outputs:  %d\n", model.num_outputs());
    printf("Nodes:    %zu\n\n", model.graph().nodes.size());

    /* 2) Graph optimizer çalıştır */
    printf("Optimizing graph...\n");
    auto report = optimizer::optimize(model);
    printf("  Conv+BN fused: %d\n", static_cast<int>(report.conv_bn_fused));
    printf("  Dead nodes removed: %d\n\n", static_cast<int>(report.nodes_removed));

    /* 3) Session oluştur */
    Session session(model);

    /* 4) Input'u yükle */
    const int64_t N = 1, C = 3, H = 64, W = 64;
    auto input_data = load_f32("simple_cnn_input.f32", N * C * H * W);
    if (input_data.empty()) return 1;

    auto input_tensor = Tensor::empty({N, C, H, W}, DType::F32);
    std::memcpy(input_tensor.data(), input_data.data(),
                input_data.size() * sizeof(float));

    session.set_input(0, &input_tensor);

    /* 5) Çalıştır */
    printf("Running inference...\n");
    if (!session.run()) {
        printf("HATA: session.run() başarısız\n");
        return 1;
    }

    const Tensor* output = session.output(0);
    if (!output) {
        printf("HATA: output yok\n");
        return 1;
    }

    printf("  Output shape: %s\n", output->shape().to_string().c_str());
    printf("  Output numel: %lld\n", (long long)output->numel());
    printf("  Run time:     %.3f ms\n\n", session.stats().last_run_ms);

    /* 6) PyTorch referansı ile karşılaştır */
    const int64_t out_count = output->numel();
    auto ref_data = load_f32("simple_cnn_output.f32", out_count);
    if (ref_data.empty()) return 1;

    const float* got = output->data<float>();

    printf("--- Comparison ---\n");
    float max_diff = 0.0f;
    double sum_got = 0.0;
    double sum_ref = 0.0;

    for (int64_t i = 0; i < out_count; ++i) {
        const float diff = std::fabs(got[i] - ref_data[i]);
        if (diff > max_diff) max_diff = diff;
        sum_got += got[i];
        sum_ref += ref_data[i];
    }

    printf("  PyTorch sum:  %.6f\n", sum_ref);
    printf("  Engine sum:   %.6f\n", sum_got);
    printf("  Max diff:     %.6e\n", max_diff);
    printf("  Tolerance:    1e-4\n");

    if (max_diff < 1e-3f) {
        printf("\n*** PASS: Engine matches PyTorch ***\n");
        return 0;
    } else if (max_diff < 1e-2f) {
        printf("\n*** MOSTLY PASS: small numerical differences ***\n");
        return 0;
    } else {
        printf("\n*** FAIL: outputs differ significantly ***\n");

        /* İlk 10 değeri göster */
        printf("\n  First 10 values:\n");
        for (int i = 0; i < 10 && i < out_count; ++i) {
            printf("    [%d]  engine=%.6f  pytorch=%.6f  diff=%.6f\n",
                   i, got[i], ref_data[i], std::fabs(got[i] - ref_data[i]));
        }

            /* Ara tensor karşılaştırması */
    printf("\n--- Intermediate layer comparison ---\n");
    
        // Layer 1: Conv1 çıktısı = intermediate[1]
    {
        const Tensor* t = session.debug_intermediate(1);
        if (t) {
            const int64_t n = t->numel();
            const float* p = t->data<float>();

            double s = 0.0;
            for (int64_t i = 0; i < n; ++i) s += p[i];
            printf("  [1] Conv1 çıktısı: shape=%s sum=%.4f\n",
                   t->shape().to_string().c_str(), s);

            /* Element-wise comparison */
            auto ref = load_f32("layer_conv1.f32", static_cast<size_t>(n));
            if (!ref.empty()) {
                float max_d = 0.0f;
                int64_t worst_i = 0;
                int64_t mismatches = 0;
                for (int64_t i = 0; i < n; ++i) {
                    const float d = std::fabs(p[i] - ref[i]);
                    if (d > max_d) { max_d = d; worst_i = i; }
                    if (d > 1e-3f) mismatches++;
                }
                printf("       max_diff=%.6e at idx %lld   mismatches=%lld/%lld (%.1f%%)\n",
                       max_d, (long long)worst_i,
                       (long long)mismatches, (long long)n,
                       100.0 * mismatches / n);
                printf("       C++  [0..7] = ");
                for (int i = 0; i < 8; ++i) printf("%9.5f ", p[i]);
                printf("\n       Ref  [0..7] = ");
                for (int i = 0; i < 8; ++i) printf("%9.5f ", ref[i]);
                printf("\n       Diff [0..7] = ");
                for (int i = 0; i < 8; ++i) printf("%9.5f ", p[i] - ref[i]);
                printf("\n");
            } else {
                printf("       (layer_conv1.f32 yüklenemedi)\n");
            }
        } else printf("  [1] NULL\n");
    }
    
    // Layer 3: Pool1 çıktısı = intermediate[3]
    {
        const Tensor* t = session.debug_intermediate(3);
        if (t) {
            printf("  [3] Pool1 çıktısı: shape=%s sum=", t->shape().to_string().c_str());
            double s = 0.0;
            const float* p = t->data<float>();
            for (int64_t i = 0; i < t->numel(); ++i) s += p[i];
            printf("%.4f\n", s);
        } else printf("  [3] NULL\n");
    }
    
    // Layer 6: Pool2 çıktısı = intermediate[6]
    {
        const Tensor* t = session.debug_intermediate(6);
        if (t) {
            printf("  [6] Pool2 çıktısı: shape=%s sum=", t->shape().to_string().c_str());
            double s = 0.0;
            const float* p = t->data<float>();
            for (int64_t i = 0; i < t->numel(); ++i) s += p[i];
            printf("%.4f\n", s);
        } else printf("  [6] NULL\n");
    }
    
        return 1;
    }
}