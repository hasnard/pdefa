/* ============================================================================
 *  examples/test_silu.cpp
 *  SiLU (fused_activation == 5) izole testi.
 *  Identity weight'li 1×1 conv → çıkış = SiLU(giriş).
 *  Referans: x / (1 + exp(-x))
 * ========================================================================== */
#include <cstdio>
#include <cstdint>
#include <cmath>

#include "engine/ops/conv2d.hpp"
#include "engine/tensor.hpp"

using namespace engine;

int main() {
    constexpr int64_t Cin = 8, Cout = 8, HW = 8;

    /* Input: -4..+4 arası 64 değer */
    alignas(32) float input[Cin * HW];
    for (int64_t i = 0; i < Cin * HW; ++i) {
        input[i] = -4.0f + 8.0f * (float)i / (float)(Cin * HW - 1);
    }

    /* Weight: identity (ci == co için 1, geri kalan 0) */
    alignas(32) float weight[Cout * Cin];
    for (int64_t i = 0; i < Cout * Cin; ++i) weight[i] = 0.0f;
    for (int64_t i = 0; i < 8; ++i) weight[i * Cin + i] = 1.0f;

    alignas(32) float output[Cout * HW];

    /* Tensor sarmalayıcıları — NCHW [1, C, 1, HW] */
    int64_t in_dims[4] = {1, Cin, 1, HW};
    int64_t w_dims[4]  = {Cout, Cin, 1, 1};
    int64_t out_dims[4]= {1, Cout, 1, HW};

    Tensor t_in  = Tensor::from_external(input,  Shape(in_dims, 4),  DType::F32);
    Tensor t_w   = Tensor::from_external(weight, Shape(w_dims, 4),   DType::F32);
    Tensor t_out = Tensor::from_external(output, Shape(out_dims, 4), DType::F32);

    ops::Conv2dConfig cfg;
    cfg.stride_h = 1; cfg.stride_w = 1;
    cfg.pad_h    = 0; cfg.pad_w    = 0;
    cfg.dil_h    = 1; cfg.dil_w    = 1;
    cfg.groups   = 1;
    cfg.activation = ops::Conv2dConfig::FusedActivation::SiLU;   /* = 5 */
    cfg.leaky_slope = 0.0f;

    ops::conv2d(t_in, t_w, nullptr, t_out, cfg);

    /* Karşılaştır */
    float max_err = 0.0f;
    for (int64_t i = 0; i < Cout * HW; ++i) {
        const float x = input[i];
        const float expected = x / (1.0f + std::exp(-x));
        const float got = output[i];
        const float err = std::fabs(got - expected);
        if (err > max_err) max_err = err;
    }

    std::printf("=== SiLU izole test ===\n");
    std::printf("samples       = %lld\n", (long long)(Cout * HW));
    std::printf("max abs err   = %.3e\n", max_err);
    std::printf("input[ 0] = %+.4f -> got %+.6f (expect %+.6f)\n",
                input[0], output[0], input[0]/(1.0f+std::exp(-input[0])));
    std::printf("input[31] = %+.4f -> got %+.6f (expect %+.6f)\n",
                input[31], output[31], input[31]/(1.0f+std::exp(-input[31])));
    std::printf("input[63] = %+.4f -> got %+.6f (expect %+.6f)\n",
                input[63], output[63], input[63]/(1.0f+std::exp(-input[63])));

    if (max_err < 1e-5f) {
        std::printf("SONUC: PASS\n");
        return 0;
    }
    std::printf("SONUC: FAIL (err=%.3e)\n", max_err);
    return 1;
}