#include "engine/engine.hpp"
#include <cstdio>
#include <cmath>

using namespace engine;

/* Manuel beklenen değer:
 * input:  [1, 1, 5, 5] hepsi 1.0
 * weight: [1, 1, 3, 3] hepsi 1.0
 * bias:   [1] 0.0
 * stride: 1, pad: 0
 * output: [1, 1, 3, 3] her eleman = 9.0 */
bool test_conv2d_basic() {
    printf("[Test 1] Basic 5x5 -> 3x3, all-ones\n");
    auto input  = Tensor::filled({1, 1, 5, 5}, DType::F32, 1.0f);
    auto weight = Tensor::filled({1, 1, 3, 3}, DType::F32, 1.0f);
    auto bias   = Tensor::filled({1}, DType::F32, 0.0f);

    ops::Conv2dConfig cfg;
    auto out = ops::conv2d(input, weight, &bias, cfg);

    printf("  Output shape: %s (expected [1, 1, 3, 3])\n",
           out.shape().to_string().c_str());

    const float* p = out.data<float>();
    bool ok = true;
    for (int i = 0; i < 9; ++i) {
        if (std::fabs(p[i] - 9.0f) > 1e-4f) {
            printf("  MISMATCH at [%d]: got %.4f, expected 9.0\n", i, p[i]);
            ok = false;
        }
    }
    if (ok) printf("  PASS: all values = 9.0\n");
    return ok;
}

/* Farklı input/weight:
 * input:  [1, 1, 3, 3]
 * weight: [1, 1, 2, 2]
 * output: [1, 1, 2, 2]
 * Beklenen: her eleman = input'un 2x2 blokları toplamı */
bool test_conv2d_varied() {
    printf("\n[Test 2] 3x3 -> 2x2, custom values\n");

    auto input  = Tensor::empty({1, 1, 3, 3}, DType::F32);
    auto weight = Tensor::empty({1, 1, 2, 2}, DType::F32);
    auto bias   = Tensor::filled({1}, DType::F32, 0.0f);

    /* input:
     * [1, 2, 3]
     * [4, 5, 6]
     * [7, 8, 9]  */
    float* ip = input.data<float>();
    const float in_data[9] = {1,2,3, 4,5,6, 7,8,9};
    for (int i = 0; i < 9; ++i) ip[i] = in_data[i];

    /* weight: all-ones 2x2 */
    float* wp = weight.data<float>();
    for (int i = 0; i < 4; ++i) wp[i] = 1.0f;

    ops::Conv2dConfig cfg;
    auto out = ops::conv2d(input, weight, &bias, cfg);

    /* Beklenen output [1, 1, 2, 2]:
     * [1+2+4+5, 2+3+5+6]   = [12, 16]
     * [4+5+7+8, 5+6+8+9]   = [24, 28] */
    const float expected[4] = {12.0f, 16.0f, 24.0f, 28.0f};
    const float* op = out.data<float>();
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(op[i] - expected[i]) > 1e-4f) {
            printf("  MISMATCH at [%d]: got %.4f, expected %.4f\n",
                   i, op[i], expected[i]);
            ok = false;
        }
    }
    if (ok) printf("  PASS: [12, 16, 24, 28]\n");
    return ok;
}

/* Stride + padding */
bool test_conv2d_stride_pad() {
    printf("\n[Test 3] Stride=2, Pad=1\n");

    auto input  = Tensor::filled({1, 1, 4, 4}, DType::F32, 1.0f);
    auto weight = Tensor::filled({1, 1, 3, 3}, DType::F32, 1.0f);
    auto bias   = Tensor::filled({1}, DType::F32, 0.0f);

    ops::Conv2dConfig cfg;
    cfg.stride_h = 2; cfg.stride_w = 2;
    cfg.pad_h = 1; cfg.pad_w = 1;
    auto out = ops::conv2d(input, weight, &bias, cfg);

    /* 4x4 input, 3x3 kernel, stride=2, pad=1 → 2x2 output
     * Her pozisyon: 3x3 çekirdek, 4x4 input üzerinde, 1 padding
     * Hesaplama: kaydırılmış 3x3 pencereler, padding'ler 0 */
    printf("  Output shape: %s (expected [1, 1, 2, 2])\n",
           out.shape().to_string().c_str());

    /* Beklenen:
     * (0,0): h=0, w=0 → pencereler (-1,-1)...(1,1) → sadece (0,0)'dan (1,1) → 4 değer
     * (0,1): h=0, w=2 → pencereler (-1,1)...(1,3) → sadece (0,2),(0,3),(1,2),(1,3) → 4 değer
     * Benzer şekilde diğerleri. */
    const float* op = out.data<float>();
    /* Doğru beklenen: 2x2 output, 3x3 kernel, stride=2, pad=1
     * [0,0]: sadece (0,0)-(1,1) 4 hücre = 4
     * [0,1]: üstte padding, altta 3x3'ün bir kısmı = 6
     * [1,0]: solda padding = 6
     * [1,1]: tam 3x3 = 9 */
    const float expected[4] = {4.0f, 6.0f, 6.0f, 9.0f};
    printf("  Values: [%.1f, %.1f, %.1f, %.1f]\n", op[0], op[1], op[2], op[3]);
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(op[i] - expected[i]) > 1e-4f) {
            printf("  MISMATCH at [%d]: got %.1f, expected %.1f\n",
                   i, op[i], expected[i]);
            ok = false;
        }
    }
    if (ok) printf("  PASS: [4, 6, 6, 9] (correct convolution)\n");
    return ok;
}

/* Fused activation */
bool test_conv2d_fused_relu() {
    printf("\n[Test 4] Fused ReLU\n");

    auto input  = Tensor::filled({1, 1, 3, 3}, DType::F32, -1.0f);
    auto weight = Tensor::filled({1, 1, 2, 2}, DType::F32, 1.0f);
    auto bias   = Tensor::filled({1}, DType::F32, 0.0f);

    ops::Conv2dConfig cfg;
    cfg.activation = ops::Conv2dConfig::FusedActivation::ReLU;
    auto out = ops::conv2d(input, weight, &bias, cfg);

    /* input hepsi -1, weight hepsi 1 → conv sonucu -4
     * ReLU(-4) = 0 */
    const float* op = out.data<float>();
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(op[i] - 0.0f) > 1e-4f) {
            printf("  MISMATCH at [%d]: got %.4f, expected 0.0\n", i, op[i]);
            ok = false;
        }
    }
    if (ok) printf("  PASS: all values = 0.0 (ReLU killed negatives)\n");
    return ok;
}

int main() {
    printf("=== Engine-AI Conv2d Tests ===\n\n");

    bool ok = true;
    ok &= test_conv2d_basic();
    ok &= test_conv2d_varied();
    ok &= test_conv2d_stride_pad();
    ok &= test_conv2d_fused_relu();

    printf("\n");
    if (ok) printf("*** ALL TESTS PASSED ***\n");
    else    printf("*** SOME TESTS FAILED ***\n");
    return ok ? 0 : 1;
}