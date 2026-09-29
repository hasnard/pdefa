#include "engine/engine.hpp"
#include <cstdio>

using namespace engine;

int main() {
    printf("=== Engine-AI Smoke Test ===\n\n");

    printf("Version:    %s\n", version());
    printf("SIMD level: %s\n", simd_level());
    printf("CPU:        %s\n", runtime::Cpu::info().brand.c_str());
    printf("Cores:      %d physical, %d logical\n",
           runtime::Cpu::topology().physical_cores,
           runtime::Cpu::topology().logical_cores);

    auto A = Tensor::filled({4, 8}, DType::F32, 2.0f);
    auto B = Tensor::filled({8, 3}, DType::F32, 3.0f);
    printf("\nA: %s\n", A.shape().to_string().c_str());
    printf("B: %s\n", B.shape().to_string().c_str());

    auto C = ops::matmul(A, B);
    printf("C = A @ B: %s\n", C.shape().to_string().c_str());
    printf("C[0][0] = %.2f (beklenen: 48.00)\n", C.at<float>(0, 0));

    auto D = Tensor::filled({2, 4}, DType::F32, -1.0f);
    ops::relu_(D);
    printf("\nReLU(-1): D[0][0] = %.2f (beklenen: 0.00)\n", D.at<float>(0, 0));

    auto input  = Tensor::filled({1, 1, 5, 5}, DType::F32, 1.0f);
    auto weight = Tensor::filled({1, 1, 3, 3}, DType::F32, 1.0f);
    auto bias   = Tensor::filled({1}, DType::F32, 0.0f);
    auto conv_out = ops::conv2d(input, weight, &bias);
    printf("Conv2d out: %s\n", conv_out.shape().to_string().c_str());
    printf("Conv2d[0,0,0,0] = %.2f (beklenen: 9.00)\n", conv_out.at<float>(0, 0, 0, 0));

    printf("\n=== ALL OK ===\n");
    return 0;
}