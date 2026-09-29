#include "engine/ops/norm.hpp"
#include "engine/runtime/thread_pool.hpp"
#include <cmath>
#include <cstring>
#include <vector>

namespace engine {
namespace ops {

void batchnorm2d_(Tensor& x,
                  const Tensor& weight,
                  const Tensor& bias,
                  const Tensor& mean,
                  const Tensor& var,
                  float eps) noexcept
{
    if (x.dtype() != DType::F32 || x.rank() != 4) return;

    /* NHWC: [N][H][W][C] — kanal dim(3) */
    const int64_t N  = x.dim(0);
    const int64_t H  = x.dim(1), W = x.dim(2);
    const int64_t C  = x.dim(3);
    const int64_t HW = H * W;

    if (weight.numel() != C) return;
    if (mean.numel()   != C) return;
    if (var.numel()    != C) return;

    const float* g = weight.data<float>();
    const float* b = bias.is_empty() ? nullptr : bias.data<float>();
    const float* m = mean.data<float>();
    const float* v = var.data<float>();
    float* xp = x.data<float>();

    std::vector<float> scale(static_cast<size_t>(C));
    std::vector<float> shift(static_cast<size_t>(C));
    for (int64_t c = 0; c < C; ++c) {
        const float s = g[c] / std::sqrt(v[c] + eps);
        scale[static_cast<size_t>(c)] = s;
        shift[static_cast<size_t>(c)] = (b ? b[c] : 0.0f) - m[c] * s;
    }

    auto& pool = runtime::GlobalPool::get();
    const int64_t total = N * HW;
    pool.parallel_for(0, total, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
            float* px = xp + t * C;
            for (int64_t c = 0; c < C; ++c) {
                px[c] = px[c] * scale[static_cast<size_t>(c)]
                              + shift[static_cast<size_t>(c)];
            }
        }
    });
}

void layernorm_(Tensor& x,
                const Tensor& weight,
                const Tensor& bias,
                float eps) noexcept
{
    if (x.dtype() != DType::F32) return;
    const int32_t r = x.rank();
    if (r < 1) return;
    const int64_t D = x.dim(-1);
    if (weight.numel() != D) return;
    if (!bias.is_empty() && bias.numel() != D) return;

    const int64_t rows = x.numel() / D;
    float* xp = x.data<float>();
    const float* g = weight.data<float>();
    const float* b = bias.is_empty() ? nullptr : bias.data<float>();

    auto& pool = runtime::GlobalPool::get();
    pool.parallel_for(0, rows, [&](int64_t r0, int64_t r1) {
        for (int64_t i = r0; i < r1; ++i) {
            float* row = xp + i * D;
            float sum = 0.0f, sq = 0.0f;
            for (int64_t j = 0; j < D; ++j) { sum += row[j]; sq += row[j]*row[j]; }
            const float mean = sum / static_cast<float>(D);
            const float var  = sq / static_cast<float>(D) - mean*mean;
            const float inv  = 1.0f / std::sqrt(var + eps);
            for (int64_t j = 0; j < D; ++j) {
                const float norm = (row[j] - mean) * inv;
                row[j] = norm * g[j] + (b ? b[j] : 0.0f);
            }
        }
    });
}

} /* namespace ops */
} /* namespace engine */