#include "engine/ops/softmax.hpp"
#include "engine/runtime/thread_pool.hpp"
#include <cmath>
#include <cstring>

namespace engine {
namespace ops {

void softmax_(Tensor& x) noexcept {
    if (x.dtype() != DType::F32) return;
    if (x.rank() < 1) return;

    /* NHWC uyumlu: son eksen kanal. NCHW'de eksen 1'di, NHWC'de 3. */
    const int64_t D = x.dim(-1);
    const int64_t rows = x.numel() / D;
    float* xp = x.data<float>();

    auto& pool = runtime::GlobalPool::get();
    pool.parallel_for(0, rows, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            float* row = xp + r * D;
            float m = row[0];
            for (int64_t j = 1; j < D; ++j) if (row[j] > m) m = row[j];
            float sum = 0.0f;
            for (int64_t j = 0; j < D; ++j) {
                const float e = std::exp(row[j] - m);
                row[j] = e;
                sum += e;
            }
            const float inv = 1.0f / sum;
            for (int64_t j = 0; j < D; ++j) row[j] *= inv;
        }
    }, 64);
}

} /* namespace ops */
} /* namespace engine */