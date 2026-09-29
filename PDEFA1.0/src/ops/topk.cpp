#include "engine/ops/topk.hpp"
#include <algorithm>
#include <vector>
#include <cstring>

namespace engine {
namespace ops {

namespace {

struct KV { float v; int32_t i; };

} /* anonymous namespace */

void topk(const Tensor& input, int64_t k, int32_t axis,
          bool largest, bool sorted,
          Tensor& values, Tensor& indices) noexcept
{
    if (input.is_empty()) return;

    const int32_t rank = input.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return;

    const int64_t axis_dim = input.dim(axis);
    if (k <= 0 || k > axis_dim) k = axis_dim;

    int64_t outer = 1;
    for (int32_t d = 0; d < axis; ++d) outer *= input.dim(d);

    int64_t inner = 1;
    for (int32_t d = axis + 1; d < rank; ++d) inner *= input.dim(d);

    const float* in = input.data<float>();
    float*  vout = values.data<float>();
    int32_t* iout = indices.data<int32_t>();

    std::vector<KV> buf(axis_dim);

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in_i = 0; in_i < inner; ++in_i) {
            const float* base = in + (o * axis_dim) * inner + in_i;
            for (int64_t a = 0; a < axis_dim; ++a) {
                buf[a].v = base[a * inner];
                buf[a].i = static_cast<int32_t>(a);
            }

            if (largest) {
                std::partial_sort(buf.begin(), buf.begin() + k, buf.end(),
                                  [](const KV& x, const KV& y) { return x.v > y.v; });
            } else {
                std::partial_sort(buf.begin(), buf.begin() + k, buf.end(),
                                  [](const KV& x, const KV& y) { return x.v < y.v; });
            }

            float*  vbase = vout + (o * k) * inner + in_i;
            int32_t* ibase = iout + (o * k) * inner + in_i;
            for (int64_t j = 0; j < k; ++j) {
                vbase[j * inner] = buf[j].v;
                ibase[j * inner] = buf[j].i;
            }
        }
    }
}

}
}