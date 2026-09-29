#include "engine/ops/gather_elements.hpp"
#include <cstring>

namespace engine {
namespace ops {

void gather_elements(const Tensor& data, const Tensor& indices,
                     int32_t axis, Tensor& output) noexcept
{
    if (data.is_empty() || indices.is_empty() || output.is_empty()) return;
    if (data.rank() != indices.rank()) return;

    const int32_t rank = data.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return;

    const int64_t axis_dim = data.dim(axis);

    int64_t outer = 1;
    for (int32_t d = 0; d < axis; ++d) outer *= data.dim(d);

    int64_t inner = 1;
    for (int32_t d = axis + 1; d < rank; ++d) inner *= data.dim(d);

    const int64_t idx_axis_dim = indices.dim(axis);

    const float*  src = data.data<float>();
    const int32_t* idx = indices.data<int32_t>();
    float*  dst = output.data<float>();

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t a = 0; a < idx_axis_dim; ++a) {
            for (int64_t i = 0; i < inner; ++i) {
                int32_t gi = idx[(o * idx_axis_dim + a) * inner + i];
                if (gi < 0) gi += static_cast<int32_t>(axis_dim);
                if (gi < 0 || gi >= axis_dim) gi = 0;

                const float  v = src[(o * axis_dim + gi) * inner + i];
                dst[(o * idx_axis_dim + a) * inner + i] = v;
            }
        }
    }
}

}
}