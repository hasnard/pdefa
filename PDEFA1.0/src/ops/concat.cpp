#include "engine/ops/concat.hpp"
#include "engine/runtime/thread_pool.hpp"
#include <immintrin.h>
#include <cstring>
#include <vector>
#include <algorithm>

namespace engine {
namespace ops {

namespace {

inline int32_t fix_axis(int32_t axis, int32_t rank) noexcept {
    return (axis < 0) ? axis + rank : axis;
}

struct LayoutInfo {
    int64_t outer = 0;
    int64_t inner = 0;
    int64_t out_axis_size = 0;
};

LayoutInfo compute_layout(const Shape& s, int32_t axis) noexcept {
    LayoutInfo li;
    li.outer = 1;
    for (int32_t i = 0; i < axis; ++i) li.outer *= s.dim(i);
    li.inner = 1;
    for (int32_t i = axis + 1; i < s.rank(); ++i) li.inner *= s.dim(i);
    li.out_axis_size = s.dim(axis);
    return li;
}

/* 4x Unrolled AVX2 Kopyalayıcı: Bellek bant genişliğini ve ILP'yi maksimize eder */
inline void copy_floats_avx2(float* __restrict dst, const float* __restrict src, int64_t count) noexcept {
    int64_t i = 0;
    
    // Unroll by 4: Her iterasyonda 128 byte (32 float) işlenir.
    // L1 Cache hattı ve Load/Store portları tamamen doyurulur.
    for (; i + 31 < count; i += 32) {
        __m256 y0 = _mm256_loadu_ps(src + i);
        __m256 y1 = _mm256_loadu_ps(src + i + 8);
        __m256 y2 = _mm256_loadu_ps(src + i + 16);
        __m256 y3 = _mm256_loadu_ps(src + i + 24);

        _mm256_storeu_ps(dst + i, y0);
        _mm256_storeu_ps(dst + i + 8, y1);
        _mm256_storeu_ps(dst + i + 16, y2);
        _mm256_storeu_ps(dst + i + 24, y3);
    }
    
    // Kalan bloklar için 1x AVX2 (32 byte)
    for (; i + 7 < count; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
    }
    
    // Skaler kuyruk
    for (; i < count; ++i) {
        dst[i] = src[i];
    }
}

constexpr int32_t kMaxStackInputs = 64;
constexpr int64_t kBlockSize = 32;   /* Outer iterasyon başına iş parçası */

} /* anonymous namespace */

void concat(const Tensor* const* inputs,
            int32_t n_inputs,
            Tensor& output,
            int32_t axis) noexcept
{
    if (n_inputs <= 0 || !inputs) return;
    if (output.dtype() != DType::F32) return;

    const int32_t rank = output.rank();
    axis = fix_axis(axis, rank);
    if (axis < 0 || axis >= rank) return;

    const LayoutInfo li = compute_layout(output.shape(), axis);
    if (li.outer == 0 || li.inner == 0) return;

    float* const __restrict out = output.data<float>();

    int64_t stack_offsets[kMaxStackInputs];
    int64_t stack_dst_offsets[kMaxStackInputs]; // Yeni: Pre-calculated bellek ofsetleri
    int64_t stack_chunks[kMaxStackInputs];
    const float* stack_src[kMaxStackInputs];

    std::vector<int64_t> heap_offsets, heap_dst_offsets, heap_chunks;
    std::vector<const float*> heap_src;

    int64_t* axis_offsets = stack_offsets;
    int64_t* dst_offsets  = stack_dst_offsets;
    int64_t* chunk_sizes  = stack_chunks;
    const float** src_bases = stack_src;

    if (n_inputs > kMaxStackInputs) {
        heap_offsets.resize(n_inputs);     axis_offsets = heap_offsets.data();
        heap_dst_offsets.resize(n_inputs); dst_offsets  = heap_dst_offsets.data();
        heap_chunks.resize(n_inputs);      chunk_sizes  = heap_chunks.data();
        heap_src.resize(n_inputs);         src_bases    = heap_src.data();
    }

    int64_t acc = 0;
    for (int32_t i = 0; i < n_inputs; ++i) {
        if (!inputs[i]) return;
        axis_offsets[i] = acc;
        dst_offsets[i]  = acc * li.inner; // Döngü dışı sabit pointer aritmetiği hesabı
        chunk_sizes[i]  = inputs[i]->dim(axis) * li.inner;
        src_bases[i]    = inputs[i]->data<float>();
        acc += inputs[i]->dim(axis);
    }

    auto& pool = runtime::GlobalPool::get();
    const int64_t dst_stride = li.out_axis_size * li.inner;
    const int64_t total_out  = li.outer * dst_stride;
    constexpr int64_t kParallelThreshold = 32768;

    /* DURUM 1: NHWC Kanal Birleştirme (inner == 1, axis == rank - 1) */
    if (li.inner == 1) {
        if (n_inputs == 2) {
            const int64_t c0 = chunk_sizes[0];
            const int64_t c1 = chunk_sizes[1];
            const float* const __restrict s0 = src_bases[0];
            const float* const __restrict s1 = src_bases[1];

            auto worker = [=](int64_t o0, int64_t o1) {
                for (int64_t o = o0; o < o1; ++o) {
                    float* const __restrict d = out + o * dst_stride;
                    copy_floats_avx2(d,      s0 + o * c0, c0);
                    copy_floats_avx2(d + c0, s1 + o * c1, c1);
                }
            };

            if (total_out >= kParallelThreshold && li.outer > kBlockSize) {
                const int64_t n_blocks = (li.outer + kBlockSize - 1) / kBlockSize;
                pool.parallel_for(0, n_blocks, [=](int64_t b0, int64_t b1) {
                    for (int64_t b = b0; b < b1; ++b) {
                        const int64_t o0 = b * kBlockSize;
                        const int64_t o1 = std::min<int64_t>(o0 + kBlockSize, li.outer);
                        worker(o0, o1);
                    }
                });
            } else {
                worker(0, li.outer);
            }
            return;
        }

        auto multi_worker = [=](int64_t o0, int64_t o1) {
            for (int64_t o = o0; o < o1; ++o) {
                float* const __restrict dst_row = out + o * dst_stride;
                for (int32_t t = 0; t < n_inputs; ++t) {
                    copy_floats_avx2(dst_row + axis_offsets[t],
                                     src_bases[t] + o * chunk_sizes[t],
                                     chunk_sizes[t]);
                }
            }
        };

        if (total_out >= kParallelThreshold && li.outer > kBlockSize) {
            const int64_t n_blocks = (li.outer + kBlockSize - 1) / kBlockSize;
            pool.parallel_for(0, n_blocks, [=](int64_t b0, int64_t b1) {
                for (int64_t b = b0; b < b1; ++b) {
                    const int64_t o0 = b * kBlockSize;
                    const int64_t o1 = std::min<int64_t>(o0 + kBlockSize, li.outer);
                    multi_worker(o0, o1);
                }
            });
        } else {
            multi_worker(0, li.outer);
        }
        return;
    }

    /* DURUM 2: Standart Concat (Spatial/Batch Birleştirme - inner > 1) */
    // std::memcpy yerine AVX2 copy_floats_avx2 entegre edildi.
    if (total_out < kParallelThreshold) {
        for (int64_t o = 0; o < li.outer; ++o) {
            float* const dst_row = out + o * dst_stride;
            for (int32_t t = 0; t < n_inputs; ++t) {
                copy_floats_avx2(dst_row + dst_offsets[t], 
                                 src_bases[t] + o * chunk_sizes[t], 
                                 chunk_sizes[t]);
            }
        }
    }
    else if (li.outer > kBlockSize) {
        const int64_t n_blocks = (li.outer + kBlockSize - 1) / kBlockSize;
        pool.parallel_for(0, n_blocks, [&](int64_t b0, int64_t b1) {
            for (int64_t b = b0; b < b1; ++b) {
                const int64_t o0 = b * kBlockSize;
                const int64_t o1 = std::min<int64_t>(o0 + kBlockSize, li.outer);
                for (int64_t o = o0; o < o1; ++o) {
                    float* const dst_row = out + o * dst_stride;
                    for (int32_t t = 0; t < n_inputs; ++t) {
                        copy_floats_avx2(dst_row + dst_offsets[t], 
                                         src_bases[t] + o * chunk_sizes[t], 
                                         chunk_sizes[t]);
                    }
                }
            }
        });
    }
    else {
        for (int32_t t = 0; t < n_inputs; ++t) {
            const int64_t chunk = chunk_sizes[t];
            float* const dst = out + dst_offsets[t];
            const float* const src = src_bases[t];

            if (chunk >= kParallelThreshold) {
                pool.parallel_for(0, chunk, [&](int64_t i0, int64_t i1) {
                    copy_floats_avx2(dst + i0, src + i0, i1 - i0);
                });
            } else {
                copy_floats_avx2(dst, src, chunk);
            }
        }
    }
}

Tensor concat(const Tensor* const* inputs,
              int32_t n_inputs,
              int32_t axis) noexcept
{
    if (n_inputs <= 0 || !inputs || !inputs[0]) return Tensor{};

    const Shape& first = inputs[0]->shape();
    const int32_t rank = first.rank();
    const int32_t ax = fix_axis(axis, rank);
    if (ax < 0 || ax >= rank) return Tensor{};

    int64_t dims[kMaxNdim];
    for (int32_t i = 0; i < rank; ++i) dims[i] = first.dim(i);
    dims[ax] = 0;

    for (int32_t i = 0; i < n_inputs; ++i) {
        const Tensor* t = inputs[i];
        if (!t || t->rank() != rank || t->dtype() != DType::F32) return Tensor{};
        dims[ax] += t->dim(ax);
    }

    Tensor out = Tensor::empty(Shape(dims, rank), DType::F32);
    if (out.is_empty()) return out;

    concat(inputs, n_inputs, out, axis);
    return out;
}

Tensor concat(const Tensor& a, const Tensor& b, int32_t axis) noexcept {
    const Tensor* ptrs[2] = { &a, &b };
    return concat(ptrs, 2, axis);
}

} /* namespace ops */
} /* namespace engine */