#ifndef ENGINE_SHAPE_HPP
#define ENGINE_SHAPE_HPP

#include "engine/dtype.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <ostream>
#include <string>

namespace engine {

inline constexpr int32_t kMaxNdim = 8;

class Shape {
public:
    constexpr Shape() noexcept = default;

    constexpr Shape(std::initializer_list<int64_t> dims) noexcept {
        for (int64_t d : dims) {
            if (ndim_ >= kMaxNdim) break;
            dims_[ndim_++] = d;
        }
        compute_strides();
    }

    template <size_t N>
    constexpr Shape(const std::array<int64_t, N>& a) noexcept {
        static_assert(N <= static_cast<size_t>(kMaxNdim), "too many dims");
        for (size_t i = 0; i < N; ++i) dims_[i] = a[i];
        ndim_ = static_cast<int32_t>(N);
        compute_strides();
    }

    constexpr Shape(const int64_t* data, int32_t ndim) noexcept
        : ndim_(ndim < 0 ? 0 : (ndim > kMaxNdim ? kMaxNdim : ndim)) {
        for (int32_t i = 0; i < ndim_; ++i) dims_[i] = data[i];
        compute_strides();
    }

    explicit constexpr Shape(int64_t d0) noexcept : ndim_(1) {
        dims_[0] = d0;
        compute_strides();
    }

    [[nodiscard]] constexpr int32_t rank() const noexcept { return ndim_; }
    [[nodiscard]] constexpr int64_t numel() const noexcept { return numel_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return numel_ == 0; }
    [[nodiscard]] constexpr bool is_scalar() const noexcept { return ndim_ == 0; }
    [[nodiscard]] constexpr bool is_contiguous() const noexcept { return true; }

    [[nodiscard]] constexpr int64_t dim(int32_t axis) const noexcept {
        if (axis < 0) axis += ndim_;
        return (axis < 0 || axis >= ndim_) ? 0 : dims_[axis];
    }

    [[nodiscard]] constexpr int64_t stride(int32_t axis) const noexcept {
        if (axis < 0) axis += ndim_;
        return (axis < 0 || axis >= ndim_) ? 0 : strides_[axis];
    }

    [[nodiscard]] constexpr int64_t front() const noexcept {
        return ndim_ > 0 ? dims_[0] : 1;
    }
    [[nodiscard]] constexpr int64_t back() const noexcept {
        return ndim_ > 0 ? dims_[ndim_ - 1] : 1;
    }

    [[nodiscard]] constexpr const int64_t* data() const noexcept { return dims_.data(); }
    [[nodiscard]] constexpr const int64_t* strides_data() const noexcept { return strides_.data(); }

    [[nodiscard]] constexpr int64_t operator[](int32_t i) const noexcept { return dims_[i]; }

    [[nodiscard]] constexpr bool operator==(const Shape& o) const noexcept {
        if (ndim_ != o.ndim_) return false;
        for (int32_t i = 0; i < ndim_; ++i) {
            if (dims_[i] != o.dims_[i]) return false;
        }
        return true;
    }

    [[nodiscard]] constexpr bool operator!=(const Shape& o) const noexcept {
        return !(*this == o);
    }

    [[nodiscard]] constexpr bool is_broadcastable_with(const Shape& o) const noexcept {
        const int32_t mr = ndim_ > o.ndim_ ? ndim_ : o.ndim_;
        for (int32_t i = 0; i < mr; ++i) {
            const int64_t a = axis_from_right(i);
            const int64_t b = o.axis_from_right(i);
            if (a != b && a != 1 && b != 1) return false;
        }
        return true;
    }

    [[nodiscard]] constexpr Shape broadcast_to(const Shape& o) const noexcept {
        const int32_t mr = ndim_ > o.ndim_ ? ndim_ : o.ndim_;
        Shape result;
        result.ndim_ = mr;
        for (int32_t i = 0; i < mr; ++i) {
            const int64_t a = axis_from_right(i);
            const int64_t b = o.axis_from_right(i);
            result.dims_[mr - 1 - i] = (a >= b) ? a : b;
        }
        result.compute_strides();
        return result;
    }

    [[nodiscard]] constexpr Shape squeeze(int32_t axis) const noexcept {
        if (axis < 0) axis += ndim_;
        if (axis < 0 || axis >= ndim_ || dims_[axis] != 1) return *this;
        Shape r;
        r.ndim_ = ndim_ - 1;
        int32_t j = 0;
        for (int32_t i = 0; i < ndim_; ++i) {
            if (i == axis) continue;
            r.dims_[j++] = dims_[i];
        }
        r.compute_strides();
        return r;
    }

    [[nodiscard]] constexpr Shape unsqueeze(int32_t axis) const noexcept {
        if (axis < 0) axis += ndim_ + 1;
        if (axis < 0 || axis > ndim_ || ndim_ >= kMaxNdim) return *this;
        Shape r;
        r.ndim_ = ndim_ + 1;
        int32_t j = 0;
        for (int32_t i = 0; i < r.ndim_; ++i) {
            if (i == axis) { r.dims_[i] = 1; continue; }
            r.dims_[i] = dims_[j++];
        }
        r.compute_strides();
        return r;
    }

    [[nodiscard]] constexpr const int64_t* begin() const noexcept { return dims_.data(); }
    [[nodiscard]] constexpr const int64_t* end() const noexcept { return dims_.data() + ndim_; }

    [[nodiscard]] std::string to_string() const {
        std::string out;
        out.reserve(2 + static_cast<size_t>(ndim_) * 6);
        out += '[';
        for (int32_t i = 0; i < ndim_; ++i) {
            if (i) out += ", ";
            out += std::to_string(dims_[i]);
        }
        out += ']';
        return out;
    }

private:
    std::array<int64_t, kMaxNdim> dims_{};
    std::array<int64_t, kMaxNdim> strides_{};
    int64_t numel_ = 1;
    int32_t ndim_ = 0;

    constexpr void compute_strides() noexcept {
        numel_ = 1;
        for (int32_t i = 0; i < ndim_; ++i) {
            if (dims_[i] <= 0) {
                numel_ = 0;
                for (int32_t k = 0; k < kMaxNdim; ++k) strides_[k] = 0;
                return;
            }
            numel_ *= dims_[i];
        }
        int64_t s = 1;
        for (int32_t i = ndim_ - 1; i >= 0; --i) {
            strides_[i] = s;
            s *= dims_[i];
        }
        for (int32_t i = ndim_; i < kMaxNdim; ++i) strides_[i] = 0;
    }

    [[nodiscard]] constexpr int64_t axis_from_right(int32_t i) const noexcept {
        const int32_t idx = ndim_ - 1 - i;
        return (idx >= 0) ? dims_[idx] : 1;
    }
};

/* Byte hesabı — F32 varsayımıyla */
[[nodiscard]] inline size_t shape_nbytes(const Shape& s, DType dt) noexcept {
    const size_t bits = static_cast<size_t>(s.numel()) * dtype_bits(dt);
    return (bits + 7) / 8;
}

inline std::ostream& operator<<(std::ostream& os, const Shape& s) {
    os << s.to_string();
    return os;
}

} /* namespace engine */

#endif