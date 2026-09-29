/* ============================================================================
 *  src/core/tensor.cpp
 *  Engine-AI — Tensor implementasyonu
 *
 *  Kritik noktalar:
 *    - Sıcak yol (data(), at(), numel()) → header'da inline, burada DEĞİL
 *    - Bu dosyada: allocation, view, clone, I/O işlemleri
 *    - Arena entegrasyonu → sıfır malloc
 * ========================================================================== */

#include "engine/tensor.hpp"
#include "engine/memory/aligned_allocator.hpp"

#include <cstring>
#include <sstream>

namespace engine {

/* ============================================================================
 *  MOVE SEMANTİĞİ
 * ========================================================================== */

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      shape_(other.shape_),
      dtype_(other.dtype_),
      nbytes_(other.nbytes_),
      owns_data_(other.owns_data_),
      from_arena_(other.from_arena_)
{
    /* Kaynağı boşalt — double-free önle */
    other.data_       = nullptr;
    other.dtype_      = DType::Unknown;
    other.nbytes_     = 0;
    other.owns_data_  = false;
    other.from_arena_ = false;
    /* shape_ move edilir ama POD olduğu için trivial */
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free_if_owning();

        data_       = other.data_;
        shape_      = other.shape_;
        dtype_      = other.dtype_;
        nbytes_     = other.nbytes_;
        owns_data_  = other.owns_data_;
        from_arena_ = other.from_arena_;

        other.data_       = nullptr;
        other.dtype_      = DType::Unknown;
        other.nbytes_     = 0;
        other.owns_data_  = false;
        other.from_arena_ = false;
    }
    return *this;
}

Tensor::~Tensor() noexcept {
    free_if_owning();
}

/* ============================================================================
 *  FABRİKA FONKSİYONLARI
 * ========================================================================== */

Tensor Tensor::empty(const Shape& shape, DType dtype) noexcept {
    Tensor t;
    t.assign_metadata(shape, dtype);
    if (t.nbytes_ == 0) return t;

    void* mem = memory::AlignedAllocator::allocate(t.nbytes_, memory::kDefaultAlign);
    if (!mem) return t;   // boş tensor döner

    t.data_       = mem;
    t.owns_data_  = true;
    t.from_arena_ = false;
    return t;
}

Tensor Tensor::zeros(const Shape& shape, DType dtype) noexcept {
    Tensor t = empty(shape, dtype);
    if (t.data_) {
        std::memset(t.data_, 0, t.nbytes_);
    }
    return t;
}

Tensor Tensor::filled(const Shape& shape, DType dtype, float value) noexcept {
    Tensor t = empty(shape, dtype);
    if (t.data_) t.fill_(value);
    return t;
}

Tensor Tensor::from_external(void* data, const Shape& shape, DType dtype) noexcept {
    Tensor t;
    t.assign_metadata(shape, dtype);
    t.data_       = data;
    t.owns_data_  = false;
    t.from_arena_ = false;
    return t;
}

Tensor Tensor::create_in_arena(memory::Arena& arena, const Shape& shape, DType dtype) noexcept {
    Tensor t;
    t.assign_metadata(shape, dtype);
    if (t.nbytes_ == 0) return t;

    void* mem = arena.allocate(t.nbytes_, memory::kDefaultAlign);
    if (!mem) return t;   // arena doldu → boş tensor

    t.data_       = mem;
    t.owns_data_  = false;   // arena sahibi
    t.from_arena_ = true;
    return t;
}

Tensor Tensor::zeros_in_arena(memory::Arena& arena, const Shape& shape, DType dtype) noexcept {
    Tensor t = create_in_arena(arena, shape, dtype);
    if (t.data_) {
        std::memset(t.data_, 0, t.nbytes_);
    }
    return t;
}

/* ============================================================================
 *  VERİ ERİŞİMİ (runtime tip kontrolü)
 * ========================================================================== */

void* Tensor::data_as(DType expected) noexcept {
    return (dtype_ == expected) ? data_ : nullptr;
}

const void* Tensor::data_as(DType expected) const noexcept {
    return (dtype_ == expected) ? data_ : nullptr;
}

/* ============================================================================
 *  VIEW / RESHAPE (ZERO-COPY)
 * ========================================================================== */

std::optional<Tensor> Tensor::view(const Shape& new_shape) const noexcept {
    /* Sadece contiguous tensor view'lanabilir.
     * Non-contiguous olsaydı, stride'lar uyuşmazdı → veri yanlış okunurdu. */
    if (!shape_.is_contiguous()) return std::nullopt;

    /* Toplam eleman sayısı korunmalı. */
    if (shape_.numel() != new_shape.numel()) return std::nullopt;

    Tensor v;
    v.data_       = data_;       // aynı bellek, kopya YOK
    v.shape_      = new_shape;
    v.dtype_      = dtype_;
    v.nbytes_     = nbytes_;
    v.owns_data_  = false;       // sahibi değil
    v.from_arena_ = from_arena_; // arena'dan gelmişse işaretle
    return v;
}

std::optional<Tensor> Tensor::squeeze(int32_t axis) const noexcept {
    const Shape squeezed = shape_.squeeze(axis);
    if (squeezed == shape_) return std::nullopt;   // değişmedi

    Tensor v;
    v.data_       = data_;
    v.shape_      = squeezed;
    v.dtype_      = dtype_;
    v.nbytes_     = nbytes_;
    v.owns_data_  = false;
    v.from_arena_ = from_arena_;
    return v;
}

std::optional<Tensor> Tensor::unsqueeze(int32_t axis) const noexcept {
    const Shape expanded = shape_.unsqueeze(axis);
    if (expanded == shape_) return std::nullopt;

    Tensor v;
    v.data_       = data_;
    v.shape_      = expanded;
    v.dtype_      = dtype_;
    v.nbytes_     = nbytes_;
    v.owns_data_  = false;
    v.from_arena_ = from_arena_;
    return v;
}

/* ============================================================================
 *  CLONE (DERİN KOPYA)
 * ========================================================================== */

Tensor Tensor::clone() const noexcept {
    if (!data_ || nbytes_ == 0) return Tensor{};

    Tensor t = empty(shape_, dtype_);
    if (t.data_ && data_) {
        std::memcpy(t.data_, data_, nbytes_);
    }
    return t;
}

Tensor Tensor::clone_in_arena(memory::Arena& arena) const noexcept {
    if (!data_ || nbytes_ == 0) return Tensor{};

    Tensor t = create_in_arena(arena, shape_, dtype_);
    if (t.data_ && data_) {
        std::memcpy(t.data_, data_, nbytes_);
    }
    return t;
}

/* ============================================================================
 *  VERİ İŞLEMLERİ
 * ========================================================================== */

void Tensor::zero_() noexcept {
    if (data_ && nbytes_ > 0) {
        std::memset(data_, 0, nbytes_);
    }
}

void Tensor::fill_(float value) noexcept {
    if (!data_ || shape_.numel() == 0) return;

    const int64_t n = shape_.numel();

    switch (dtype_) {
        case DType::F32: {
            auto* p = static_cast<float*>(data_);
            for (int64_t i = 0; i < n; ++i) p[i] = value;
            break;
        }
        case DType::I32: {
            auto* p = static_cast<int32_t*>(data_);
            const int32_t v = static_cast<int32_t>(value);
            for (int64_t i = 0; i < n; ++i) p[i] = v;
            break;
        }
        case DType::I64: {
            auto* p = static_cast<int64_t*>(data_);
            const int64_t v = static_cast<int64_t>(value);
            for (int64_t i = 0; i < n; ++i) p[i] = v;
            break;
        }
        default:
            /* Diğer tipler için sessizce atla (ileride eklenecek). */
            break;
    }
}

bool Tensor::copy_from(const Tensor& src) noexcept {
    if (!data_ || !src.data_) return false;
    if (nbytes_ != src.nbytes_) return false;
    if (dtype_ != src.dtype_) return false;
    if (shape_ != src.shape_) return false;

    std::memcpy(data_, src.data_, nbytes_);
    return true;
}

bool Tensor::copy_from_raw(const void* src, size_t bytes) noexcept {
    if (!data_ || !src) return false;
    if (bytes > nbytes_) return false;

    std::memcpy(data_, src, bytes);
    return true;
}

bool Tensor::copy_to_raw(void* dst, size_t bytes) const noexcept {
    if (!data_ || !dst) return false;
    if (bytes > nbytes_) return false;

    std::memcpy(dst, data_, bytes);
    return true;
}

/* ============================================================================
 *  DEBUG
 * ========================================================================== */

std::string Tensor::to_string() const {
    std::ostringstream os;
    os << "Tensor(shape=" << shape_.to_string()
       << ", dtype=" << dtype_name(dtype_)
       << ", nbytes=" << nbytes_
       << ", owns=" << (owns_data_ ? "yes" : "no")
       << ", arena=" << (from_arena_ ? "yes" : "no")
       << ", ptr=" << data_ << ")";
    return os.str();
}

/* ============================================================================
 *  İÇ YARDIMCILAR
 * ========================================================================== */

void Tensor::assign_metadata(const Shape& shape, DType dtype) noexcept {
    shape_ = shape;
    dtype_ = dtype;
    nbytes_ = compute_nbytes(shape, dtype);
}

void Tensor::free_if_owning() noexcept {
    /* Arena'dan gelen bellek arena'ya aittir — free etmeyiz. */
    if (owns_data_ && data_ && !from_arena_) {
        memory::AlignedAllocator::deallocate(data_);
    }
    data_       = nullptr;
    owns_data_  = false;
    from_arena_ = false;
    nbytes_     = 0;
}

} /* namespace engine */