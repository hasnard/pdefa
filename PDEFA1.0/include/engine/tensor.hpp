/* ============================================================================
 *  include/engine/tensor.hpp
 *  Engine-AI — Ana veri kabı (Tensor)
 *
 *  NE YAPAR?
 *    - Çok boyutlu sayısal veri tutar (float, int8, vs.)
 *    - Şekil (Shape) + veri tipi (DType) + ham bellek (data) birleşimi
 *    - Owning (sahibi) veya non-owning (view) olabilir
 *    - Arena'dan sıfır-malloc tahsis edilebilir
 *    - Zero-copy view / reshape destekler
 *
 *  HIZ NEDEN ÖNEMLİ?
 *    Bir YOLO inference'ta ~500 tensor yaratılır/yok edilir.
 *    PyTorch her biri için ayrı allocation yapar → ~50 µs overhead.
 *    Bizim Tensor:
 *      - Arena ile 0 malloc
 *      - View ile 0 copy
 *      - Move ile 0 pointer churn
 *
 *  BELLEK DÜZENİ (row-major, C-contiguous):
 *
 *      Shape: {2, 3, 4}   numel = 24
 *      Strides: {12, 4, 1}
 *
 *      flat index = i*12 + j*4 + k
 *      data[flat_index]  ← tek bir lineer erişim
 *
 *  TASARIM İLKELERİ:
 *    1) Move-only (kazara kopya = bug kaynağı)
 *    2) Sıcak yol = noexcept + inline
 *    3) Zero-copy view'lar (performans kraldır)
 *    4) Arena-aware (inference sırasında malloc yok)
 *    5) Type-safe erişim (`data<T>()` — compile-time kontrol)
 * ========================================================================== */

#ifndef ENGINE_TENSOR_HPP
#define ENGINE_TENSOR_HPP

#include "engine/dtype.hpp"
#include "engine/shape.hpp"
#include "engine/memory/arena.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

namespace engine {

/* ============================================================================
 *  FORWARD DECLARATIONS
 * ========================================================================== */
class Tensor;

/* ============================================================================
 *  TENSOR SINIFI
 * ========================================================================== */

class Tensor {
public:
    /* ------------------------------------------------------------------------
     *  KURUCULAR — SAHİPLİK SEMANTİĞİ
     * ---------------------------------------------------------------------- */

    /* Boş tensor (skaler değil, sadece "hiç veri yok"). */
    Tensor() noexcept = default;

    /* Sahipli tensor — kendi belleğini ayırır (64-byte hizalı).
     * Not: Sıcak yolda DEĞİL, genelde model yüklemede kullanılır. */
    [[nodiscard]] static Tensor empty(const Shape& shape, DType dtype) noexcept;

    /* Sahipli, sıfırla doldurulmuş. */
    [[nodiscard]] static Tensor zeros(const Shape& shape, DType dtype) noexcept;

    /* Sahipli, bir değerle doldurulmuş (sadece F32/I32 için). */
    [[nodiscard]] static Tensor filled(const Shape& shape, DType dtype, float value) noexcept;

    /* Non-owning view — mevcut belleğe referans (kopya YOK).
     * Kullanıcı belleğin ömrünü garanti etmelidir. */
    [[nodiscard]] static Tensor from_external(
        void* data,
        const Shape& shape,
        DType dtype) noexcept;

    /* Arena'dan tahsis — sıfır malloc, inference için ideal. */
    [[nodiscard]] static Tensor create_in_arena(
        memory::Arena& arena,
        const Shape& shape,
        DType dtype) noexcept;

    /* Arena'dan, sıfırla doldurulmuş. */
    [[nodiscard]] static Tensor zeros_in_arena(
        memory::Arena& arena,
        const Shape& shape,
        DType dtype) noexcept;

    /* ----------------------------------------------------------------
     *  KOPYALAMA / TAŞIMA
     * ---------------------------------------------------------------- */
    Tensor(const Tensor&)            = delete;   // kazara kopya yok
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor() noexcept;

    /* ------------------------------------------------------------------------
     *  TEMEL SORGULAR (sıcak yol — hepsi inline, noexcept)
     * ---------------------------------------------------------------------- */

    [[nodiscard]] const Shape&  shape()   const noexcept { return shape_; }
    [[nodiscard]] DType         dtype()   const noexcept { return dtype_; }
    [[nodiscard]] int64_t       numel()   const noexcept { return shape_.numel(); }
    [[nodiscard]] int32_t       rank()    const noexcept { return shape_.rank(); }
    [[nodiscard]] size_t        nbytes()  const noexcept { return nbytes_; }

    [[nodiscard]] bool is_empty()      const noexcept { return data_ == nullptr; }
    [[nodiscard]] bool is_contiguous() const noexcept { return shape_.is_contiguous(); }
    [[nodiscard]] bool is_owning()     const noexcept { return owns_data_; }
    [[nodiscard]] bool is_view()       const noexcept { return !owns_data_ && data_ != nullptr; }
    [[nodiscard]] bool is_scalar()     const noexcept { return shape_.is_scalar(); }

    /* Boyut erişimi */
    [[nodiscard]] int64_t dim(int32_t axis)    const noexcept { return shape_.dim(axis); }
    [[nodiscard]] int64_t stride(int32_t axis) const noexcept { return shape_.stride(axis); }

    /* ------------------------------------------------------------------------
     *  VERİ ERİŞİMİ (sıcak yol)
     * ---------------------------------------------------------------------- */

    /* Ham pointer (void*) — C API, generic kod için. */
    [[nodiscard]] void*       data()       noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }

    /* Tip güvenli pointer. Yanlış T → derleme hatası. */
    template <typename T>
    [[nodiscard]] T* data() noexcept {
        static_assert(!std::is_pointer_v<T>, "data<T>() expects value type");
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
        return static_cast<T*>(data_);
    }

    template <typename T>
    [[nodiscard]] const T* data() const noexcept {
        static_assert(!std::is_pointer_v<T>, "data<T>() expects value type");
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
        return static_cast<const T*>(data_);
    }

    /* Runtime tip kontrolü ile pointer (nadiren gerekir). */
    [[nodiscard]] void* data_as(DType expected) noexcept;
    [[nodiscard]] const void* data_as(DType expected) const noexcept;

    /* ------------------------------------------------------------------------
     *  İNDEKSLEME (sıcak yol)
     * ---------------------------------------------------------------------- */

    /* 1D erişim (flat). Bounds check YOK (release'de). */
    template <typename T>
    [[nodiscard]] T& at_flat(int64_t i) noexcept {
        return static_cast<T*>(data_)[i];
    }
    template <typename T>
    [[nodiscard]] const T& at_flat(int64_t i) const noexcept {
        return static_cast<const T*>(data_)[i];
    }

    /* N-boyutlu erişim (variadic template — compile-time rank). */
    template <typename T, typename... Idx>
    [[nodiscard]] T& at(Idx... idx) noexcept {
        static_assert(sizeof...(Idx) > 0, "at() requires at least one index");
        return static_cast<T*>(data_)[flat_index(idx...)];
    }
    template <typename T, typename... Idx>
    [[nodiscard]] const T& at(Idx... idx) const noexcept {
        static_assert(sizeof...(Idx) > 0, "at() requires at least one index");
        return static_cast<const T*>(data_)[flat_index(idx...)];
    }

    /* Pointer offset — kernel'ler için (bounds check YOK). */
    [[nodiscard]] void* offset_ptr(int64_t flat_offset) noexcept {
        return static_cast<uint8_t*>(data_) + flat_offset * dtype_size(dtype_);
    }
    [[nodiscard]] const void* offset_ptr(int64_t flat_offset) const noexcept {
        return static_cast<const uint8_t*>(data_) + flat_offset * dtype_size(dtype_);
    }

    /* ------------------------------------------------------------------------
     *  DÖNÜŞÜMLER — ZERO-COPY (view) VEYA COPY (clone)
     * ---------------------------------------------------------------------- */

    /* View: mevcut belleği paylaşır. Sadece contiguous tensor için. */
    [[nodiscard]] std::optional<Tensor> view(const Shape& new_shape) const noexcept;

    /* Clone: yeni bellek, veri kopyalanır. Arena verilirse oradan. */
    [[nodiscard]] Tensor clone() const noexcept;
    [[nodiscard]] Tensor clone_in_arena(memory::Arena& arena) const noexcept;

    /* Squeeze / unsqueeze (view döner — ucuz). */
    [[nodiscard]] std::optional<Tensor> squeeze(int32_t axis) const noexcept;
    [[nodiscard]] std::optional<Tensor> unsqueeze(int32_t axis) const noexcept;

    /* ------------------------------------------------------------------------
     *  VERİ İŞLEMLERİ
     * ---------------------------------------------------------------------- */

    /* Belleği sıfırla (memset). */
    void zero_() noexcept;

    /* F32/I32 için değer doldur. */
    void fill_(float value) noexcept;

    /* Başka bir tensor'dan kopyala (aynı shape + dtype şart). */
    [[nodiscard]] bool copy_from(const Tensor& src) noexcept;

    /* Ham bellekten kopyala. */
    [[nodiscard]] bool copy_from_raw(const void* src, size_t bytes) noexcept;

    /* Ham belleğe kopyala. */
    [[nodiscard]] bool copy_to_raw(void* dst, size_t bytes) const noexcept;

    /* ------------------------------------------------------------------------
     *  DEBUG / LOG
     * ---------------------------------------------------------------------- */
    [[nodiscard]] std::string to_string() const;

private:
    /* ------------------------------------------------------------------------
     *  İÇ VERİ
     * ---------------------------------------------------------------------- */

    void*   data_       = nullptr;
    Shape   shape_{};
    DType   dtype_      = DType::Unknown;
    size_t  nbytes_     = 0;         // data'nın toplam byte boyutu
    bool    owns_data_  = false;     // true → destructor'da serbest bırak
    bool    from_arena_ = false;     // true → destructor free etmesin (arena yönetir)

    /* Flat index hesaplama (variadic template). */
    template <typename... Idx>
    [[nodiscard]] int64_t flat_index(Idx... idx) const noexcept {
        static_assert(sizeof...(Idx) <= static_cast<size_t>(kMaxNdim),
                      "Too many indices");
        const int64_t arr[sizeof...(Idx)] = { static_cast<int64_t>(idx)... };
        int64_t offset = 0;
        const int32_t n = static_cast<int32_t>(sizeof...(Idx));
        /* Row-major: offset = Σ idx[i] * stride[i] */
        for (int32_t i = 0; i < n; ++i) {
            offset += arr[i] * shape_.stride(i);
        }
        return offset;
    }

    /* İç yardımcı — constructor'lardan çağrılır. */
    void assign_metadata(const Shape& shape, DType dtype) noexcept;
    void free_if_owning() noexcept;
};

/* ============================================================================
 *  KISA YARDIMCILAR (inline, header-only)
 * ========================================================================== */

/* Byte hesabı — Shape::numel * dtype_size (bit cinsinden). */
[[nodiscard]] inline size_t compute_nbytes(const Shape& s, DType dt) noexcept {
    const size_t bits = static_cast<size_t>(s.numel()) * dtype_bits(dt);
    return (bits + 7) / 8;
}

/* İki tensor aynı metadata'ya sahip mi? (veri karşılaştırması YOK) */
[[nodiscard]] inline bool same_metadata(const Tensor& a, const Tensor& b) noexcept {
    return a.dtype() == b.dtype() && a.shape() == b.shape();
}

} /* namespace engine */

#endif /* ENGINE_TENSOR_HPP */