/* ============================================================================
 *  include/engine/dtype.hpp
 *  Engine-AI — Veri tipi tanımları ve özellikleri
 *
 *  Bu dosya motorun TEMEL taşıdır:
 *    - Hangi sayısal tipler destekleniyor?         → DType enum
 *    - Her tipin boyutu, işareti, kategorisi nedir? → constexpr sorgular
 *    - C++ tipi ↔ DType eşlemesi nasıl yapılır?     → dtype_of<T>() / ctype_of<>
 *    - Karışık tipte op'larda hangi tip kazanır?    → promote_dtype()
 *
 *  TASARIM İLKELERİ:
 *    1) Sıfır runtime maliyeti: Her sorgu `constexpr`, tablo lookup.
 *    2) Thread-safe: Hiçbir mutable global yok — doğal olarak çoklu
 *       çekirdekte güvenli, kilitsiz (lock-free) erişim.
 *    3) ABI stabil: `enum class DType : uint8_t` → C API'de de aynen kullanılır.
 *    4) Extensible: Yeni tip eklemek için enum + tablo satırı = 2 satır.
 * ========================================================================== */

#ifndef ENGINE_DTYPE_HPP
#define ENGINE_DTYPE_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <functional>

namespace engine {

/* ============================================================================
 *  1) DType ENUM
 *
 *  Sıralama ÖNEMLİDİR: Aşağıdaki tablolar bu sırayla indekslenir.
 *  Yeni tip eklersen, en SONA (Bool'dan sonra) ekle ve tabloları güncelle.
 * ========================================================================== */
enum class DType : uint8_t {
    Unknown = 0,   // bilinmeyen / hatalı

    // ---- Tam sayılar ----
    Bool    = 1,   // 1 bit mantıksal (8 bit'te saklanır)
    I8      = 2,   // int8_t
    U8      = 3,   // uint8_t
    I16     = 4,   // int16_t
    U16     = 5,   // uint16_t
    I32     = 6,   // int32_t
    U32     = 7,   // uint32_t
    I64     = 8,   // int64_t
    U64     = 9,   // uint64_t

    // ---- Kayan noktalı ----
    F16     = 10,  // IEEE 754 half
    BF16    = 11,  // bfloat16 (Google/Intel brain-float)
    F32     = 12,  // float
    F64     = 13,  // double

    // ---- Quantized (ileride INT8 inference için) ----
    QInt8   = 14,  // symmetric int8  (scale + zero_point yok, tam simetrik)
    QUInt8  = 15,  // asymmetric uint8 (scale + zero_point ile)
    QInt4   = 16,  // packed int4 (2 değer 1 byte'ta)

    _Count  = 17   // SENTINEL — son eleman, asla kullanma
};

/* Toplam tip sayısı — döngülerde `for (int i = 0; i < kNumDTypes; ++i)` */
inline constexpr size_t kNumDTypes = static_cast<size_t>(DType::_Count);

/* ============================================================================
 *  2) DAHİLİ TABLOLAR (detail namespace — kullanıcı görmez)
 *
 *  Neden tablo? switch'ten hızlı ve branchless. Compiler bunları statik
 *  veri olarak koyar, lookup tek bir bellek erişimi.
 * ========================================================================== */
namespace detail {

/* ---- Flag bit'leri ---- */
inline constexpr uint8_t FLAG_FLOAT     = 1u << 0;  // ondalık sayı
inline constexpr uint8_t FLAG_SIGNED    = 1u << 1;  // işaretli
inline constexpr uint8_t FLAG_INTEGER   = 1u << 2;  // tam sayı (bool dahil)
inline constexpr uint8_t FLAG_BOOL      = 1u << 3;  // mantıksal
inline constexpr uint8_t FLAG_QUANTIZED = 1u << 4;  // quantize edilmiş
inline constexpr uint8_t FLAG_PACKED    = 1u << 5;  // sub-byte (INT4 gibi)

/* ---- Boyut tablosu (BİT cinsinden) ---- */
inline constexpr uint16_t kSizeBitsTable[kNumDTypes] = {
    /* Unknown */ 0,
    /* Bool    */ 8,
    /* I8      */ 8,
    /* U8      */ 8,
    /* I16     */ 16,
    /* U16     */ 16,
    /* I32     */ 32,
    /* U32     */ 32,
    /* I64     */ 64,
    /* U64     */ 64,
    /* F16     */ 16,
    /* BF16    */ 16,
    /* F32     */ 32,
    /* F64     */ 64,
    /* QInt8   */ 8,
    /* QUInt8  */ 8,
    /* QInt4   */ 4,
};

/* ---- Flag tablosu ---- */
inline constexpr uint8_t kFlagsTable[kNumDTypes] = {
    /* Unknown */ 0,
    /* Bool    */ FLAG_BOOL | FLAG_INTEGER,
    /* I8      */ FLAG_INTEGER | FLAG_SIGNED,
    /* U8      */ FLAG_INTEGER,
    /* I16     */ FLAG_INTEGER | FLAG_SIGNED,
    /* U16     */ FLAG_INTEGER,
    /* I32     */ FLAG_INTEGER | FLAG_SIGNED,
    /* U32     */ FLAG_INTEGER,
    /* I64     */ FLAG_INTEGER | FLAG_SIGNED,
    /* U64     */ FLAG_INTEGER,
    /* F16     */ FLAG_FLOAT | FLAG_SIGNED,
    /* BF16    */ FLAG_FLOAT | FLAG_SIGNED,
    /* F32     */ FLAG_FLOAT | FLAG_SIGNED,
    /* F64     */ FLAG_FLOAT | FLAG_SIGNED,
    /* QInt8   */ FLAG_QUANTIZED | FLAG_INTEGER | FLAG_SIGNED,
    /* QUInt8  */ FLAG_QUANTIZED | FLAG_INTEGER,
    /* QInt4   */ FLAG_QUANTIZED | FLAG_INTEGER | FLAG_SIGNED | FLAG_PACKED,
};

/* ---- İsim tablosu (string_view — allocation YOK) ---- */
inline constexpr std::string_view kNameTable[kNumDTypes] = {
    "unknown", "bool",
    "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64",
    "f16", "bf16", "f32", "f64",
    "qint8", "quint8", "qint4",
};

/* ---- Promotion rank tablosu (karışık op'larda hangi tip kazanır) ----
 *  Aynı kategoride yüksek rank her zaman kazanır.
 *  Kategoriler arası: float > integer > bool. */
inline constexpr uint8_t kRankTable[kNumDTypes] = {
    /* Unknown */ 0,
    /* Bool    */ 1,
    /* I8      */ 2,  /* U8  */ 3,
    /* I16     */ 4,  /* U16 */ 5,
    /* I32     */ 6,  /* U32 */ 7,
    /* I64     */ 8,  /* U64 */ 9,
    /* F16     */ 10, /* BF16 */ 11, /* F32 */ 12, /* F64 */ 13,
    /* QInt8   */ 2,  /* QUInt8 */ 3, /* QInt4 */ 1,
};

/* Kategori sıralaması (promotion'da önce kategorilere bakılır) */
inline constexpr uint8_t kCategoryRank[kNumDTypes] = {
    /* Unknown */ 0,
    /* Bool    */ 1,
    /* I8 U8 I16 U16 I32 U32 I64 U64 */ 2,2,2,2,2,2,2,2,
    /* F16 BF16 F32 F64 */ 3,3,3,3,
    /* QInt8 QUInt8 QInt4 */ 4,4,4,
};

/* İndeksleme yardımcısı — implicit cast'lerden kurtulmak için */
inline constexpr size_t idx(DType dt) noexcept {
    const auto i = static_cast<size_t>(dt);
    return i < kNumDTypes ? i : 0;   // hatalı enum → Unknown'a düşür
}

} /* namespace detail */

/* ============================================================================
 *  3) TEMEL SORGULAR (hepsi constexpr, hepsi inline)
 * ========================================================================== */

/* ---- Boyut: byte cinsinden (sub-byte tiplerde 0 döner) ---- */
[[nodiscard]] constexpr size_t dtype_size(DType dt) noexcept {
    return detail::kSizeBitsTable[detail::idx(dt)] / 8;
}

/* ---- Boyut: bit cinsinden (INT4 gibi tipler için şart) ---- */
[[nodiscard]] constexpr size_t dtype_bits(DType dt) noexcept {
    return detail::kSizeBitsTable[detail::idx(dt)];
}

/* ---- İsim (allocation yok, pointer stable) ---- */
[[nodiscard]] constexpr std::string_view dtype_name(DType dt) noexcept {
    return detail::kNameTable[detail::idx(dt)];
}

/* ---- Kategori kontrolleri ---- */
[[nodiscard]] constexpr bool dtype_is_floating(DType dt) noexcept {
    return (detail::kFlagsTable[detail::idx(dt)] & detail::FLAG_FLOAT) != 0;
}
[[nodiscard]] constexpr bool dtype_is_integer(DType dt) noexcept {
    return (detail::kFlagsTable[detail::idx(dt)] & detail::FLAG_INTEGER) != 0;
}
[[nodiscard]] constexpr bool dtype_is_signed(DType dt) noexcept {
    return (detail::kFlagsTable[detail::idx(dt)] & detail::FLAG_SIGNED) != 0;
}
[[nodiscard]] constexpr bool dtype_is_bool(DType dt) noexcept {
    return (detail::kFlagsTable[detail::idx(dt)] & detail::FLAG_BOOL) != 0;
}
[[nodiscard]] constexpr bool dtype_is_quantized(DType dt) noexcept {
    return (detail::kFlagsTable[detail::idx(dt)] & detail::FLAG_QUANTIZED) != 0;
}
[[nodiscard]] constexpr bool dtype_is_packed(DType dt) noexcept {
    return (detail::kFlagsTable[detail::idx(dt)] & detail::FLAG_PACKED) != 0;
}
[[nodiscard]] constexpr bool dtype_is_valid(DType dt) noexcept {
    return dt != DType::Unknown && static_cast<size_t>(dt) < kNumDTypes;
}

/* ---- Doğal hizalama (SIMD için önemli) ----
 *  Not: SIMD yüklemeleri 32/64-byte ister ama bu, TENSOR'ın bellek
 *  hizalaması meselesi (bkz. aligned_allocator.hpp). Buradaki değer,
 *  tipin doğal C++ hizalamasıdır. */
[[nodiscard]] constexpr size_t dtype_alignof(DType dt) noexcept {
    // Basit kural: doğal hizalama = boyut (max 8)
    const size_t s = dtype_size(dt);
    return s > 8 ? 8 : (s == 0 ? 1 : s);
}

/* ---- SIMD lane sayısı: bir vektör register'ına kaç eleman sığar? ----
 *  Örnek: F32 + 256-bit (AVX2) = 8 lane
 *         F32 + 512-bit (AVX512) = 16 lane
 *  Runtime CPU bilgisine göre kernel tarafından kullanılır. */
[[nodiscard]] constexpr int dtype_simd_lanes(DType dt, int vector_bits) noexcept {
    const size_t bits = dtype_bits(dt);
    if (bits == 0) return 0;
    return vector_bits / static_cast<int>(bits);
}

/* ============================================================================
 *  4) COMPILE-TIME TİP EŞLEMESİ (C++ tipi ↔ DType)
 *
 *  Kullanım:
 *      DType dt = engine::dtype_of<float>();   // DType::F32
 *  Bu, template kernel'lerde tip güvenliği sağlar.
 * ========================================================================== */

/* İleri bildirim: bu tipler başka bir başlıkta tanımlanacak */
struct float16;    // half precision
struct bfloat16;   // brain float

namespace detail {

template <typename T> struct dtype_traits { static constexpr DType value = DType::Unknown; };

template <> struct dtype_traits<bool>     { static constexpr DType value = DType::Bool;  };
template <> struct dtype_traits<int8_t>   { static constexpr DType value = DType::I8;    };
template <> struct dtype_traits<uint8_t>  { static constexpr DType value = DType::U8;    };
template <> struct dtype_traits<int16_t>  { static constexpr DType value = DType::I16;   };
template <> struct dtype_traits<uint16_t> { static constexpr DType value = DType::U16;   };
template <> struct dtype_traits<int32_t>  { static constexpr DType value = DType::I32;   };
template <> struct dtype_traits<uint32_t> { static constexpr DType value = DType::U32;   };
template <> struct dtype_traits<int64_t>  { static constexpr DType value = DType::I64;   };
template <> struct dtype_traits<uint64_t> { static constexpr DType value = DType::U64;   };
template <> struct dtype_traits<float>    { static constexpr DType value = DType::F32;   };
template <> struct dtype_traits<double>   { static constexpr DType value = DType::F64;   };
template <> struct dtype_traits<float16>  { static constexpr DType value = DType::F16;   };
template <> struct dtype_traits<bfloat16> { static constexpr DType value = DType::BF16;  };

} /* namespace detail */

/* C++ tipinden DType'a */
template <typename T>
[[nodiscard]] constexpr DType dtype_of() noexcept {
    return detail::dtype_traits<std::remove_cv_t<std::remove_reference_t<T>>>::value;
}

/* DType'tan C++ tipine (compile-time) */
template <DType DT> struct ctype_of;
template <> struct ctype_of<DType::Bool>  { using type = bool;     };
template <> struct ctype_of<DType::I8>    { using type = int8_t;   };
template <> struct ctype_of<DType::U8>    { using type = uint8_t;  };
template <> struct ctype_of<DType::I16>   { using type = int16_t;  };
template <> struct ctype_of<DType::U16>   { using type = uint16_t; };
template <> struct ctype_of<DType::I32>   { using type = int32_t;  };
template <> struct ctype_of<DType::U32>   { using type = uint32_t; };
template <> struct ctype_of<DType::I64>   { using type = int64_t;  };
template <> struct ctype_of<DType::U64>   { using type = uint64_t; };
template <> struct ctype_of<DType::F16>   { using type = float16;  };
template <> struct ctype_of<DType::BF16>  { using type = bfloat16; };
template <> struct ctype_of<DType::F32>   { using type = float;    };
template <> struct ctype_of<DType::F64>   { using type = double;   };
template <> struct ctype_of<DType::QInt8> { using type = int8_t;   };
template <> struct ctype_of<DType::QUInt8>{ using type = uint8_t;  };

template <DType DT> using ctype_of_t = typename ctype_of<DT>::type;

/* ============================================================================
 *  5) STRING ↔ DType (parse & serialize)
 * ========================================================================== */

[[nodiscard]] constexpr DType dtype_from_string(std::string_view s) noexcept {
    // Linear search — tablo sadece 17 eleman, cache'de hep sıcak
    for (size_t i = 0; i < kNumDTypes; ++i) {
        if (detail::kNameTable[i] == s) return static_cast<DType>(i);
    }
    return DType::Unknown;
}

/* ============================================================================
 *  6) TYPE PROMOTION (karışık tipli op'lar için)
 *
 *  Kural özeti:
 *    - Aynı tip → o tip
 *    - Float > Integer > Bool (kategori)
 *    - Aynı kategoride daha geniş olan kazanır
 *    - Quantized tipler, non-quantized ile karşılaşırsa non-quantized kazanır
 *    - Unknown her zaman Unknown'a zehirler
 *
 *  Kullanım: Conv + Bias gibi op'larda hangi tipte hesaplanacağına karar verir.
 * ========================================================================== */
[[nodiscard]] constexpr DType promote_dtype(DType a, DType b) noexcept {
    if (a == b)              return a;
    if (a == DType::Unknown) return DType::Unknown;
    if (b == DType::Unknown) return DType::Unknown;

    const auto ia = detail::idx(a);
    const auto ib = detail::idx(b);

    // Quantized bir taraf varsa, non-quantized taraf kazanır
    const bool qa = (detail::kFlagsTable[ia] & detail::FLAG_QUANTIZED) != 0;
    const bool qb = (detail::kFlagsTable[ib] & detail::FLAG_QUANTIZED) != 0;
    if (qa != qb) return qa ? b : a;

    // Kategori karşılaştırma
    const uint8_t ca = detail::kCategoryRank[ia];
    const uint8_t cb = detail::kCategoryRank[ib];
    if (ca != cb) return ca > cb ? a : b;

    // Aynı kategori → yüksek rank kazanır
    return detail::kRankTable[ia] >= detail::kRankTable[ib] ? a : b;
}

/* 3'lü ve 4'lü promotion kısayolları (ileride graph optimizer kullanacak) */
[[nodiscard]] constexpr DType promote_dtype(DType a, DType b, DType c) noexcept {
    return promote_dtype(promote_dtype(a, b), c);
}
[[nodiscard]] constexpr DType promote_dtype(DType a, DType b, DType c, DType d) noexcept {
    return promote_dtype(promote_dtype(a, b, c), d);
}

/* ============================================================================
 *  7) OPERATÖRLER (enum karşılaştırma için)
 *
 *  enum class karşılaştırmayı destekler ama bazı yardımcılar işleri kolaylaştırır.
 * ========================================================================== */
[[nodiscard]] constexpr bool operator<(DType a, DType b) noexcept {
    return static_cast<uint8_t>(a) < static_cast<uint8_t>(b);
}

/* ============================================================================
 *  8) std::hash DESTEĞİ (unordered_map<DType, ...> için)
 * ========================================================================== */
namespace detail {

struct DTypeHash {
    [[nodiscard]] size_t operator()(DType dt) const noexcept {
        // DType zaten uint8_t, direkt hash'i yeterli — std::hash çağırmaya gerek yok
        return static_cast<size_t>(dt);
    }
};

} /* namespace detail */

} /* namespace engine */

/* ---- std::hash uzmanlığı (global namespace'te olmalı) ---- */
template <>
struct std::hash<engine::DType> {
    [[nodiscard]] size_t operator()(engine::DType dt) const noexcept {
        return static_cast<size_t>(dt);
    }
};

#endif /* ENGINE_DTYPE_HPP */