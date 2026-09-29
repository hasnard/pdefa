/* ============================================================================
 *  src/api/engine_api.cpp
 *  Engine-AI — C API implementasyonu
 *
 *  C++ dünyasını C dünyasına köprüler. try/catch ile sarılı, exception sızdırmaz.
 *
 *  ÖNEMLİ: EngineTensor artık iki mod destekler:
 *    - owned    : sahibi olduğumuz tensor (kullanıcı destroy eder)
 *    - borrowed : başkasının (örn. Session) tensor'ı (destroy edilmez)
 * ========================================================================== */

#include "engine/engine_c_api.h"
#include "engine/tensor.hpp"
#include "engine/model.hpp"
#include "engine/session.hpp"
#include "engine/loader/custom_loader.hpp"
#include "engine/runtime/cpu_info.hpp"
#include "engine/kernels/kernel_registry.hpp"
#include "engine/version.hpp"
#include "engine/ops/matmul.hpp"
#include "engine/ops/activation.hpp"
#include "engine/ops/elementwise.hpp"   /* ADD */


#include <random>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <cmath>                        /* ADD: std::exp */
#include <algorithm>
#include <immintrin.h>                  /* ADD: _mm256_* */

namespace {

constexpr uint32_t kMagicContext = 0xE0C0DE01u;
constexpr uint32_t kMagicModel   = 0xE0C0DE02u;
constexpr uint32_t kMagicSession = 0xE0C0DE03u;
constexpr uint32_t kMagicTensor  = 0xE0C0DE04u;

thread_local std::string g_last_error;

inline void set_error(const char* msg) noexcept {
    try { g_last_error = msg ? msg : ""; } catch (...) {}
}
inline void clear_error() noexcept { try { g_last_error.clear(); } catch (...) {} }

template <typename Fn>
EngineStatus safe_call(Fn&& fn) noexcept {
    try { return fn(); }
    catch (const std::bad_alloc&) { set_error("out of memory"); return ENGINE_ERR_OUT_OF_MEMORY; }
    catch (const std::exception& e) { set_error(e.what()); return ENGINE_ERR_RUNTIME; }
    catch (...) { set_error("unknown exception"); return ENGINE_ERR_UNKNOWN; }
}

engine::DType to_engine_dtype(EngineDType dt) noexcept {
    switch (dt) {
        case ENGINE_F32:  return engine::DType::F32;
        case ENGINE_F16:  return engine::DType::F16;
        case ENGINE_BF16: return engine::DType::BF16;
        case ENGINE_I8:   return engine::DType::I8;
        case ENGINE_U8:   return engine::DType::U8;
        case ENGINE_I32:  return engine::DType::I32;
        case ENGINE_I64:  return engine::DType::I64;
        case ENGINE_BOOL: return engine::DType::Bool;
    }
    return engine::DType::Unknown;
}

EngineDType from_engine_dtype(engine::DType dt) noexcept {
    switch (dt) {
        case engine::DType::F32:  return ENGINE_F32;
        case engine::DType::F16:  return ENGINE_F16;
        case engine::DType::BF16: return ENGINE_BF16;
        case engine::DType::I8:   return ENGINE_I8;
        case engine::DType::U8:   return ENGINE_U8;
        case engine::DType::I32:  return ENGINE_I32;
        case engine::DType::I64:  return ENGINE_I64;
        case engine::DType::Bool: return ENGINE_BOOL;
        default:                  return ENGINE_F32;
    }
}

/* ============================================================================
 *  OPS — yardımcılar (binary broadcast + unary check)
 * ========================================================================== */

enum class BinOp { Add, Sub, Mul, Div };

inline bool compute_broadcast_shape(
    const engine::Tensor& a, const engine::Tensor& b,
    int64_t* out_dims, int32_t* out_rank) noexcept
{
    const int32_t ra = a.rank();
    const int32_t rb = b.rank();
    const int32_t ro = ra > rb ? ra : rb;
    if (ro > 8) return false;
    for (int32_t d = 0; d < ro; ++d) {
        const int64_t da = (d < ro - ra) ? 1 : a.dim(d - (ro - ra));
        const int64_t db = (d < ro - rb) ? 1 : b.dim(d - (ro - rb));
        if (da != db && da != 1 && db != 1) return false;
        out_dims[d] = (da > db) ? da : db;
    }
    *out_rank = ro;
    return true;
}

inline void run_binary_broadcast(
    const engine::Tensor& a, const engine::Tensor& b, engine::Tensor& out,
    BinOp op) noexcept
{
    const int32_t ro = out.rank();
    const int32_t ra = a.rank();
    const int32_t rb = b.rank();

    const int64_t na = a.numel();
    const int64_t nb = b.numel();
    const int64_t no = out.numel();

    const float* ap = a.data<float>();
    const float* bp = b.data<float>();
    float*       op_ = out.data<float>();

    /* Fast path 1: tamamen aynı shape */
    if (na == no && nb == no) {
        int64_t i = 0;
        switch (op) {
            case BinOp::Add:
                for (; i + 7 < no; i += 8)
                    _mm256_storeu_ps(op_ + i, _mm256_add_ps(
                        _mm256_loadu_ps(ap + i), _mm256_loadu_ps(bp + i)));
                for (; i < no; ++i) op_[i] = ap[i] + bp[i];
                break;
            case BinOp::Sub:
                for (; i + 7 < no; i += 8)
                    _mm256_storeu_ps(op_ + i, _mm256_sub_ps(
                        _mm256_loadu_ps(ap + i), _mm256_loadu_ps(bp + i)));
                for (; i < no; ++i) op_[i] = ap[i] - bp[i];
                break;
            case BinOp::Mul:
                for (; i + 7 < no; i += 8)
                    _mm256_storeu_ps(op_ + i, _mm256_mul_ps(
                        _mm256_loadu_ps(ap + i), _mm256_loadu_ps(bp + i)));
                for (; i < no; ++i) op_[i] = ap[i] * bp[i];
                break;
            case BinOp::Div:
                for (; i < no; ++i) op_[i] = ap[i] / bp[i];
                break;
        }
        return;
    }

    /* Fast path 2: b scalar */
    if (nb == 1) {
        const float bv = bp[0];
        const __m256 bv8 = _mm256_set1_ps(bv);
        int64_t i = 0;
        switch (op) {
            case BinOp::Add:
                for (; i + 7 < no; i += 8)
                    _mm256_storeu_ps(op_ + i, _mm256_add_ps(_mm256_loadu_ps(ap + i), bv8));
                for (; i < no; ++i) op_[i] = ap[i] + bv;
                break;
            case BinOp::Sub:
                for (; i + 7 < no; i += 8)
                    _mm256_storeu_ps(op_ + i, _mm256_sub_ps(_mm256_loadu_ps(ap + i), bv8));
                for (; i < no; ++i) op_[i] = ap[i] - bv;
                break;
            case BinOp::Mul:
                for (; i + 7 < no; i += 8)
                    _mm256_storeu_ps(op_ + i, _mm256_mul_ps(_mm256_loadu_ps(ap + i), bv8));
                for (; i < no; ++i) op_[i] = ap[i] * bv;
                break;
            case BinOp::Div:
                for (; i < no; ++i) op_[i] = ap[i] / bv;
                break;
        }
        return;
    }

    /* Fast path 3: son-eksen channel broadcast (SE bloğu) */
    const int64_t C_last = out.dim(ro - 1);
    bool b_is_channel = (rb >= 1 && b.dim(rb - 1) == C_last);
    if (b_is_channel) {
        for (int32_t d = 0; d < rb - 1; ++d)
            if (b.dim(d) != 1) { b_is_channel = false; break; }
    }
    if (b_is_channel && nb == C_last && na == no) {
        const int64_t outer = no / C_last;
        for (int64_t o = 0; o < outer; ++o) {
            const float* ar = ap + o * C_last;
            float*       orow = op_ + o * C_last;
            int64_t c = 0;
            switch (op) {
                case BinOp::Add:
                    for (; c + 7 < C_last; c += 8)
                        _mm256_storeu_ps(orow + c, _mm256_add_ps(
                            _mm256_loadu_ps(ar + c), _mm256_loadu_ps(bp + c)));
                    for (; c < C_last; ++c) orow[c] = ar[c] + bp[c];
                    break;
                case BinOp::Sub:
                    for (; c + 7 < C_last; c += 8)
                        _mm256_storeu_ps(orow + c, _mm256_sub_ps(
                            _mm256_loadu_ps(ar + c), _mm256_loadu_ps(bp + c)));
                    for (; c < C_last; ++c) orow[c] = ar[c] - bp[c];
                    break;
                case BinOp::Mul:
                    for (; c + 7 < C_last; c += 8)
                        _mm256_storeu_ps(orow + c, _mm256_mul_ps(
                            _mm256_loadu_ps(ar + c), _mm256_loadu_ps(bp + c)));
                    for (; c < C_last; ++c) orow[c] = ar[c] * bp[c];
                    break;
                case BinOp::Div:
                    for (; c + 7 < C_last; c += 8)
                        _mm256_storeu_ps(orow + c, _mm256_div_ps(
                            _mm256_loadu_ps(ar + c), _mm256_loadu_ps(bp + c)));
                    for (; c < C_last; ++c) orow[c] = ar[c] / bp[c];
                    break;
            }
        }
        return;
    }

    /* Slow path: genel broadcast (nadir) */
    int64_t out_dims[8];
    for (int32_t d = 0; d < ro; ++d) out_dims[d] = out.dim(d);
    for (int64_t i = 0; i < no; ++i) {
        int64_t t = i;
        int64_t o_idx[8];
        for (int32_t d = ro - 1; d >= 0; --d) {
            o_idx[d] = t % out_dims[d];
            t /= out_dims[d];
        }
        int64_t ai = 0, bi = 0;
        for (int32_t d = 0; d < ra; ++d)
            ai = ai * a.dim(d) + ((a.dim(d) == 1) ? 0 : o_idx[ro - ra + d]);
        for (int32_t d = 0; d < rb; ++d)
            bi = bi * b.dim(d) + ((b.dim(d) == 1) ? 0 : o_idx[ro - rb + d]);
        const float av = ap[ai], bv = bp[bi];
        switch (op) {
            case BinOp::Add: op_[i] = av + bv; break;
            case BinOp::Sub: op_[i] = av - bv; break;
            case BinOp::Mul: op_[i] = av * bv; break;
            case BinOp::Div: op_[i] = av / bv; break;
        }
    }
}


} /* anonymous namespace */

/* ============================================================================
 *  HANDLE STRUCT'LARI — opaque'ların gerçek yüzü
 * ========================================================================== */

struct EngineTensor {
    uint32_t magic = kMagicTensor;

    /* Sahibi olduğumuz tensor (kullanıcı destroy eder) */
    std::unique_ptr<engine::Tensor> owned;

    /* Non-owning: başkasının tensor'ına bakıyoruz (örn. Session output).
     * Bu durumda destroy ÇAĞRILMAZ. */
    engine::Tensor* borrowed = nullptr;

    /* Tek erişim noktası — hangisi dolu ise onu döner */
    engine::Tensor*       ptr()       noexcept { return owned ? owned.get() : borrowed; }
    const engine::Tensor* ptr() const noexcept { return owned ? owned.get() : borrowed; }

    bool is_borrowed() const noexcept { return borrowed != nullptr && !owned; }
};

struct EngineContext {
    uint32_t magic = kMagicContext;
    int32_t  num_threads = 0;
};

struct EngineModel {
    uint32_t magic = kMagicModel;
    std::unique_ptr<engine::Model> inner;
};

struct EngineSession {
    uint32_t magic = kMagicSession;
    std::unique_ptr<engine::Session> inner;
};



/* ============================================================================
 *  OPS yardımcıları — EngineTensor tam tanımından SONRA
 *  (magic/ptr/is_borrowed alanlarına erişir)
 * ========================================================================== */
namespace {

inline EngineStatus binary_op_impl(
    const EngineTensor* a, const EngineTensor* b,
    EngineTensor** out, BinOp op) noexcept
{
    if (!a || a->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!b || b->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!out) return ENGINE_ERR_NULL_PTR;
    *out = nullptr;

    return safe_call([&]() -> EngineStatus {
        clear_error();
        const engine::Tensor& ta = *a->ptr();
        const engine::Tensor& tb = *b->ptr();

        int64_t dims[8];
        int32_t rank = 0;
        if (!compute_broadcast_shape(ta, tb, dims, &rank)) {
            set_error("broadcast shape mismatch");
            return ENGINE_ERR_INVALID_SHAPE;
        }
        engine::Shape os(dims, rank);

        auto* ot = new (std::nothrow) EngineTensor();
        if (!ot) return ENGINE_ERR_OUT_OF_MEMORY;
        ot->owned = std::make_unique<engine::Tensor>(
            engine::Tensor::empty(os, engine::DType::F32));
        if (ot->owned->is_empty() && os.numel() > 0) {
            delete ot;
            return ENGINE_ERR_OUT_OF_MEMORY;
        }
        run_binary_broadcast(ta, tb, *ot->owned, op);
        *out = ot;
        return ENGINE_OK;
    });
}

inline EngineStatus unary_check(EngineTensor* t) noexcept {
    if (!t || t->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (t->is_borrowed()) {
        set_error("cannot modify borrowed tensor");
        return ENGINE_ERR_INVALID_ARG;
    }
    if (t->ptr()->dtype() != engine::DType::F32) {
        set_error("only F32 supported");
        return ENGINE_ERR_INVALID_ARG;
    }
    return ENGINE_OK;
}

} /* anonymous namespace */






/* ============================================================================
 *  META
 * ========================================================================== */

extern "C" ENGINE_API const char* ENGINE_CALL engine_version(void) {
#ifdef ENGINE_VERSION_STRING
    return ENGINE_VERSION_STRING;
#else
    return "0.1.0";
#endif
}

extern "C" ENGINE_API const char* ENGINE_CALL engine_last_error(void) {
    return g_last_error.empty() ? "" : g_last_error.c_str();
}

extern "C" ENGINE_API const char* ENGINE_CALL engine_status_string(EngineStatus s) {
    switch (s) {
        case ENGINE_OK:                 return "OK";
        case ENGINE_ERR_NULL_PTR:       return "null pointer";
        case ENGINE_ERR_INVALID_HANDLE: return "invalid handle";
        case ENGINE_ERR_OUT_OF_MEMORY:  return "out of memory";
        case ENGINE_ERR_FILE_NOT_FOUND: return "file not found";
        case ENGINE_ERR_INVALID_FORMAT: return "invalid format";
        case ENGINE_ERR_INVALID_SHAPE:  return "invalid shape";
        case ENGINE_ERR_INVALID_ARG:    return "invalid argument";
        case ENGINE_ERR_RUNTIME:        return "runtime error";
        case ENGINE_ERR_NOT_IMPL:       return "not implemented";
        default:                        return "unknown";
    }
}

extern "C" ENGINE_API size_t ENGINE_CALL engine_dtype_size(EngineDType dt) {
    return engine::dtype_size(to_engine_dtype(dt));
}

extern "C" ENGINE_API const char* ENGINE_CALL engine_simd_level(void) {
    return engine::kernels::Kernels::active_level_name();
}

extern "C" ENGINE_API const char* ENGINE_CALL engine_cpu_brand(void) {
    static std::string s;
    if (s.empty()) s = engine::runtime::Cpu::info().brand;
    return s.c_str();
}

/* ============================================================================
 *  CONTEXT
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_context_create(EngineContext** out_ctx) {
    if (!out_ctx) { set_error("out_ctx is null"); return ENGINE_ERR_NULL_PTR; }
    *out_ctx = nullptr;
    return safe_call([&]() -> EngineStatus {
        clear_error();
        auto* c = new (std::nothrow) EngineContext();
        if (!c) return ENGINE_ERR_OUT_OF_MEMORY;
        c->num_threads = engine::runtime::Cpu::topology().recommended_threads();
        *out_ctx = c;
        return ENGINE_OK;
    });
}

extern "C" ENGINE_API void ENGINE_CALL engine_context_destroy(EngineContext* ctx) {
    if (!ctx || ctx->magic != kMagicContext) return;
    ctx->magic = 0;
    delete ctx;
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_context_set_num_threads(EngineContext* ctx, int32_t n) {
    if (!ctx || ctx->magic != kMagicContext) return ENGINE_ERR_INVALID_HANDLE;
    if (n <= 0) return ENGINE_ERR_INVALID_ARG;
    ctx->num_threads = n;
    engine::runtime::GlobalPool::set_num_threads(n);
    return ENGINE_OK;
}

extern "C" ENGINE_API int32_t ENGINE_CALL engine_context_get_num_threads(const EngineContext* ctx) {
    if (!ctx || ctx->magic != kMagicContext) return 0;
    return ctx->num_threads;
}

extern "C" ENGINE_API int32_t ENGINE_CALL engine_context_num_physical_cores(void) {
    return engine::runtime::Cpu::topology().physical_cores;
}

extern "C" ENGINE_API int32_t ENGINE_CALL engine_context_has_avx2(void) {
    return engine::runtime::Cpu::features().avx2 ? 1 : 0;
}

extern "C" ENGINE_API int32_t ENGINE_CALL engine_context_has_avx512(void) {
    return engine::runtime::Cpu::features().has_avx512() ? 1 : 0;
}

/* ============================================================================
 *  TENSOR
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_tensor_create(
    EngineTensor** out_t, const int64_t* shape, int32_t ndim, EngineDType dtype)
{
    if (!out_t || !shape) { set_error("null argument"); return ENGINE_ERR_NULL_PTR; }
    *out_t = nullptr;
    return safe_call([&]() -> EngineStatus {
        clear_error();
        auto* t = new (std::nothrow) EngineTensor();
        if (!t) return ENGINE_ERR_OUT_OF_MEMORY;
        t->owned = std::make_unique<engine::Tensor>(
            engine::Tensor::empty(engine::Shape(shape, ndim), to_engine_dtype(dtype)));
        if (t->owned->is_empty() && engine::Shape(shape, ndim).numel() > 0) {
            delete t;
            return ENGINE_ERR_OUT_OF_MEMORY;
        }
        *out_t = t;
        return ENGINE_OK;
    });
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_tensor_create_from_data(
    EngineTensor** out_t, const int64_t* shape, int32_t ndim, EngineDType dtype,
    const void* data, size_t bytes)
{
    if (!out_t || !shape || !data) { set_error("null argument"); return ENGINE_ERR_NULL_PTR; }
    *out_t = nullptr;
    return safe_call([&]() -> EngineStatus {
        clear_error();
        auto* t = new (std::nothrow) EngineTensor();
        if (!t) return ENGINE_ERR_OUT_OF_MEMORY;
        t->owned = std::make_unique<engine::Tensor>(
            engine::Tensor::empty(engine::Shape(shape, ndim), to_engine_dtype(dtype)));
        if (t->owned->is_empty() && engine::Shape(shape, ndim).numel() > 0) {
            delete t;
            return ENGINE_ERR_OUT_OF_MEMORY;
        }
        if (bytes > t->owned->nbytes()) {
            delete t;
            set_error("data too large");
            return ENGINE_ERR_INVALID_ARG;
        }
        std::memcpy(t->owned->data(), data, bytes);
        *out_t = t;
        return ENGINE_OK;
    });
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_tensor_data(
    EngineTensor* t, void** out_data, size_t* out_bytes)
{
    if (!t || t->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!out_data) return ENGINE_ERR_NULL_PTR;
    *out_data = t->ptr()->data();
    if (out_bytes) *out_bytes = t->ptr()->nbytes();
    return ENGINE_OK;
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_tensor_shape(
    const EngineTensor* t, int64_t* out_shape, int32_t* out_ndim)
{
    if (!t || t->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    const auto& s = t->ptr()->shape();
    if (out_shape) {
        for (int32_t i = 0; i < s.rank(); ++i) out_shape[i] = s.dim(i);
    }
    if (out_ndim) *out_ndim = s.rank();
    return ENGINE_OK;
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_tensor_copy_from(
    EngineTensor* t, const void* src, size_t bytes)
{
    if (!t || t->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!src) return ENGINE_ERR_NULL_PTR;
    if (t->is_borrowed()) {
        set_error("cannot write to borrowed tensor");
        return ENGINE_ERR_INVALID_ARG;
    }
    if (bytes > t->ptr()->nbytes()) return ENGINE_ERR_INVALID_ARG;
    std::memcpy(t->ptr()->data(), src, bytes);
    return ENGINE_OK;
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_tensor_copy_to(
    const EngineTensor* t, void* dst, size_t bytes)
{
    if (!t || t->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!dst) return ENGINE_ERR_NULL_PTR;
    if (bytes > t->ptr()->nbytes()) return ENGINE_ERR_INVALID_ARG;
    std::memcpy(dst, t->ptr()->data(), bytes);
    return ENGINE_OK;
}

extern "C" ENGINE_API int64_t ENGINE_CALL engine_tensor_numel(const EngineTensor* t) {
    return (t && t->magic == kMagicTensor) ? t->ptr()->numel() : 0;
}

extern "C" ENGINE_API EngineDType ENGINE_CALL engine_tensor_dtype(const EngineTensor* t) {
    return (t && t->magic == kMagicTensor) ? from_engine_dtype(t->ptr()->dtype()) : ENGINE_F32;
}

extern "C" ENGINE_API int32_t ENGINE_CALL engine_tensor_rank(const EngineTensor* t) {
    return (t && t->magic == kMagicTensor) ? t->ptr()->rank() : 0;
}

extern "C" ENGINE_API void ENGINE_CALL engine_tensor_destroy(EngineTensor* t) {
    if (!t || t->magic != kMagicTensor) return;

    /* Session'ın verdiği non-owning tensor'ları SİLME.
     * (thread_local static wrapper'lar buraya düşer) */
    if (t->is_borrowed()) {
        t->borrowed = nullptr;
        return;
    }

    t->magic = 0;
    delete t;
}

/* ============================================================================
 *  MODEL
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_model_load(
    EngineContext* ctx, const char* path, EngineModel** out_model)
{
    (void)ctx;
    if (!out_model || !path) { set_error("null argument"); return ENGINE_ERR_NULL_PTR; }
    *out_model = nullptr;
    return safe_call([&]() -> EngineStatus {
        clear_error();
        auto* m = new (std::nothrow) EngineModel();
        if (!m) return ENGINE_ERR_OUT_OF_MEMORY;
        m->inner = std::make_unique<engine::Model>();
        if (!engine::loader::load_engine_file(path, *m->inner)) {
            delete m;
            set_error("failed to load model file");
            return ENGINE_ERR_INVALID_FORMAT;
        }
        *out_model = m;
        return ENGINE_OK;
    });
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_model_input_count(const EngineModel* m, int32_t* out_n) {
    if (!m || m->magic != kMagicModel) return ENGINE_ERR_INVALID_HANDLE;
    if (out_n) *out_n = m->inner->num_inputs();
    return ENGINE_OK;
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_model_output_count(const EngineModel* m, int32_t* out_n) {
    if (!m || m->magic != kMagicModel) return ENGINE_ERR_INVALID_HANDLE;
    if (out_n) *out_n = m->inner->num_outputs();
    return ENGINE_OK;
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_model_name(const EngineModel* m, const char** out_name) {
    if (!m || m->magic != kMagicModel) return ENGINE_ERR_INVALID_HANDLE;
    if (!out_name) return ENGINE_ERR_NULL_PTR;
    *out_name = m->inner->name().c_str();
    return ENGINE_OK;
}

extern "C" ENGINE_API void ENGINE_CALL engine_model_destroy(EngineModel* m) {
    if (!m || m->magic != kMagicModel) return;
    m->magic = 0;
    delete m;
}

/* ============================================================================
 *  SESSION
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_session_create(
    EngineContext* ctx, EngineModel* model, EngineSession** out_sess)
{
    (void)ctx;
    if (!out_sess) return ENGINE_ERR_NULL_PTR;
    *out_sess = nullptr;
    if (!model || model->magic != kMagicModel) return ENGINE_ERR_INVALID_HANDLE;
    return safe_call([&]() -> EngineStatus {
        clear_error();
        auto* s = new (std::nothrow) EngineSession();
        if (!s) return ENGINE_ERR_OUT_OF_MEMORY;
        s->inner = std::make_unique<engine::Session>(*model->inner);
        *out_sess = s;
        return ENGINE_OK;
    });
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_session_set_input(
    EngineSession* sess, int32_t index, const EngineTensor* input)
{
    if (!sess || sess->magic != kMagicSession) return ENGINE_ERR_INVALID_HANDLE;
    if (!input || input->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    sess->inner->set_input(index, input->ptr());
    return ENGINE_OK;
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_session_run(EngineSession* sess) {
    if (!sess || sess->magic != kMagicSession) return ENGINE_ERR_INVALID_HANDLE;
    return sess->inner->run() ? ENGINE_OK : ENGINE_ERR_RUNTIME;
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_session_get_output(
    const EngineSession* sess, int32_t index, const EngineTensor** out_t)
{
    if (!out_t) return ENGINE_ERR_NULL_PTR;
    *out_t = nullptr;
    if (!sess || sess->magic != kMagicSession) return ENGINE_ERR_INVALID_HANDLE;

    const engine::Tensor* t = sess->inner->output(index);
    if (!t) { set_error("output not available"); return ENGINE_ERR_INVALID_ARG; }

    /* Non-owning wrapper: Session sahibi, biz sadece gösteriyoruz.
     * Kullanıcı bunu engine_tensor_destroy ile silmemeli — zaten
     * is_borrowed() kontrolü sayesinde silinmez.
     *
     * Uyarı: Aynı thread'de iki output aynı anda tutulursa üstüne yazar.
     * Output'u hemen kopyalayın (numpy() gibi). */
    static thread_local EngineTensor wrapper;
    wrapper.magic    = kMagicTensor;
    wrapper.owned.reset();                      /* sahiplik yok */
    wrapper.borrowed = const_cast<engine::Tensor*>(t);

    *out_t = &wrapper;
    return ENGINE_OK;
}

extern "C" ENGINE_API void ENGINE_CALL engine_session_destroy(EngineSession* sess) {
    if (!sess || sess->magic != kMagicSession) return;
    sess->magic = 0;
    delete sess;
}

/* ============================================================================
 *  KOLAYLIK — tek seferlik
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_run_simple(
    const char* model_path,
    const EngineTensor* const* inputs, int32_t num_inputs,
    EngineTensor** outputs, int32_t num_outputs)
{
    if (!model_path || !inputs || !outputs) return ENGINE_ERR_NULL_PTR;

    return safe_call([&]() -> EngineStatus {
        clear_error();

        engine::Model model;
        if (!engine::loader::load_engine_file(model_path, model)) {
            set_error("failed to load model");
            return ENGINE_ERR_INVALID_FORMAT;
        }

        engine::Session sess(model);
        for (int32_t i = 0; i < num_inputs; ++i) {
            if (!inputs[i]) return ENGINE_ERR_NULL_PTR;
            sess.set_input(i, inputs[i]->ptr());
        }

        if (!sess.run()) {
            set_error("inference failed");
            return ENGINE_ERR_RUNTIME;
        }

        for (int32_t i = 0; i < num_outputs; ++i) {
            const engine::Tensor* t = sess.output(i);
            if (!t) { outputs[i] = nullptr; continue; }
            auto* et = new (std::nothrow) EngineTensor();
            if (!et) return ENGINE_ERR_OUT_OF_MEMORY;
            et->owned = std::make_unique<engine::Tensor>(t->clone());
            outputs[i] = et;
        }
        return ENGINE_OK;
    });
}



/* ============================================================================
 *  MODEL INFO
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_model_input_info(
    const EngineModel* m, int32_t idx,
    const char** out_name,
    int64_t* out_shape, int32_t* out_ndim,
    EngineDType* out_dtype)
{
    if (!m || m->magic != kMagicModel) return ENGINE_ERR_INVALID_HANDLE;
    if (idx < 0 || idx >= m->inner->num_inputs()) {
        set_error("input index out of range");
        return ENGINE_ERR_INVALID_ARG;
    }

    const auto& graph = m->inner->graph();

    /* v2: gerçek metadata varsa kullan */
    if (idx < static_cast<int32_t>(graph.input_infos.size())) {
        const auto& info = graph.input_infos[idx];

        if (out_name) {
            static thread_local std::string name_buf;
            name_buf = info.name.empty()
                ? ("input_" + std::to_string(idx))
                : info.name;
            *out_name = name_buf.c_str();
        }
        if (out_shape) {
            for (int32_t i = 0; i < info.ndim; ++i)
                out_shape[i] = info.shape[i];
        }
        if (out_ndim)  *out_ndim  = info.ndim;
        if (out_dtype) *out_dtype = from_engine_dtype(info.dtype);
        return ENGINE_OK;
    }

    /* v1 fallback: metadata yok, varsayılan */
    if (out_name) {
        static thread_local std::string name_buf;
        name_buf = "input_" + std::to_string(idx);
        *out_name = name_buf.c_str();
    }
    if (out_shape) {
        out_shape[0] = 1;
        out_shape[1] = 3;
        out_shape[2] = 640;
        out_shape[3] = 640;
    }
    if (out_ndim)  *out_ndim  = 4;
    if (out_dtype) *out_dtype = ENGINE_F32;
    return ENGINE_OK;
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_model_output_info(
    const EngineModel* m, int32_t idx,
    const char** out_name,
    int64_t* out_shape, int32_t* out_ndim,
    EngineDType* out_dtype)
{
    if (!m || m->magic != kMagicModel) return ENGINE_ERR_INVALID_HANDLE;
    if (idx < 0 || idx >= m->inner->num_outputs()) {
        set_error("output index out of range");
        return ENGINE_ERR_INVALID_ARG;
    }

    const auto& graph = m->inner->graph();

    /* v2: gerçek metadata */
    if (idx < static_cast<int32_t>(graph.output_infos.size())) {
        const auto& info = graph.output_infos[idx];

        if (out_name) {
            static thread_local std::string name_buf;
            name_buf = info.name.empty()
                ? ("output_" + std::to_string(idx))
                : info.name;
            *out_name = name_buf.c_str();
        }
        if (out_shape) {
            for (int32_t i = 0; i < info.ndim; ++i)
                out_shape[i] = info.shape[i];
        }
        if (out_ndim)  *out_ndim  = info.ndim;
        if (out_dtype) *out_dtype = from_engine_dtype(info.dtype);
        return ENGINE_OK;
    }

    /* v1 fallback */
    if (out_name) {
        static thread_local std::string name_buf;
        name_buf = "output_" + std::to_string(idx);
        *out_name = name_buf.c_str();
    }
    if (out_shape) {
        out_shape[0] = 1;
        out_shape[1] = 84;
        out_shape[2] = 8400;
    }
    if (out_ndim)  *out_ndim  = 3;
    if (out_dtype) *out_dtype = ENGINE_F32;
    return ENGINE_OK;
}

/* ============================================================================
 *  SESSION STATS
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_session_get_stats(
    const EngineSession* sess, EngineSessionStats* out_stats)
{
    if (!sess || sess->magic != kMagicSession) return ENGINE_ERR_INVALID_HANDLE;
    if (!out_stats) return ENGINE_ERR_NULL_PTR;

    const auto& s = sess->inner->stats();
    out_stats->last_run_ms = s.last_run_ms;
    out_stats->runs        = s.runs;
    return ENGINE_OK;
}

/* ============================================================================
 *  PROFILING CALLBACK
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_session_set_profile_callback(
    EngineSession* sess, EngineProfileFn fn, void* user)
{
    if (!sess || sess->magic != kMagicSession) return ENGINE_ERR_INVALID_HANDLE;
    sess->inner->set_profile_callback(
        reinterpret_cast<engine::Session::NodeProfileFn>(fn), user);
    return ENGINE_OK;
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_session_clear_profile_callback(
    EngineSession* sess)
{
    if (!sess || sess->magic != kMagicSession) return ENGINE_ERR_INVALID_HANDLE;
    sess->inner->clear_profile_callback();
    return ENGINE_OK;
}

/* ============================================================================
 *  DEBUG INTERMEDIATE
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_session_debug_intermediate(
    const EngineSession* sess, int32_t idx, const EngineTensor** out_t)
{
    if (!out_t) return ENGINE_ERR_NULL_PTR;
    *out_t = nullptr;
    if (!sess || sess->magic != kMagicSession) return ENGINE_ERR_INVALID_HANDLE;

    const engine::Tensor* t = sess->inner->debug_intermediate(idx);
    if (!t) { set_error("intermediate not available"); return ENGINE_ERR_INVALID_ARG; }

    static thread_local EngineTensor wrapper;
    wrapper.magic    = kMagicTensor;
    wrapper.owned.reset();
    wrapper.borrowed = const_cast<engine::Tensor*>(t);
    *out_t = &wrapper;
    return ENGINE_OK;
}

/* ============================================================================
 *  OPS — doğrudan
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_matmul(
    const EngineTensor* a, const EngineTensor* b, EngineTensor* out)
{
    if (!a || a->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!b || b->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!out || out->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (out->is_borrowed()) {
        set_error("cannot write to borrowed tensor");
        return ENGINE_ERR_INVALID_ARG;
    }

    return safe_call([&]() -> EngineStatus {
        clear_error();
        engine::ops::matmul(*a->ptr(), *b->ptr(), *out->ptr());
        return ENGINE_OK;
    });
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_relu(EngineTensor* t) {
    if (!t || t->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (t->is_borrowed()) return ENGINE_ERR_INVALID_ARG;
    return safe_call([&]() -> EngineStatus {
        clear_error();
        engine::ops::relu_(*t->ptr());
        return ENGINE_OK;
    });
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_silu(EngineTensor* t) {
    if (!t || t->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (t->is_borrowed()) return ENGINE_ERR_INVALID_ARG;
    return safe_call([&]() -> EngineStatus {
        clear_error();
        engine::ops::silu_(*t->ptr());
        return ENGINE_OK;
    });
}

/* ============================================================================
 *  OPS — Binary elementwise (broadcast)
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_add(
    const EngineTensor* a, const EngineTensor* b, EngineTensor** out)
{
    return binary_op_impl(a, b, out, BinOp::Add);
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_sub(
    const EngineTensor* a, const EngineTensor* b, EngineTensor** out)
{
    return binary_op_impl(a, b, out, BinOp::Sub);
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_mul(
    const EngineTensor* a, const EngineTensor* b, EngineTensor** out)
{
    return binary_op_impl(a, b, out, BinOp::Mul);
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_div(
    const EngineTensor* a, const EngineTensor* b, EngineTensor** out)
{
    return binary_op_impl(a, b, out, BinOp::Div);
}

/* ============================================================================
 *  OPS — Unary in-place
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_sigmoid(EngineTensor* t) {
    EngineStatus s = unary_check(t);
    if (s != ENGINE_OK) return s;
    return safe_call([&]() -> EngineStatus {
        clear_error();
        engine::ops::sigmoid_(*t->ptr());
        return ENGINE_OK;
    });
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_tanh(EngineTensor* t) {
    EngineStatus s = unary_check(t);
    if (s != ENGINE_OK) return s;
    return safe_call([&]() -> EngineStatus {
        clear_error();
        engine::ops::tanh_(*t->ptr());
        return ENGINE_OK;
    });
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_gelu(EngineTensor* t) {
    EngineStatus s = unary_check(t);
    if (s != ENGINE_OK) return s;
    return safe_call([&]() -> EngineStatus {
        clear_error();
        engine::ops::gelu_(*t->ptr());
        return ENGINE_OK;
    });
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_clip(
    EngineTensor* t, float lo, float hi)
{
    EngineStatus s = unary_check(t);
    if (s != ENGINE_OK) return s;
    if (lo > hi) { set_error("clip: lo > hi"); return ENGINE_ERR_INVALID_ARG; }

    return safe_call([&]() -> EngineStatus {
        clear_error();
        float* p = t->ptr()->data<float>();
        const int64_t n = t->ptr()->numel();
        const __m256 vlo = _mm256_set1_ps(lo);
        const __m256 vhi = _mm256_set1_ps(hi);
        int64_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 v = _mm256_loadu_ps(p + i);
            v = _mm256_max_ps(v, vlo);
            v = _mm256_min_ps(v, vhi);
            _mm256_storeu_ps(p + i, v);
        }
        for (; i < n; ++i) {
            if (p[i] < lo) p[i] = lo;
            if (p[i] > hi) p[i] = hi;
        }
        return ENGINE_OK;
    });
}

/* ============================================================================
 *  OPS — Softmax (in-place, axis)
 * ========================================================================== */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_softmax(
    EngineTensor* t, int32_t axis)
{
    EngineStatus s = unary_check(t);
    if (s != ENGINE_OK) return s;

    return safe_call([&]() -> EngineStatus {
        clear_error();
        engine::Tensor& ten = *t->ptr();
        const int32_t rank = ten.rank();
        int32_t ax = axis;
        if (ax < 0) ax += rank;
        if (ax < 0 || ax >= rank) {
            set_error("softmax: axis out of range");
            return ENGINE_ERR_INVALID_ARG;
        }

        int64_t outer = 1;
        for (int32_t d = 0; d < ax; ++d) outer *= ten.dim(d);
        const int64_t axis_dim = ten.dim(ax);
        int64_t inner = 1;
        for (int32_t d = ax + 1; d < rank; ++d) inner *= ten.dim(d);

        float* p = ten.data<float>();
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t i = 0; i < inner; ++i) {
                float* base = p + o * axis_dim * inner + i;
                float mx = -1e30f;
                for (int64_t a = 0; a < axis_dim; ++a) {
                    const float v = base[a * inner];
                    if (v > mx) mx = v;
                }
                float sum = 0.0f;
                for (int64_t a = 0; a < axis_dim; ++a) {
                    const float e = std::exp(base[a * inner] - mx);
                    base[a * inner] = e;
                    sum += e;
                }
                const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
                for (int64_t a = 0; a < axis_dim; ++a) {
                    base[a * inner] *= inv;
                }
            }
        }
        return ENGINE_OK;
    });
}


/* ============================================================================
 *  RANDOM — randn / rand
 * ========================================================================== */

namespace {

inline EngineStatus random_impl(
    const int64_t* shape, int32_t rank, uint64_t seed,
    EngineTensor** out, bool normal) noexcept
{
    if (!shape) return ENGINE_ERR_NULL_PTR;
    if (!out)   return ENGINE_ERR_NULL_PTR;
    if (rank < 0 || rank > 8) return ENGINE_ERR_INVALID_SHAPE;
    *out = nullptr;

    return safe_call([&]() -> EngineStatus {
        clear_error();
        engine::Shape os(shape, rank);

        auto* ot = new (std::nothrow) EngineTensor();
        if (!ot) return ENGINE_ERR_OUT_OF_MEMORY;
        ot->owned = std::make_unique<engine::Tensor>(
            engine::Tensor::empty(os, engine::DType::F32));
        if (ot->owned->is_empty() && os.numel() > 0) {
            delete ot;
            return ENGINE_ERR_OUT_OF_MEMORY;
        }

        float* p = ot->owned->data<float>();
        const int64_t n = os.numel();

        std::mt19937_64 rng(seed);
        if (normal) {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (int64_t i = 0; i < n; ++i) p[i] = dist(rng);
        } else {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (int64_t i = 0; i < n; ++i) p[i] = dist(rng);
        }

        *out = ot;
        return ENGINE_OK;
    });
}

} /* anonymous namespace */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_random_randn(
    const int64_t* shape, int32_t rank, uint64_t seed, EngineTensor** out)
{
    return random_impl(shape, rank, seed, out, /*normal=*/true);
}

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_random_rand(
    const int64_t* shape, int32_t rank, uint64_t seed, EngineTensor** out)
{
    return random_impl(shape, rank, seed, out, /*normal=*/false);
}















/* ============================================================================
 *  OPS — PART 2: reduce / argmax / gather / reshape / transpose / concat / stack
 *  Motor içi mantık, saf C++ (AVX2 yok — gerektiğinde eklenir).
 * ========================================================================== */

namespace {

inline EngineStatus reduce_impl(
    const EngineTensor* a, int32_t axis, int32_t keepdims,
    EngineTensor** out, int mode /* 0=max 1=sum 2=mean */) noexcept
{
    if (!a || a->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!out) return ENGINE_ERR_NULL_PTR;
    *out = nullptr;

    return safe_call([&]() -> EngineStatus {
        clear_error();
        const engine::Tensor& in = *a->ptr();
        if (in.dtype() != engine::DType::F32) {
            set_error("reduce: only F32 supported");
            return ENGINE_ERR_INVALID_ARG;
        }

        const int32_t rank = in.rank();
        int32_t ax = axis;
        if (ax < 0) ax += rank;
        if (ax < 0 || ax >= rank) {
            set_error("reduce: axis out of range");
            return ENGINE_ERR_INVALID_ARG;
        }

        int64_t outer = 1;
        for (int32_t d = 0; d < ax; ++d) outer *= in.dim(d);
        const int64_t axis_dim = in.dim(ax);
        int64_t inner = 1;
        for (int32_t d = ax + 1; d < rank; ++d) inner *= in.dim(d);

        int64_t out_dims[8];
        int32_t orank = 0;
        for (int32_t d = 0; d < rank; ++d) {
            if (d == ax) { if (keepdims) out_dims[orank++] = 1; }
            else         { out_dims[orank++] = in.dim(d); }
        }
        if (orank == 0) { out_dims[0] = 1; orank = 1; }
        engine::Shape os(out_dims, orank);

        auto* ot = new (std::nothrow) EngineTensor();
        if (!ot) return ENGINE_ERR_OUT_OF_MEMORY;
        ot->owned = std::make_unique<engine::Tensor>(
            engine::Tensor::empty(os, engine::DType::F32));
        if (ot->owned->is_empty() && os.numel() > 0) {
            delete ot; return ENGINE_ERR_OUT_OF_MEMORY;
        }

        const float* ip = in.data<float>();
        float* op_ = ot->owned->data<float>();

        if (mode == 0) {
            for (int64_t o = 0; o < outer; ++o)
                for (int64_t i = 0; i < inner; ++i) {
                    float mx = -3.4e38f;
                    const float* base = ip + (o * axis_dim) * inner + i;
                    for (int64_t k = 0; k < axis_dim; ++k)
                        if (base[k * inner] > mx) mx = base[k * inner];
                    op_[o * inner + i] = mx;
                }
        } else {
            for (int64_t o = 0; o < outer; ++o)
                for (int64_t i = 0; i < inner; ++i) {
                    float s = 0.0f;
                    const float* base = ip + (o * axis_dim) * inner + i;
                    for (int64_t k = 0; k < axis_dim; ++k) s += base[k * inner];
                    if (mode == 2) s /= (float)axis_dim;
                    op_[o * inner + i] = s;
                }
        }

        *out = ot;
        return ENGINE_OK;
    });
}

inline EngineStatus argmax_impl(
    const EngineTensor* a, int32_t axis, int32_t keepdims,
    EngineTensor** out) noexcept
{
    if (!a || a->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!out) return ENGINE_ERR_NULL_PTR;
    *out = nullptr;

    return safe_call([&]() -> EngineStatus {
        clear_error();
        const engine::Tensor& in = *a->ptr();
        if (in.dtype() != engine::DType::F32) {
            set_error("argmax: only F32 supported");
            return ENGINE_ERR_INVALID_ARG;
        }

        const int32_t rank = in.rank();
        int32_t ax = axis;
        if (ax < 0) ax += rank;
        if (ax < 0 || ax >= rank) {
            set_error("argmax: axis out of range");
            return ENGINE_ERR_INVALID_ARG;
        }

        int64_t outer = 1;
        for (int32_t d = 0; d < ax; ++d) outer *= in.dim(d);
        const int64_t axis_dim = in.dim(ax);
        int64_t inner = 1;
        for (int32_t d = ax + 1; d < rank; ++d) inner *= in.dim(d);

        int64_t out_dims[8];
        int32_t orank = 0;
        for (int32_t d = 0; d < rank; ++d) {
            if (d == ax) { if (keepdims) out_dims[orank++] = 1; }
            else         { out_dims[orank++] = in.dim(d); }
        }
        if (orank == 0) { out_dims[0] = 1; orank = 1; }
        engine::Shape os(out_dims, orank);

        auto* ot = new (std::nothrow) EngineTensor();
        if (!ot) return ENGINE_ERR_OUT_OF_MEMORY;
        ot->owned = std::make_unique<engine::Tensor>(
            engine::Tensor::empty(os, engine::DType::I32));
        if (ot->owned->is_empty() && os.numel() > 0) {
            delete ot; return ENGINE_ERR_OUT_OF_MEMORY;
        }

        const float* ip = in.data<float>();
        int32_t* op_ = ot->owned->data<int32_t>();

        for (int64_t o = 0; o < outer; ++o)
            for (int64_t i = 0; i < inner; ++i) {
                const float* base = ip + (o * axis_dim) * inner + i;
                float best = base[0];
                int32_t bi = 0;
                for (int64_t k = 1; k < axis_dim; ++k)
                    if (base[k * inner] > best) { best = base[k * inner]; bi = (int32_t)k; }
                op_[o * inner + i] = bi;
            }

        *out = ot;
        return ENGINE_OK;
    });
}

inline EngineStatus reshape_impl(
    const EngineTensor* a, const int64_t* shape, int32_t rank,
    EngineTensor** out) noexcept
{
    if (!a || a->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!shape || !out) return ENGINE_ERR_NULL_PTR;
    if (rank < 0 || rank > 8) return ENGINE_ERR_INVALID_SHAPE;
    *out = nullptr;

    return safe_call([&]() -> EngineStatus {
        clear_error();
        const engine::Tensor& in = *a->ptr();
        if (in.dtype() != engine::DType::F32) {
            set_error("reshape: only F32 supported");
            return ENGINE_ERR_INVALID_ARG;
        }

        int64_t dims[8];
        int64_t known = 1;
        int32_t neg_idx = -1;
        for (int32_t i = 0; i < rank; ++i) {
            dims[i] = shape[i];
            if (dims[i] == -1) neg_idx = i;
            else if (dims[i] > 0) known *= dims[i];
        }
        if (neg_idx >= 0)
            dims[neg_idx] = (known > 0) ? (in.numel() / known) : 0;

        int64_t total = 1;
        for (int32_t i = 0; i < rank; ++i) total *= dims[i];
        if (total != in.numel()) {
            set_error("reshape: numel mismatch");
            return ENGINE_ERR_INVALID_SHAPE;
        }

        engine::Shape os(dims, rank);
        auto* ot = new (std::nothrow) EngineTensor();
        if (!ot) return ENGINE_ERR_OUT_OF_MEMORY;
        ot->owned = std::make_unique<engine::Tensor>(
            engine::Tensor::empty(os, engine::DType::F32));
        if (ot->owned->is_empty() && os.numel() > 0) {
            delete ot; return ENGINE_ERR_OUT_OF_MEMORY;
        }
        std::memcpy(ot->owned->data(), in.data(), in.nbytes());
        *out = ot;
        return ENGINE_OK;
    });
}

inline EngineStatus transpose_impl(
    const EngineTensor* a, const int32_t* perm, int32_t rank,
    EngineTensor** out) noexcept
{
    if (!a || a->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!perm || !out) return ENGINE_ERR_NULL_PTR;
    *out = nullptr;

    return safe_call([&]() -> EngineStatus {
        clear_error();
        const engine::Tensor& in = *a->ptr();
        if (in.dtype() != engine::DType::F32) {
            set_error("transpose: only F32 supported");
            return ENGINE_ERR_INVALID_ARG;
        }

        if (rank != in.rank()) {
            set_error("transpose: perm rank mismatch");
            return ENGINE_ERR_INVALID_SHAPE;
        }

        int32_t pv[8];
        for (int32_t i = 0; i < rank; ++i) {
            int32_t p = perm[i];
            if (p < 0) p += rank;
            if (p < 0 || p >= rank) {
                set_error("transpose: invalid perm");
                return ENGINE_ERR_INVALID_ARG;
            }
            pv[i] = p;
        }

        int64_t out_dims[8];
        for (int32_t i = 0; i < rank; ++i) out_dims[i] = in.dim(pv[i]);
        engine::Shape os(out_dims, rank);

        auto* ot = new (std::nothrow) EngineTensor();
        if (!ot) return ENGINE_ERR_OUT_OF_MEMORY;
        ot->owned = std::make_unique<engine::Tensor>(
            engine::Tensor::empty(os, engine::DType::F32));
        if (ot->owned->is_empty() && os.numel() > 0) {
            delete ot; return ENGINE_ERR_OUT_OF_MEMORY;
        }

        const float* ip = in.data<float>();
        float* op_ = ot->owned->data<float>();

        int64_t old_str[8];
        old_str[rank - 1] = 1;
        for (int32_t d = rank - 2; d >= 0; --d)
            old_str[d] = old_str[d + 1] * in.dim(d + 1);

        const int64_t n = ot->owned->numel();
        int64_t o_idx[8];
        for (int64_t i = 0; i < n; ++i) {
            int64_t t = i;
            for (int32_t d = rank - 1; d >= 0; --d) {
                o_idx[d] = t % out_dims[d];
                t /= out_dims[d];
            }
            int64_t src = 0;
            for (int32_t d = 0; d < rank; ++d)
                src += o_idx[d] * old_str[pv[d]];
            op_[i] = ip[src];
        }

        *out = ot;
        return ENGINE_OK;
    });
}

inline EngineStatus concat_impl(
    const EngineTensor* const* inputs, int32_t n, int32_t axis,
    EngineTensor** out) noexcept
{
    if (!inputs || n < 1 || !out) return ENGINE_ERR_NULL_PTR;
    *out = nullptr;

    return safe_call([&]() -> EngineStatus {
        clear_error();
        for (int32_t i = 0; i < n; ++i)
            if (!inputs[i] || inputs[i]->magic != kMagicTensor)
                return ENGINE_ERR_INVALID_HANDLE;

        const engine::Tensor& t0 = *inputs[0]->ptr();
        const int32_t rank = t0.rank();
        int32_t ax = axis;
        if (ax < 0) ax += rank;
        if (ax < 0 || ax >= rank) {
            set_error("concat: axis out of range");
            return ENGINE_ERR_INVALID_ARG;
        }

        int64_t out_dims[8];
        for (int32_t d = 0; d < rank; ++d) out_dims[d] = t0.dim(d);
        out_dims[ax] = 0;
        for (int32_t i = 0; i < n; ++i) {
            const engine::Tensor& ti = *inputs[i]->ptr();
            if (ti.rank() != rank) {
                set_error("concat: rank mismatch");
                return ENGINE_ERR_INVALID_SHAPE;
            }
            for (int32_t d = 0; d < rank; ++d) {
                if (d == ax) continue;
                if (ti.dim(d) != out_dims[d]) {
                    set_error("concat: non-axis dim mismatch");
                    return ENGINE_ERR_INVALID_SHAPE;
                }
            }
            out_dims[ax] += ti.dim(ax);
        }
        engine::Shape os(out_dims, rank);

        auto* ot = new (std::nothrow) EngineTensor();
        if (!ot) return ENGINE_ERR_OUT_OF_MEMORY;
        ot->owned = std::make_unique<engine::Tensor>(
            engine::Tensor::empty(os, engine::DType::F32));
        if (ot->owned->is_empty() && os.numel() > 0) {
            delete ot; return ENGINE_ERR_OUT_OF_MEMORY;
        }

        int64_t outer = 1;
        for (int32_t d = 0; d < ax; ++d) outer *= out_dims[d];
        int64_t inner = 1;
        for (int32_t d = ax + 1; d < rank; ++d) inner *= out_dims[d];

        float* op_ = ot->owned->data<float>();
        for (int64_t o = 0; o < outer; ++o) {
            int64_t dst_off = o * out_dims[ax] * inner;
            for (int32_t i = 0; i < n; ++i) {
                const engine::Tensor& ti = *inputs[i]->ptr();
                const int64_t ax_dim_i = ti.dim(ax);
                const float* src = ti.data<float>() + o * ax_dim_i * inner;
                std::memcpy(op_ + dst_off, src,
                            (size_t)(ax_dim_i * inner * sizeof(float)));
                dst_off += ax_dim_i * inner;
            }
        }

        *out = ot;
        return ENGINE_OK;
    });
}

inline EngineStatus stack_impl(
    const EngineTensor* const* inputs, int32_t n, int32_t axis,
    EngineTensor** out) noexcept
{
    if (!inputs || n < 1 || !out) return ENGINE_ERR_NULL_PTR;
    *out = nullptr;

    return safe_call([&]() -> EngineStatus {
        clear_error();
        for (int32_t i = 0; i < n; ++i)
            if (!inputs[i] || inputs[i]->magic != kMagicTensor)
                return ENGINE_ERR_INVALID_HANDLE;

        const engine::Tensor& t0 = *inputs[0]->ptr();
        const int32_t rank = t0.rank();
        const int32_t new_rank = rank + 1;
        int32_t ax = axis;
        if (ax < 0) ax += new_rank;
        if (ax < 0 || ax >= new_rank) {
            set_error("stack: axis out of range");
            return ENGINE_ERR_INVALID_ARG;
        }

        for (int32_t i = 1; i < n; ++i) {
            if (inputs[i]->ptr()->rank() != rank) {
                set_error("stack: rank mismatch");
                return ENGINE_ERR_INVALID_SHAPE;
            }
            for (int32_t d = 0; d < rank; ++d)
                if (inputs[i]->ptr()->dim(d) != t0.dim(d)) {
                    set_error("stack: shape mismatch");
                    return ENGINE_ERR_INVALID_SHAPE;
                }
        }

        int64_t out_dims[8];
        int32_t j = 0;
        for (int32_t d = 0; d < new_rank; ++d) {
            if (d == ax) out_dims[d] = n;
            else         out_dims[d] = t0.dim(j++);
        }
        engine::Shape os(out_dims, new_rank);

        auto* ot = new (std::nothrow) EngineTensor();
        if (!ot) return ENGINE_ERR_OUT_OF_MEMORY;
        ot->owned = std::make_unique<engine::Tensor>(
            engine::Tensor::empty(os, engine::DType::F32));
        if (ot->owned->is_empty() && os.numel() > 0) {
            delete ot; return ENGINE_ERR_OUT_OF_MEMORY;
        }

        int64_t outer = 1;
        for (int32_t d = 0; d < ax; ++d) outer *= out_dims[d];
        const int64_t inner = (outer > 0) ? (t0.numel() / outer) : t0.numel();

        float* op_ = ot->owned->data<float>();
        for (int64_t o = 0; o < outer; ++o)
            for (int32_t i = 0; i < n; ++i) {
                const float* src = inputs[i]->ptr()->data<float>() + o * inner;
                float* dst = op_ + (o * (int64_t)n + (int64_t)i) * inner;
                std::memcpy(dst, src, (size_t)(inner * sizeof(float)));
            }

        *out = ot;
        return ENGINE_OK;
    });
}

inline EngineStatus gather_impl(
    const EngineTensor* a, const EngineTensor* idx, int32_t axis,
    EngineTensor** out) noexcept
{
    if (!a || a->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!idx || idx->magic != kMagicTensor) return ENGINE_ERR_INVALID_HANDLE;
    if (!out) return ENGINE_ERR_NULL_PTR;
    *out = nullptr;

    return safe_call([&]() -> EngineStatus {
        clear_error();
        const engine::Tensor& data = *a->ptr();
        const engine::Tensor& indices = *idx->ptr();

        if (data.dtype() != engine::DType::F32) {
            set_error("gather: only F32 data supported");
            return ENGINE_ERR_INVALID_ARG;
        }

        const int32_t rank = data.rank();
        int32_t ax = axis;
        if (ax < 0) ax += rank;
        if (ax < 0 || ax >= rank) {
            set_error("gather: axis out of range");
            return ENGINE_ERR_INVALID_ARG;
        }

        const int64_t n_idx = indices.numel();

        int64_t out_dims[8];
        int32_t orank = 0;
        for (int32_t d = 0; d < ax; ++d) out_dims[orank++] = data.dim(d);
        out_dims[orank++] = n_idx;
        for (int32_t d = ax + 1; d < rank; ++d) out_dims[orank++] = data.dim(d);
        engine::Shape os(out_dims, orank);

        auto* ot = new (std::nothrow) EngineTensor();
        if (!ot) return ENGINE_ERR_OUT_OF_MEMORY;
        ot->owned = std::make_unique<engine::Tensor>(
            engine::Tensor::empty(os, engine::DType::F32));
        if (ot->owned->is_empty() && os.numel() > 0) {
            delete ot; return ENGINE_ERR_OUT_OF_MEMORY;
        }

        int64_t outer = 1;
        for (int32_t d = 0; d < ax; ++d) outer *= data.dim(d);
        int64_t inner = 1;
        for (int32_t d = ax + 1; d < rank; ++d) inner *= data.dim(d);
        const int64_t ax_dim = data.dim(ax);

        const float* dp = data.data<float>();
        float* op_ = ot->owned->data<float>();

        auto get_idx = [&](int64_t i) -> int64_t {
            switch (indices.dtype()) {
                case engine::DType::I64: return indices.data<int64_t>()[i];
                case engine::DType::I32: return indices.data<int32_t>()[i];
                default:                 return (int64_t)indices.data<float>()[i];
            }
        };

        for (int64_t o = 0; o < outer; ++o)
            for (int64_t j = 0; j < n_idx; ++j) {
                int64_t gi = get_idx(j);
                if (gi < 0) gi += ax_dim;
                if (gi < 0 || gi >= ax_dim) gi = 0;
                const float* src = dp + (o * ax_dim + gi) * inner;
                float* dst = op_ + (o * n_idx + j) * inner;
                std::memcpy(dst, src, (size_t)(inner * sizeof(float)));
            }

        *out = ot;
        return ENGINE_OK;
    });
}

} /* anonymous namespace */

/* ---- Public wrappers ---- */

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_reduce_max(
    const EngineTensor* a, int32_t axis, int32_t keepdims, EngineTensor** out)
{ return reduce_impl(a, axis, keepdims, out, 0); }

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_reduce_sum(
    const EngineTensor* a, int32_t axis, int32_t keepdims, EngineTensor** out)
{ return reduce_impl(a, axis, keepdims, out, 1); }

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_reduce_mean(
    const EngineTensor* a, int32_t axis, int32_t keepdims, EngineTensor** out)
{ return reduce_impl(a, axis, keepdims, out, 2); }

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_argmax(
    const EngineTensor* a, int32_t axis, int32_t keepdims, EngineTensor** out)
{ return argmax_impl(a, axis, keepdims, out); }

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_gather(
    const EngineTensor* a, const EngineTensor* indices, int32_t axis, EngineTensor** out)
{ return gather_impl(a, indices, axis, out); }

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_reshape(
    const EngineTensor* a, const int64_t* shape, int32_t rank, EngineTensor** out)
{ return reshape_impl(a, shape, rank, out); }

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_transpose(
    const EngineTensor* a, const int32_t* perm, int32_t rank, EngineTensor** out)
{ return transpose_impl(a, perm, rank, out); }

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_concat(
    const EngineTensor* const* inputs, int32_t n, int32_t axis, EngineTensor** out)
{ return concat_impl(inputs, n, axis, out); }

extern "C" ENGINE_API EngineStatus ENGINE_CALL engine_ops_stack(
    const EngineTensor* const* inputs, int32_t n, int32_t axis, EngineTensor** out)
{ return stack_impl(inputs, n, axis, out); }