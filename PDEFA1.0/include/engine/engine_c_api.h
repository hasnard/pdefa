/* ============================================================================
 *  include/engine/engine_c_api.h
 *  Engine-AI — Public C API (FFI köprüsü)
 * ========================================================================== */

#ifndef ENGINE_C_API_H
#define ENGINE_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef ENGINE_BUILDING_SHARED
#    define ENGINE_API __declspec(dllexport)
#  elif defined(ENGINE_SHARED)
#    define ENGINE_API __declspec(dllimport)
#  else
#    define ENGINE_API
#  endif
#  define ENGINE_CALL __cdecl
#else
#  if defined(ENGINE_BUILDING_SHARED) && (defined(__GNUC__) || defined(__clang__))
#    define ENGINE_API __attribute__((visibility("default")))
#  else
#    define ENGINE_API
#  endif
#  define ENGINE_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EngineContext EngineContext;
typedef struct EngineModel   EngineModel;
typedef struct EngineSession EngineSession;
typedef struct EngineTensor  EngineTensor;

typedef enum EngineStatus {
    ENGINE_OK                 = 0,
    ENGINE_ERR_NULL_PTR       = 1,
    ENGINE_ERR_INVALID_HANDLE = 2,
    ENGINE_ERR_OUT_OF_MEMORY  = 3,
    ENGINE_ERR_FILE_NOT_FOUND = 4,
    ENGINE_ERR_INVALID_FORMAT = 5,
    ENGINE_ERR_INVALID_SHAPE  = 6,
    ENGINE_ERR_INVALID_ARG    = 7,
    ENGINE_ERR_RUNTIME        = 8,
    ENGINE_ERR_NOT_IMPL       = 9,
    ENGINE_ERR_UNKNOWN        = 99
} EngineStatus;

typedef enum EngineDType {
    ENGINE_F32  = 0,
    ENGINE_F16  = 1,
    ENGINE_BF16 = 2,
    ENGINE_I8   = 3,
    ENGINE_U8   = 4,
    ENGINE_I32  = 5,
    ENGINE_I64  = 6,
    ENGINE_BOOL = 7
} EngineDType;

/* ============ META ============ */
ENGINE_API const char* ENGINE_CALL engine_version(void);
ENGINE_API const char* ENGINE_CALL engine_last_error(void);
ENGINE_API const char* ENGINE_CALL engine_status_string(EngineStatus s);
ENGINE_API size_t      ENGINE_CALL engine_dtype_size(EngineDType dt);
ENGINE_API const char* ENGINE_CALL engine_simd_level(void);
ENGINE_API const char* ENGINE_CALL engine_cpu_brand(void);

/* ============ CONTEXT ============ */
ENGINE_API EngineStatus ENGINE_CALL engine_context_create(EngineContext** out_ctx);
ENGINE_API void         ENGINE_CALL engine_context_destroy(EngineContext* ctx);
ENGINE_API EngineStatus ENGINE_CALL engine_context_set_num_threads(EngineContext* ctx, int32_t n);
ENGINE_API int32_t      ENGINE_CALL engine_context_get_num_threads(const EngineContext* ctx);
ENGINE_API int32_t      ENGINE_CALL engine_context_num_physical_cores(void);
ENGINE_API int32_t      ENGINE_CALL engine_context_has_avx2(void);
ENGINE_API int32_t      ENGINE_CALL engine_context_has_avx512(void);

/* ============ TENSOR ============ */
ENGINE_API EngineStatus ENGINE_CALL engine_tensor_create(
    EngineTensor** out_t, const int64_t* shape, int32_t ndim, EngineDType dtype);
ENGINE_API EngineStatus ENGINE_CALL engine_tensor_create_from_data(
    EngineTensor** out_t, const int64_t* shape, int32_t ndim, EngineDType dtype,
    const void* data, size_t bytes);
ENGINE_API EngineStatus ENGINE_CALL engine_tensor_data(
    EngineTensor* t, void** out_data, size_t* out_bytes);
ENGINE_API EngineStatus ENGINE_CALL engine_tensor_shape(
    const EngineTensor* t, int64_t* out_shape, int32_t* out_ndim);
ENGINE_API EngineStatus ENGINE_CALL engine_tensor_copy_from(
    EngineTensor* t, const void* src, size_t bytes);
ENGINE_API EngineStatus ENGINE_CALL engine_tensor_copy_to(
    const EngineTensor* t, void* dst, size_t bytes);
ENGINE_API int64_t      ENGINE_CALL engine_tensor_numel(const EngineTensor* t);
ENGINE_API EngineDType  ENGINE_CALL engine_tensor_dtype(const EngineTensor* t);
ENGINE_API int32_t      ENGINE_CALL engine_tensor_rank(const EngineTensor* t);
ENGINE_API void         ENGINE_CALL engine_tensor_destroy(EngineTensor* t);

/* ============ MODEL ============ */
ENGINE_API EngineStatus ENGINE_CALL engine_model_load(
    EngineContext* ctx, const char* path, EngineModel** out_model);
ENGINE_API EngineStatus ENGINE_CALL engine_model_input_count(const EngineModel* m, int32_t* out_n);
ENGINE_API EngineStatus ENGINE_CALL engine_model_output_count(const EngineModel* m, int32_t* out_n);
ENGINE_API EngineStatus ENGINE_CALL engine_model_name(const EngineModel* m, const char** out_name);
ENGINE_API void         ENGINE_CALL engine_model_destroy(EngineModel* m);
ENGINE_API EngineStatus ENGINE_CALL engine_model_input_info(
    const EngineModel* m, int32_t idx,
    const char** out_name, int64_t* out_shape, int32_t* out_ndim, EngineDType* out_dtype);
ENGINE_API EngineStatus ENGINE_CALL engine_model_output_info(
    const EngineModel* m, int32_t idx,
    const char** out_name, int64_t* out_shape, int32_t* out_ndim, EngineDType* out_dtype);

/* ============ SESSION ============ */
ENGINE_API EngineStatus ENGINE_CALL engine_session_create(
    EngineContext* ctx, EngineModel* model, EngineSession** out_sess);
ENGINE_API EngineStatus ENGINE_CALL engine_session_set_input(
    EngineSession* sess, int32_t index, const EngineTensor* input);
ENGINE_API EngineStatus ENGINE_CALL engine_session_run(EngineSession* sess);
ENGINE_API EngineStatus ENGINE_CALL engine_session_get_output(
    const EngineSession* sess, int32_t index, const EngineTensor** out_t);
ENGINE_API void         ENGINE_CALL engine_session_destroy(EngineSession* sess);
ENGINE_API EngineStatus ENGINE_CALL engine_session_debug_intermediate(
    const EngineSession* sess, int32_t idx, const EngineTensor** out_t);

typedef struct {
    double   last_run_ms;
    uint64_t runs;
} EngineSessionStats;
ENGINE_API EngineStatus ENGINE_CALL engine_session_get_stats(
    const EngineSession* sess, EngineSessionStats* out_stats);

typedef void (*EngineProfileFn)(const char* node_name, double ms, void* user);
ENGINE_API EngineStatus ENGINE_CALL engine_session_set_profile_callback(
    EngineSession* sess, EngineProfileFn fn, void* user);
ENGINE_API EngineStatus ENGINE_CALL engine_session_clear_profile_callback(
    EngineSession* sess);

ENGINE_API EngineStatus ENGINE_CALL engine_run_simple(
    const char* model_path,
    const EngineTensor* const* inputs, int32_t num_inputs,
    EngineTensor** outputs, int32_t num_outputs);

/* ============================================================================
 *  OPS
 * ========================================================================== */

/* ---- Binary elementwise (broadcast) — yeni tensor döner ---- */
ENGINE_API EngineStatus ENGINE_CALL engine_ops_add(
    const EngineTensor* a, const EngineTensor* b, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_sub(
    const EngineTensor* a, const EngineTensor* b, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_mul(
    const EngineTensor* a, const EngineTensor* b, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_div(
    const EngineTensor* a, const EngineTensor* b, EngineTensor** out);

/* ---- MatMul — mevcut stil (pre-allocated out) ---- */
ENGINE_API EngineStatus ENGINE_CALL engine_ops_matmul(
    const EngineTensor* a, const EngineTensor* b, EngineTensor* out);

/* ---- Unary — IN-PLACE (t değişir) ---- */
ENGINE_API EngineStatus ENGINE_CALL engine_ops_relu(EngineTensor* t);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_silu(EngineTensor* t);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_sigmoid(EngineTensor* t);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_tanh(EngineTensor* t);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_gelu(EngineTensor* t);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_clip(EngineTensor* t, float lo, float hi);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_softmax(EngineTensor* t, int32_t axis);

/* ---- Reductions — yeni tensor döner (PART 2) ---- */
ENGINE_API EngineStatus ENGINE_CALL engine_ops_reduce_max(
    const EngineTensor* a, int32_t axis, int32_t keepdims, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_reduce_sum(
    const EngineTensor* a, int32_t axis, int32_t keepdims, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_reduce_mean(
    const EngineTensor* a, int32_t axis, int32_t keepdims, EngineTensor** out);

/* ---- Index — yeni tensor döner (PART 2) ---- */
ENGINE_API EngineStatus ENGINE_CALL engine_ops_argmax(
    const EngineTensor* a, int32_t axis, int32_t keepdims, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_gather(
    const EngineTensor* a, const EngineTensor* indices, int32_t axis, EngineTensor** out);

/* ---- Shape — yeni tensor döner (PART 2) ---- */
ENGINE_API EngineStatus ENGINE_CALL engine_ops_reshape(
    const EngineTensor* a, const int64_t* shape, int32_t rank, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_transpose(
    const EngineTensor* a, const int32_t* perm, int32_t rank, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_concat(
    const EngineTensor* const* inputs, int32_t n, int32_t axis, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_stack(
    const EngineTensor* const* inputs, int32_t n, int32_t axis, EngineTensor** out);



/* ---- Reductions — yeni tensor döner ---- */
ENGINE_API EngineStatus ENGINE_CALL engine_ops_reduce_max(
    const EngineTensor* a, int32_t axis, int32_t keepdims, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_reduce_sum(
    const EngineTensor* a, int32_t axis, int32_t keepdims, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_reduce_mean(
    const EngineTensor* a, int32_t axis, int32_t keepdims, EngineTensor** out);

/* ---- Index — yeni tensor döner ---- */
ENGINE_API EngineStatus ENGINE_CALL engine_ops_argmax(
    const EngineTensor* a, int32_t axis, int32_t keepdims, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_gather(
    const EngineTensor* a, const EngineTensor* indices, int32_t axis, EngineTensor** out);

/* ---- Shape — yeni tensor döner ---- */
ENGINE_API EngineStatus ENGINE_CALL engine_ops_reshape(
    const EngineTensor* a, const int64_t* shape, int32_t rank, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_transpose(
    const EngineTensor* a, const int32_t* perm, int32_t rank, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_concat(
    const EngineTensor* const* inputs, int32_t n, int32_t axis, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_ops_stack(
    const EngineTensor* const* inputs, int32_t n, int32_t axis, EngineTensor** out);








/* ============================================================================
 *  RANDOM — yeni tensor döner
 * ========================================================================== */
ENGINE_API EngineStatus ENGINE_CALL engine_random_randn(
    const int64_t* shape, int32_t rank, uint64_t seed, EngineTensor** out);
ENGINE_API EngineStatus ENGINE_CALL engine_random_rand(
    const int64_t* shape, int32_t rank, uint64_t seed, EngineTensor** out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ENGINE_C_API_H */