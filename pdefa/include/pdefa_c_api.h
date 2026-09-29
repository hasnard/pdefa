#ifndef PDEFA_C_API_H
#define PDEFA_C_API_H

#include <stdint.h>
#include <stddef.h>
#include "engine/engine_c_api.h"

#if defined(_WIN32)
    #if defined(PDEFA_BUILD)
        #define PDEFA_API __declspec(dllexport)
    #else
        #define PDEFA_API __declspec(dllimport)
    #endif
    #define PDEFA_CALL __cdecl
#else
    #define PDEFA_API __attribute__((visibility("default")))
    #define PDEFA_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============ META ============ */
PDEFA_API const char* PDEFA_CALL pdefa_version(void);
PDEFA_API const char* PDEFA_CALL pdefa_last_error(void);

/* ============ IMAGE ============ */
typedef struct PdefaImage PdefaImage;

typedef enum {
    PDEFA_PIXFMT_GRAY8  = 0,
    PDEFA_PIXFMT_RGB8   = 1,
    PDEFA_PIXFMT_RGBA8  = 2,
    PDEFA_PIXFMT_GRAY16 = 3,
} PdefaPixelFormat;

PDEFA_API PdefaImage* PDEFA_CALL pdefa_image_load(const char* path);
PDEFA_API void        PDEFA_CALL pdefa_image_free(PdefaImage* img);

PDEFA_API int32_t PDEFA_CALL pdefa_image_width   (const PdefaImage* img);
PDEFA_API int32_t PDEFA_CALL pdefa_image_height  (const PdefaImage* img);
PDEFA_API int32_t PDEFA_CALL pdefa_image_channels(const PdefaImage* img);
PDEFA_API int32_t PDEFA_CALL pdefa_image_format  (const PdefaImage* img); /* PdefaPixelFormat */
PDEFA_API const uint8_t* PDEFA_CALL pdefa_image_pixels(const PdefaImage* img,
                                                        size_t* out_bytes);

/* ============ PREPROCESS ============ */
typedef struct {
    int32_t target_w;
    int32_t target_h;
    float   scale;      /* orn. 1/255 */
    float   pad_value;  /* orn. 0 veya 114 */
} PdefaPreprocessConfig;

typedef struct {
    int32_t target_w,  target_h;
    int32_t orig_w,    orig_h;
    int32_t pad_x,     pad_y;
    int32_t scaled_w,  scaled_h;
    float   ratio;
} PdefaPreprocessMeta;

/* Image -> NCHW F32 tensor + meta. out_tensor sahipli. */
PDEFA_API EngineStatus PDEFA_CALL pdefa_preprocess_letterbox(
    const PdefaImage* img,
    const PdefaPreprocessConfig* cfg,
    EngineTensor** out_tensor,
    PdefaPreprocessMeta* out_meta);

/* Tek cagri: dosya -> tensor + meta (en hizli yol) */
PDEFA_API EngineStatus PDEFA_CALL pdefa_prepare_for_engine(
    const char* path,
    int32_t target_w, int32_t target_h,
    float scale, float pad_value,
    EngineTensor** out_tensor,
    PdefaPreprocessMeta* out_meta);

/* Kisayol: sadece tensor (default scale=1/255, pad=0) */
PDEFA_API EngineStatus PDEFA_CALL pdefa_load_image_as_tensor(
    const char* image_path,
    int32_t target_w, int32_t target_h,
    EngineTensor** out_tensor);

/* ============ POST-PROCESS (YOLO) ============ */
/* in_box / out_box: [x1,y1,x2,y2] */
PDEFA_API EngineStatus PDEFA_CALL pdefa_map_box_to_original(
    const float* in_box,
    const PdefaPreprocessMeta* meta,
    float* out_box);

PDEFA_API EngineStatus PDEFA_CALL pdefa_map_point_to_original(
    float* x, float* y,
    const PdefaPreprocessMeta* meta);

#ifdef __cplusplus
}
#endif
#endif