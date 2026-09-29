#pragma once
#include "imageloader/image.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace imageloader {

// ------------------------------------------------------------------
// Preprocessing ayarlari
// ------------------------------------------------------------------
struct PreprocessConfig {
    int   target_w  = 640;             // model input genisligi
    int   target_h  = 640;             // model input yuksekligi
    float scale     = 1.0f / 255.0f;   // normalize: pixel * scale
    float pad_value = 0.0f;            // letterbox padding degeri
};

// ------------------------------------------------------------------
// Preprocessing sonucu (NCHW float tensor + meta)
// ------------------------------------------------------------------
struct PreprocessResult {
    // NCHW formatinda: [3][H][W] (Direct API kullanildiginda bellek tahsisi yapilmaz, bos kalir)
    std::vector<float> data;

    int   target_w = 0;
    int   target_h = 0;
    int   pad_x    = 0;
    int   pad_y    = 0;
    int   scaled_w = 0;
    int   scaled_h = 0;
    float ratio    = 1.0f;

    bool valid() const { return !data.empty() || target_w > 0; }
};

// ------------------------------------------------------------------
// Ana fonksiyonlar: Letterbox + normalize + NCHW tensor
// - Aspect ratio korur, ortalar
// - Bilinear resize
// - Gray/RGBA -> 3 kanal (RGB)
// - Cikti: [3][H][W] float, NCHW
// ------------------------------------------------------------------

// 1. SIFIR TAHSIS / CACHE-FRIENDLY API:
// Tensor verisini dogrudan modelin onceden ayrilmis input pointer'ina yazar.
// std::vector tahsisi ve cift yazma (double-write) yapmaz, L2/L3 cache'i korur.
void preprocess_letterbox_direct(const Image& img,
                                 const PreprocessConfig& cfg,
                                 float* out_tensor,
                                 PreprocessResult& meta);

// 2. Standart Kopya API (Geriye donuk uyumluluk icin):
// Kendi icinde std::vector tahsis edip dondurur.
PreprocessResult preprocess_letterbox(const Image& img,
                                       const PreprocessConfig& cfg = {});

// ------------------------------------------------------------------
// Yuksek seviye API — inference motorunun cagiracagi fonksiyonlar
// ------------------------------------------------------------------

// Diskten dogrudan model girdi tamponuna tek geciste yazar (En hizli yol)
bool load_and_preprocess_direct(const std::string& path,
                                const PreprocessConfig& cfg,
                                float* out_tensor,
                                PreprocessResult& meta);

// Tek satirda: dosya yolu -> NCHW tensor
PreprocessResult load_and_preprocess(const std::string& path,
                                       const PreprocessConfig& cfg = {});

// Batch: birden fazla dosya -> N tane tensor
std::vector<PreprocessResult> load_and_preprocess_batch(
    const std::vector<std::string>& paths,
    const PreprocessConfig& cfg = {});

} // namespace imageloader