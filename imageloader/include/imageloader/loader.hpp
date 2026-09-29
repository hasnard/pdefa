#pragma once
#include "imageloader/image.hpp"
#include <string>

namespace imageloader {

// Magic byte'a göre uygun decoder'a yönlendirir.
// Şu an sadece PNG destekleniyor.
Image load_image(const std::string& path);

} // namespace imageloader