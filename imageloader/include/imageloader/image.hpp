#pragma once
#include <cstdint>
#include <vector>

namespace imageloader {

enum class PixelFormat {
    GRAY8,   // 1 kanal
    RGB8,    // 3 kanal
    RGBA8,   // 4 kanal
    GRAY16   // 1 kanal, 16-bit (rezerve)
};

struct Image {
    uint32_t width  = 0;
    uint32_t height = 0;
    PixelFormat format = PixelFormat::RGB8;
    std::vector<uint8_t> pixels;   // satır-major, sıkı paketli

    size_t channels() const {
        switch (format) {
            case PixelFormat::GRAY8:  return 1;
            case PixelFormat::RGB8:   return 3;
            case PixelFormat::RGBA8:  return 4;
            case PixelFormat::GRAY16: return 1;
        }
        return 0;
    }

    size_t bytes_per_pixel() const {
        switch (format) {
            case PixelFormat::GRAY8:  return 1;
            case PixelFormat::RGB8:   return 3;
            case PixelFormat::RGBA8:  return 4;
            case PixelFormat::GRAY16: return 2;
        }
        return 0;
    }
};

} // namespace imageloader