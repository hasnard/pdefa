#include "imageloader/loader.hpp"
#include "imageloader/png_decoder.hpp"
#include "imageloader/jpeg_decoder.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace imageloader {

static bool is_png(const uint8_t* d, size_t n) {
    static const uint8_t SIG[8] = {0x89,'P','N','G',0x0D,0x0A,0x1A,0x0A};
    return n >= 8 && std::memcmp(d, SIG, 8) == 0;
}

static bool is_jpeg(const uint8_t* d, size_t n) {
    return n >= 3 && d[0] == 0xFF && d[1] == 0xD8 && d[2] == 0xFF;
}

// C-File API ile doğrudan diskten tek hamlede (bulk) okuma
static std::vector<uint8_t> read_file_fast(const std::string& path) {
    FILE* fp = nullptr;
#if defined(_MSC_VER)
    if (fopen_s(&fp, path.c_str(), "rb") != 0 || !fp)
        throw std::runtime_error("load_image: dosya acilamadi: " + path);
#else
    fp = std::fopen(path.c_str(), "rb");
    if (!fp)
        throw std::runtime_error("load_image: dosya acilamadi: " + path);
#endif

#if defined(_MSC_VER)
    _fseeki64(fp, 0, SEEK_END);
    int64_t size = _ftelli64(fp);
    _fseeki64(fp, 0, SEEK_SET);
#else
    std::fseek(fp, 0, SEEK_END);
    long size = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
#endif

    if (size < 0) {
        std::fclose(fp);
        throw std::runtime_error("load_image: dosya boyutu okunamadi: " + path);
    }

    std::vector<uint8_t> buf(static_cast<size_t>(size));
    if (size > 0) {
        size_t read_bytes = std::fread(buf.data(), 1, static_cast<size_t>(size), fp);
        if (read_bytes != static_cast<size_t>(size)) {
            std::fclose(fp);
            throw std::runtime_error("load_image: okuma hatasi: " + path);
        }
    }
    std::fclose(fp);
    return buf;
}

Image load_image(const std::string& path) {
    std::vector<uint8_t> buf = read_file_fast(path);
    if (buf.size() < 4) throw std::runtime_error("load_image: dosya cok kucuk");

    if (is_png (buf.data(), buf.size())) return png::decode_png  (buf.data(), buf.size());
    if (is_jpeg(buf.data(), buf.size())) return jpeg::decode_jpeg(buf.data(), buf.size());

    throw std::runtime_error("load_image: desteklenmeyen format: " + path);
}

} // namespace imageloader