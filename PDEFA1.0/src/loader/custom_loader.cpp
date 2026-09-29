#include "engine/loader/custom_loader.hpp"
#include "engine/memory/aligned_allocator.hpp"
#include "engine/optimizer/layout_transform.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

/* Global pointer-keyed cache'leri temizleyen fonksiyonlar.
 * Yeni bir model yüklendiğinde eski cache entry'leri eski pointer'ları
 * işaret eder ve adres yeniden kullanımı (address reuse) yüzünden yanlış
 * ağırlıklar dönebilir. Bu yüzden her yeni model yüklemesinde temizlenmeli. */
extern "C" void engine_conv2d_winograd_clear_cache() noexcept;
extern "C" void engine_matmul_clear_cache() noexcept;
extern "C" void engine_conv2d_direct_clear_cache() noexcept;
extern "C" void engine_conv2d_gemm_clear_cache() noexcept;
extern "C" void engine_conv2d_pointwise_clear_cache() noexcept;


namespace engine {
namespace loader {

namespace {

constexpr uint32_t kMagic    = 0x4E474E45u;  // "ENGN"
constexpr uint32_t kVersion1 = 1;
constexpr uint32_t kVersion2 = 2;

template <typename T>
bool read_pod(std::FILE* f, T& out) noexcept {
    return std::fread(&out, sizeof(T), 1, f) == 1;
}

bool read_bytes(std::FILE* f, void* dst, size_t n) noexcept {
    return std::fread(dst, 1, n, f) == n;
}

/* v2: tek bir TensorInfo oku */
bool read_tensor_info(std::FILE* f, TensorInfo& out) noexcept {
    uint32_t name_len = 0;
    if (!read_pod(f, name_len) || name_len > 4096) return false;
    if (name_len > 0) {
        out.name.resize(name_len);
        if (!read_bytes(f, out.name.data(), name_len)) return false;
    }

    uint8_t ndim = 0;
    if (!read_pod(f, ndim) || ndim > kMaxNdim) return false;
    out.ndim = static_cast<int32_t>(ndim);
    for (uint8_t d = 0; d < ndim; ++d) {
        if (!read_pod(f, out.shape[d])) return false;
    }

    uint8_t dtype_u8 = 0;
    if (!read_pod(f, dtype_u8)) return false;
    out.dtype = static_cast<DType>(dtype_u8);
    return true;
}

} /* anonymous namespace */

bool load_engine_file(const std::string& path, Model& out_model) noexcept {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    uint32_t magic = 0, version = 0;
    uint32_t num_weights = 0, num_nodes = 0, num_inputs = 0, num_outputs = 0;
    char brand[64] = {0};

    if (!read_pod(f, magic) || !read_pod(f, version)) { std::fclose(f); return false; }
    if (magic != kMagic)                              { std::fclose(f); return false; }
    if (version != kVersion1 && version != kVersion2) { std::fclose(f); return false; }

    if (!read_pod(f, num_weights) || !read_pod(f, num_nodes)) { std::fclose(f); return false; }
    if (!read_pod(f, num_inputs)  || !read_pod(f, num_outputs)) { std::fclose(f); return false; }
    if (!read_bytes(f, brand, sizeof(brand)))                 { std::fclose(f); return false; }

    struct WEntry { std::string name; DType dt; Shape shape; uint64_t offset; uint64_t size; };
    std::vector<WEntry> wentries;
    wentries.reserve(num_weights);

    for (uint32_t i = 0; i < num_weights; ++i) {
        uint32_t name_len = 0;
        if (!read_pod(f, name_len) || name_len > 4096) { std::fclose(f); return false; }
        std::string name(name_len, '\0');
        if (!read_bytes(f, name.data(), name_len))     { std::fclose(f); return false; }

        uint8_t dtype_u8 = 0, rank_u8 = 0;
        if (!read_pod(f, dtype_u8) || !read_pod(f, rank_u8)) { std::fclose(f); return false; }
        if (rank_u8 > kMaxNdim) { std::fclose(f); return false; }

        int64_t dims[kMaxNdim] = {0};
        for (uint8_t d = 0; d < rank_u8; ++d) {
            if (!read_pod(f, dims[d])) { std::fclose(f); return false; }
        }
        uint64_t offset = 0, size = 0;
        if (!read_pod(f, offset) || !read_pod(f, size)) { std::fclose(f); return false; }

        WEntry e;
        e.name   = std::move(name);
        e.dt     = static_cast<DType>(dtype_u8);
        e.shape  = Shape(dims, static_cast<int32_t>(rank_u8));
        e.offset = offset;
        e.size   = size;
        wentries.push_back(std::move(e));
    }

    struct NEntry {
        OpType op;
        std::string name;
        std::vector<int32_t> inputs, outputs;
        OpParams params;
    };
    std::vector<NEntry> nentries;
    nentries.reserve(num_nodes);

    for (uint32_t i = 0; i < num_nodes; ++i) {
        uint8_t op_u8 = 0;
        if (!read_pod(f, op_u8)) { std::fclose(f); return false; }

        uint32_t name_len = 0;
        if (!read_pod(f, name_len) || name_len > 4096) { std::fclose(f); return false; }
        std::string name(name_len, '\0');
        if (!read_bytes(f, name.data(), name_len))     { std::fclose(f); return false; }

        uint16_t n_in = 0, n_out = 0;
        if (!read_pod(f, n_in) || n_in > 64)           { std::fclose(f); return false; }
        if (!read_pod(f, n_out) || n_out > 64)         { std::fclose(f); return false; }

        NEntry e;
        e.op   = static_cast<OpType>(op_u8);
        e.name = std::move(name);
        e.inputs.resize(n_in);
        e.outputs.resize(n_out);
        for (uint16_t j = 0; j < n_in; ++j) {
            if (!read_pod(f, e.inputs[j])) { std::fclose(f); return false; }
        }
        for (uint16_t j = 0; j < n_out; ++j) {
            if (!read_pod(f, e.outputs[j])) { std::fclose(f); return false; }
        }
        if (!read_bytes(f, &e.params, sizeof(OpParams))) { std::fclose(f); return false; }
        nentries.push_back(std::move(e));
    }

    std::vector<int32_t> graph_inputs(num_inputs);
    std::vector<int32_t> graph_outputs(num_outputs);
    for (uint32_t i = 0; i < num_inputs; ++i) {
        if (!read_pod(f, graph_inputs[i])) { std::fclose(f); return false; }
    }
    for (uint32_t i = 0; i < num_outputs; ++i) {
        if (!read_pod(f, graph_outputs[i])) { std::fclose(f); return false; }
    }

    /* v2: input/output infoları */
    std::vector<TensorInfo> input_infos;
    std::vector<TensorInfo> output_infos;

    if (version >= kVersion2) {
        input_infos.resize(num_inputs);
        output_infos.resize(num_outputs);

        for (uint32_t i = 0; i < num_inputs; ++i) {
            if (!read_tensor_info(f, input_infos[i])) { std::fclose(f); return false; }
        }
        for (uint32_t i = 0; i < num_outputs; ++i) {
            if (!read_tensor_info(f, output_infos[i])) { std::fclose(f); return false; }
        }
    }

    const long data_start = std::ftell(f);
    if (data_start < 0) { std::fclose(f); return false; }

    out_model = Model{};
    out_model.set_name(path);

    for (auto& we : wentries) {
        if (std::fseek(f, data_start + static_cast<long>(we.offset), SEEK_SET) != 0) {
            std::fclose(f); return false;
        }
        Tensor t = Tensor::empty(we.shape, we.dt);
        if (t.is_empty()) { std::fclose(f); return false; }
        if (!read_bytes(f, t.data(), t.nbytes())) { std::fclose(f); return false; }
        out_model.add_weight(std::move(t));
    }

    Graph& g = out_model.graph();
    g.nodes.reserve(nentries.size());
    for (auto& ne : nentries) {
        Node n;
        n.op      = ne.op;
        n.name    = std::move(ne.name);
        n.inputs  = std::move(ne.inputs);
        n.outputs = std::move(ne.outputs);
        n.params  = ne.params;
        g.nodes.push_back(std::move(n));
    }
    g.graph_inputs  = std::move(graph_inputs);
    g.graph_outputs = std::move(graph_outputs);
    g.input_infos   = std::move(input_infos);
    g.output_infos  = std::move(output_infos);

    std::fclose(f);

    /* Global pointer-keyed cache'leri temizle — yeni model yüklendiğinde
     * eski pointer'lar yeni ağırlıklarla çakışabilir. Bu temizlik ardışık
     * model yüklemelerinde NaN/Inf ve hang sorunlarını engeller. */
    engine_conv2d_winograd_clear_cache();
    engine_matmul_clear_cache();
    engine_conv2d_direct_clear_cache();
    engine_conv2d_gemm_clear_cache();
    engine_conv2d_pointwise_clear_cache();

    return true;
}

bool save_engine_file(const std::string& path, const Model& model) noexcept {
    (void)path; (void)model;
    return false;
}

Tensor load_weight_bin(const std::string& path,
                       const Shape& shape, DType dtype) noexcept
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return Tensor{};
    Tensor t = Tensor::empty(shape, dtype);
    if (t.is_empty()) { std::fclose(f); return Tensor{}; }
    const size_t got = std::fread(t.data(), 1, t.nbytes(), f);
    std::fclose(f);
    if (got != t.nbytes()) return Tensor{};
    return t;
}

} /* namespace loader */
} /* namespace engine */