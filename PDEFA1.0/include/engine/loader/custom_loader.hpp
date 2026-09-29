#ifndef ENGINE_LOADER_CUSTOM_LOADER_HPP
#define ENGINE_LOADER_CUSTOM_LOADER_HPP

#include "engine/model.hpp"
#include <string>

namespace engine {
namespace loader {

/* Engine-AI native format (.engine) yükleme.
 *
 * Dosya düzeni (little-endian):
 *   [Header   ] magic "ENGN", version u32, num_weights u32, num_nodes u32,
 *               num_inputs u32, num_outputs u32, brand[64]
 *   [Weights  ] her biri: name_len u32, name bytes,
 *                         dtype u8, rank u8, dims i64[rank], offset u64, size u64
 *   [Nodes    ] her biri: op u8, name_len u32, name bytes,
 *                         n_in u16, in[i32], n_out u16, out[i32],
 *                         params (sabit 64 byte blob)
 *   [IO map   ] graph_inputs i32[num_inputs], graph_outputs i32[num_outputs]
 *   [Data blob] ham tensor byte'ları
 */
[[nodiscard]] bool load_engine_file(const std::string& path, Model& out_model) noexcept;

/* Kaydetme (opsiyonel, ileride) */
[[nodiscard]] bool save_engine_file(const std::string& path, const Model& model) noexcept;

/* Sadece ağırlıkları .bin olarak yükle — basit senaryo */
[[nodiscard]] Tensor load_weight_bin(const std::string& path,
                                     const Shape& shape,
                                     DType dtype) noexcept;

} /* namespace loader */
} /* namespace engine */
#endif