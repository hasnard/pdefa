/* ============================================================================
 *  include/engine/engine.hpp
 *  Engine-AI — Ana şemsiye başlık
 *
 *  Kullanıcı tek satırda #include <engine/engine.hpp> diyerek tüm motoru alır.
 * ========================================================================== */

#ifndef ENGINE_ENGINE_HPP
#define ENGINE_ENGINE_HPP

/* ---- Versiyon ---- */
#include "engine/version.hpp"

/* ---- Temel tipler ---- */
#include "engine/dtype.hpp"
#include "engine/shape.hpp"
#include "engine/tensor.hpp"

/* ---- Bellek ---- */
#include "engine/memory/aligned_allocator.hpp"
#include "engine/memory/arena.hpp"

/* ---- Runtime ---- */
#include "engine/runtime/cpu_info.hpp"
#include "engine/runtime/thread_pool.hpp"

/* ---- Ops ---- */
#include "engine/ops/matmul.hpp"
#include "engine/ops/activation.hpp"
#include "engine/ops/conv2d.hpp"
#include "engine/ops/pooling.hpp"
#include "engine/ops/norm.hpp"
#include "engine/ops/elementwise.hpp"
#include "engine/ops/concat.hpp"
#include "engine/ops/upsample.hpp"
#include "engine/ops/softmax.hpp"

/* ---- Optimizer ---- */
#include "engine/optimizer/graph_optimizer.hpp"
#include "engine/optimizer/layout_transform.hpp"
#include "engine/optimizer/quantization.hpp"

/* ---- Model & Session ---- */
#include "engine/graph.hpp"
#include "engine/model.hpp"
#include "engine/session.hpp"
#include "engine/loader/custom_loader.hpp"
namespace engine {

/* Kolaylık: sürüm string'i */
[[nodiscard]] inline const char* version() noexcept {
    return ENGINE_VERSION_STRING;
}

/* Kolaylık: SIMD seviyesi */
[[nodiscard]] inline const char* simd_level() noexcept {
    return runtime::Cpu::level_name(runtime::Cpu::simd_level());
}

} /* namespace engine */

#endif /* ENGINE_ENGINE_HPP */