/* ============================================================================
 *  include/engine/optimizer/graph_optimizer.hpp
 *  Engine-AI — Graph iyileştirici
 *
 *  NE YAPAR?
 *    1) Conv+BN+ReLU fusion → tek op
 *    2) Dead code elimination → kullanılmayan node'ları sil
 *    3) Constant folding → derlemede hesaplanabilen sabitleri birleştir
 *    4) Redundant activation kaldırma
 *
 *  Conv+BN FUSION (en kritik):
 *    BN: y = (x - mean) / sqrt(var + eps) * gamma + beta
 *    Conv: y = W*x + b
 *    Birleşik:
 *      scale  = gamma / sqrt(var + eps)
 *      W_new  = W * scale
 *      b_new  = (b - mean) * scale + beta
 *    → BN node'u silinir, Conv'un ağırlıkları güncellenir
 *    %20-30 hız, %10 bellek tasarrufu
 * ========================================================================== */

#ifndef ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_HPP
#define ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_HPP

#include "engine/model.hpp"

namespace engine {
namespace optimizer {

struct OptimizeOptions {
    bool fuse_conv_bn      = true;
    bool fuse_conv_bn_relu = true;   /* Conv+BN+ReLU → Conv(fused) tek op */
    bool dead_code_elim    = true;
    bool constant_folding  = true;
};

struct OptimizeReport {
    int32_t conv_bn_fused    = 0;
    int32_t conv_relu_fused  = 0;
    int32_t nodes_removed    = 0;
    int32_t constants_folded = 0;
};

/* Ana giriş — modeli yerinde optimize eder */
[[nodiscard]] OptimizeReport optimize(Model& model,
                                      const OptimizeOptions& opts = {}) noexcept;

} /* namespace optimizer */
} /* namespace engine */

#endif