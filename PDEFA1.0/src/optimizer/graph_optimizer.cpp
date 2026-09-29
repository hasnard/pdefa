/* ============================================================================
 *  src/optimizer/graph_optimizer.cpp
 *  Engine-AI — Graph optimizer implementasyonu
 * ========================================================================== */

#include "engine/optimizer/graph_optimizer.hpp"
#include "engine/memory/aligned_allocator.hpp"
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace engine {
namespace optimizer {

namespace {




 
    
/* ============================================================================
 *  CONV + BN FUSION
 *
 *  Conv ağırlıklarını BN parametreleriyle ölçekler.
 *  BN node'unu graph'tan kaldırır, Conv'un çıkışını BN çıkışına bağlar.
 * ========================================================================== */
int32_t fuse_conv_bn(Model& m) noexcept {
    Graph& g = m.graph();
    int32_t fused = 0;



    /* Her Conv node'unu bul, çıkışını kullanan BN node'u var mı? */
    for (size_t ci = 0; ci < g.nodes.size(); ++ci) {
        Node& conv = g.nodes[ci];
        if (conv.op != OpType::Conv2d) continue;
        if (conv.outputs.empty()) continue;

        const int32_t conv_out = conv.outputs[0];
        if (conv_out < 0) continue;

        /* BN bul: input'u conv_out olan */
        int32_t bn_idx = -1;
        for (size_t bi = 0; bi < g.nodes.size(); ++bi) {
            const Node& bn = g.nodes[bi];
            if (bn.op != OpType::BatchNorm2d) continue;
            if (bn.inputs.empty()) continue;
            if (bn.inputs[0] == conv_out) {
                bn_idx = static_cast<int32_t>(bi);
                break;
            }
        }
        if (bn_idx < 0) continue;

        Node& bn = g.nodes[static_cast<size_t>(bn_idx)];

        if (conv.inputs.size() < 2) continue;
        const int32_t w_idx = conv.inputs[1];
        
        /* YENİ MİMARİ: Weight index'leri <= -1000 olmalı! */
        if (w_idx > -1000) continue;

        /* BN parametreleri weight olarak saklı: inputs[1..4] */
        if (bn.inputs.size() < 5) continue;
        const int32_t gamma_idx = bn.inputs[1];
        const int32_t beta_idx  = bn.inputs[2];
        const int32_t mean_idx  = bn.inputs[3];
        const int32_t var_idx   = bn.inputs[4];

        /* YENİ MİMARİ: Tüm BN parametreleri de <= -1000 olmalı */
        if (gamma_idx > -1000 || beta_idx > -1000 || mean_idx > -1000 || var_idx > -1000) continue;

        /* Negatif indexleri gerçek RAM indekslerine (size_t) dönüştür */
        const size_t real_w_idx     = static_cast<size_t>(-w_idx - 1000);
        const size_t real_gamma_idx = static_cast<size_t>(-gamma_idx - 1000);
        const size_t real_beta_idx  = static_cast<size_t>(-beta_idx - 1000);
        const size_t real_mean_idx  = static_cast<size_t>(-mean_idx - 1000);
        const size_t real_var_idx   = static_cast<size_t>(-var_idx - 1000);

        /* Ağırlıkları al (referans) */
        Tensor& W = m.weight(real_w_idx);
        const Tensor& gamma = m.weight(real_gamma_idx);
        const Tensor& beta  = m.weight(real_beta_idx);
        const Tensor& mean  = m.weight(real_mean_idx);
        const Tensor& var   = m.weight(real_var_idx);

        if (W.dtype() != DType::F32) continue;
        if (gamma.numel() != W.dim(0)) continue;

        /* Conv bias var mı? Yine Flat Namespace kurallarına göre bakıyoruz */
        int32_t b_idx = (conv.inputs.size() > 2) ? conv.inputs[2] : 0;
        const bool has_bias = (b_idx <= -1000);
        const size_t real_b_idx = has_bias ? static_cast<size_t>(-b_idx - 1000) : 0;

        const float eps = bn.params.eps;
        const int64_t Cout = W.dim(0);
        const int64_t per_channel = W.numel() / Cout;

        const float* gammap = gamma.data<float>();
        const float* betap  = beta.data<float>();
        const float* meanp  = mean.data<float>();
        const float* varp   = var.data<float>();

        float* Wp = W.data<float>();

        /* Bias'ı yeni ağırlığa göre yeniden hesapla */
        std::vector<float> new_bias(static_cast<size_t>(Cout));
        for (int64_t co = 0; co < Cout; ++co) {
            const float scale = gammap[co] / std::sqrt(varp[co] + eps);
            const float b_old = has_bias ? m.weight(real_b_idx).data<float>()[co] : 0.0f;
            
            new_bias[static_cast<size_t>(co)] = (b_old - meanp[co]) * scale + betap[co];

            /* W ağırlıklarını ölçekle */
            float* W_row = Wp + co * per_channel;
            for (int64_t k = 0; k < per_channel; ++k) {
                W_row[k] *= scale;
            }
        }

        /* Yeni bias tensor'ı oluştur ve ekle */
        Tensor new_b_t = Tensor::empty(Shape{static_cast<int64_t>(Cout)}, DType::F32);
        if (new_b_t.is_empty()) continue;
        std::memcpy(new_b_t.data(), new_bias.data(), Cout * sizeof(float));
        m.add_weight(std::move(new_b_t));
        
        /* YENİ MİMARİ: Yeni bias'ın index'ini C++'tan negatif Flat Namespace'e çevir */
        const int32_t new_b_idx = -1000 - static_cast<int32_t>(m.num_weights() - 1);

        /* Conv node'unu güncelle */
        conv.inputs = { conv.inputs[0], w_idx, new_b_idx };
        conv.outputs = bn.outputs;

        /* BN node'unu işaretle — silinecek */
        bn.op = OpType::Unknown;
        bn.inputs.clear();
        bn.outputs.clear();

        fused++;
    }

    /* Unknown node'ları sil */
    if (fused > 0) {
        g.nodes.erase(
            std::remove_if(g.nodes.begin(), g.nodes.end(),
                [](const Node& n) { return n.op == OpType::Unknown; }),
            g.nodes.end());
    }

    return fused;
}

/* ============================================================================
 *  DEAD CODE ELIMINATION
 *
 *  Graph output'larından geriye doğru ulaşılabilen node'ları işaretle.
 *  Ulaşılamayanları sil.
 * ========================================================================== */
int32_t eliminate_dead_code(Model& m) noexcept {
    Graph& g = m.graph();
    const int32_t n = static_cast<int32_t>(g.nodes.size());
    if (n == 0) return 0;

    /* Her intermediate index için üreten node */
    std::unordered_map<int32_t, int32_t> producer;
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t out : g.nodes[static_cast<size_t>(i)].outputs) {
            if (out >= 0) producer[out] = i;
        }
    }

    /* Canlı node'ları geriye doğru işaretle */
    std::vector<bool> live(static_cast<size_t>(n), false);
    std::vector<int32_t> stack;

    /* Graph output'larına bağlı node'ları başlat */
    for (int32_t out_idx : g.graph_outputs) {
        auto it = producer.find(out_idx);
        if (it != producer.end()) stack.push_back(it->second);
    }

    while (!stack.empty()) {
        const int32_t node = stack.back();
        stack.pop_back();
        if (live[static_cast<size_t>(node)]) continue;
        live[static_cast<size_t>(node)] = true;

        for (int32_t in_idx : g.nodes[static_cast<size_t>(node)].inputs) {
            if (in_idx < 0) continue;
            auto it = producer.find(in_idx);
            if (it != producer.end()) stack.push_back(it->second);
        }
    }

    /* Canlı olmayanları sil */
    int32_t removed = 0;
    std::vector<Node> new_nodes;
    new_nodes.reserve(g.nodes.size());
    for (int32_t i = 0; i < n; ++i) {
        if (live[static_cast<size_t>(i)]) {
            new_nodes.push_back(std::move(g.nodes[static_cast<size_t>(i)]));
        } else {
            removed++;
        }
    }
    g.nodes = std::move(new_nodes);
    return removed;
}

/* ============================================================================
 *  CONV + RELU FUSION
 *
 *  Conv → ReLU pattern'ini tek Conv(fused_activation=ReLU) node'una indirir.
 *  Conv'un çıkışını ReLU node'unun çıkışına bağlar, ReLU'yu siler.
 * ========================================================================== */
int32_t fuse_conv_relu(Model& m) noexcept {
    Graph& g = m.graph();
    int32_t fused = 0;

    for (size_t ci = 0; ci < g.nodes.size(); ++ci) {
        Node& conv = g.nodes[ci];
        if (conv.op != OpType::Conv2d) continue;
        if (conv.outputs.empty()) continue;

        const int32_t conv_out = conv.outputs[0];
        if (conv_out < 0) continue;

        /* Conv'un fused_activation'ı zaten ayarlıysa dokunma */
        if (conv.params.fused_activation != 0) continue;

        /* Bu intermediate'i input olarak kullanan ReLU node'u var mı? */
        int32_t relu_idx = -1;
        for (size_t ri = 0; ri < g.nodes.size(); ++ri) {
            const Node& relu = g.nodes[ri];
            if (relu.op != OpType::ReLU) continue;
            if (relu.inputs.empty()) continue;
            if (relu.inputs[0] == conv_out) {
                relu_idx = static_cast<int32_t>(ri);
                break;
            }
        }
        if (relu_idx < 0) continue;

        Node& relu = g.nodes[static_cast<size_t>(relu_idx)];

        /* FUSE:
         *   conv.fused_activation = 1 (ReLU)
         *   conv.outputs = relu.outputs (output index'ini ReLU'dan devral)
         *   relu'yu sil */
        conv.params.fused_activation = 1;   /* ReLU */
        conv.outputs = relu.outputs;

        /* ReLU node'unu iptal et */
        relu.op = OpType::Unknown;
        relu.inputs.clear();
        relu.outputs.clear();
        fused++;
    }

    if (fused > 0) {
        g.nodes.erase(
            std::remove_if(g.nodes.begin(), g.nodes.end(),
                [](const Node& n) { return n.op == OpType::Unknown; }),
            g.nodes.end());
    }

    return fused;
}

} /* anonymous namespace */

/* ============================================================================
 *  ANA GİRİŞ
 * ========================================================================== */

OptimizeReport optimize(Model& model, const OptimizeOptions& opts) noexcept {
    OptimizeReport r;

    /* SIRA ÖNEMLİ:
     *   1) Conv+BN fusion önce (BN parametrelerini Conv'a yedir)
     *   2) Sonra Conv+ReLU fusion (BN zaten gitmişse daha temiz)
     *   3) En son dead code elimination */

    if (opts.fuse_conv_bn) {
        r.conv_bn_fused = fuse_conv_bn(model);
    }

    if (opts.fuse_conv_bn_relu) {
        r.conv_relu_fused = fuse_conv_relu(model);
    }

    if (opts.dead_code_elim) {
        r.nodes_removed += eliminate_dead_code(model);
    }

    return r;
}

} /* namespace optimizer */ 
} /* namespace engine */