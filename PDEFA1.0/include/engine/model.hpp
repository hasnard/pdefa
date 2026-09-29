#ifndef ENGINE_MODEL_HPP
#define ENGINE_MODEL_HPP

#include "engine/graph.hpp"
#include "engine/tensor.hpp"
#include <string>
#include <vector>

namespace engine {

/* Model = ağırlıklar + graph + metadata.
 * Immutable after load — session'lar tarafından paylaşılabilir. */
class Model {
public:
    Model() = default;
    ~Model() = default;

    /* Taşıma serbest */
    Model(Model&&) noexcept = default;
    Model& operator=(Model&&) noexcept = default;

    /* Kopyalama YOK */
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    /* ---- Ağırlıklar ---- */
    [[nodiscard]] size_t num_weights() const noexcept { return weights_.size(); }
    [[nodiscard]] const Tensor& weight(size_t i) const noexcept { return weights_[i]; }
    [[nodiscard]] Tensor& weight(size_t i) noexcept { return weights_[i]; }

    /* ---- Graph ---- */
    [[nodiscard]] const Graph& graph() const noexcept { return graph_; }
    [[nodiscard]] Graph& graph() noexcept { return graph_; }

    /* ---- Metadata ---- */
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    void set_name(std::string n) { name_ = std::move(n); }

    /* ---- Yardımcılar ---- */
    [[nodiscard]] int32_t num_inputs()  const noexcept { return static_cast<int32_t>(graph_.graph_inputs.size()); }
    [[nodiscard]] int32_t num_outputs() const noexcept { return static_cast<int32_t>(graph_.graph_outputs.size()); }

    /* Ağırlık ekle (loader veya kullanıcı) */
    void add_weight(Tensor t) { weights_.push_back(std::move(t)); }

private:
    std::string          name_ = "unnamed";
    std::vector<Tensor>  weights_;
    Graph                graph_;
};

} /* namespace engine */
#endif