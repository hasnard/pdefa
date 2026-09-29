#include "engine/ops/matmul.hpp"
#include "engine/ops/conv2d.hpp"
#include "engine/ops/activation.hpp"
#include "engine/ops/pooling.hpp"
#include "engine/runtime/thread_pool.hpp"
#include "engine/ops/norm.hpp"
#include "engine/ops/concat.hpp"
#include "engine/ops/upsample.hpp"
#include "engine/ops/softmax.hpp"
#include "engine/ops/elementwise.hpp"
#include "engine/ops/topk.hpp"
#include "engine/ops/gather_elements.hpp"
#include "engine/memory/aligned_allocator.hpp"
#include "engine/optimizer/layout_transform.hpp"
#include "engine/tensor.hpp"
#include "engine/session.hpp"
#include <unordered_set>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cmath>

namespace engine {

namespace {
#include "session_parts/session_helpers.inc"

/* ============================================================
 *  RUNTIME DEBUG HOOK — derleme olmadan node debug'ı
 *
 *  Env var'lar:
 *    PDEFA_TRACE       = "*"              # glob pattern (örn. "*attn*", "/model.10/*")
 *    PDEFA_TRACE_LEVEL = "1"|"2"|"3"      # 1=shapes, 2=+first4, 3=dump
 *    PDEFA_TRACE_LOG   = "D:\trace.log"   # default: stderr
 * ============================================================ */
inline bool debug_match_pattern(const std::string& name,
                                 const std::string& pattern) noexcept {
    if (pattern.empty()) return false;
    if (pattern == "*") return true;
    const size_t star = pattern.find('*');
    if (star == std::string::npos) return name == pattern;
    const std::string prefix = pattern.substr(0, star);
    const std::string suffix = pattern.substr(star + 1);
    if (name.size() < prefix.size() + suffix.size()) return false;
    if (name.compare(0, prefix.size(), prefix) != 0) return false;
    if (!suffix.empty() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;
    return true;
}

inline FILE* debug_get_log() noexcept {
    static FILE* g_log  = nullptr;
    static bool  g_init = false;
    if (!g_init) {
        g_init = true;
        const char* path = std::getenv("PDEFA_TRACE_LOG");
        if (path && *path) g_log = std::fopen(path, "w");
        if (!g_log) g_log = stderr;
    }
    return g_log;
}

inline bool debug_enabled(const char* node_name) noexcept {
    static const char* pattern = nullptr;
    static bool inited = false;
    if (!inited) {
        inited  = true;
        pattern = std::getenv("PDEFA_TRACE");
    }
    if (!pattern || !*pattern) return false;
    return debug_match_pattern(node_name, pattern);
}

inline int debug_level() noexcept {
    static int lvl = -1;
    if (lvl < 0) {
        const char* s = std::getenv("PDEFA_TRACE_LEVEL");
        lvl = (s && *s) ? std::atoi(s) : 1;
    }
    return lvl;
}

inline void debug_print_tensor_brief(FILE* log, const Tensor& t) {
    std::fprintf(log, " rank=%d dims=[%lld,%lld,%lld,%lld]",
                 t.rank(),
                 (long long)(t.rank()>0?t.dim(0):0),
                 (long long)(t.rank()>1?t.dim(1):0),
                 (long long)(t.rank()>2?t.dim(2):0),
                 (long long)(t.rank()>3?t.dim(3):0));
    if (debug_level() >= 2 && t.dtype() == DType::F32 && t.numel() > 0) {
        const float* p = t.data<float>();
        std::fprintf(log, " first4=[%.4f,%.4f,%.4f,%.4f]",
                     p[0],
                     t.numel()>1?p[1]:0.0f,
                     t.numel()>2?p[2]:0.0f,
                     t.numel()>3?p[3]:0.0f);
    }
}

} // anonymous namespace

/* ============================================================
 *  Kurucu / Yikici
 * ============================================================ */
Session::Session(const Model& model) noexcept : model_(&model) {
    external_inputs_.resize(static_cast<size_t>(model.num_inputs()), nullptr);
    fuse_conv_activations_();

    const Graph& g = model.graph();
    int32_t max_out_idx = -1;
    for (const Node& n : g.nodes) {
        for (int32_t o : n.outputs) {
            if (o > max_out_idx) max_out_idx = o;
        }
    }

    if (max_out_idx >= 0) {
        const size_t req_size = static_cast<size_t>(max_out_idx) + 1;
        intermediate_.resize(req_size);
        base_ref_counts_.assign(req_size, 0);
        tensor_sizes_.assign(req_size, 0);
        current_ref_counts_.resize(req_size);

        for (const Node& n : g.nodes) {
            for (int32_t in_idx : n.inputs) {
                if (in_idx >= 0 && in_idx < static_cast<int32_t>(req_size)) {
                    base_ref_counts_[static_cast<size_t>(in_idx)]++;
                }
            }
        }
        for (int32_t out_idx : g.graph_outputs) {
            if (out_idx >= 0 && out_idx < static_cast<int32_t>(req_size)) {
                base_ref_counts_[static_cast<size_t>(out_idx)]++;
            }
        }
    }
}

Session::~Session() noexcept {
    if (plan_runner_) {
        plan_runner_->reset();
    }
    for (void* raw : raw_allocated_blocks_) {
        if (raw) memory::AlignedAllocator::deallocate(raw);
    }
    raw_allocated_blocks_.clear();
}

int32_t Session::num_inputs()  const noexcept { return model_->num_inputs(); }
int32_t Session::num_outputs() const noexcept { return model_->num_outputs(); }

void Session::set_input(int32_t idx, const Tensor* t) noexcept {
    if (idx < 0 || idx >= static_cast<int32_t>(external_inputs_.size())) return;
    external_inputs_[static_cast<size_t>(idx)] = t;
}

const Tensor* Session::output(int32_t idx) const noexcept {
    if (idx < 0 || idx >= static_cast<int32_t>(outputs_.size())) return nullptr;
    return &outputs_[static_cast<size_t>(idx)];
}

/* ============================================================
 *  Conv + Aktivasyon Fusion
 * ============================================================ */
#include "session_parts/session_fusion.inc"

/* ============================================================
 *  O(1) Bellek Havuzu ve Statik Plan
 * ============================================================ */
#include "session_parts/session_memory.inc"

/* ============================================================
 *  RUN — NHWC iç mimari
 * ============================================================ */
bool Session::run() noexcept {
    if (!model_) return false;

    const auto t_start = std::chrono::high_resolution_clock::now();

    outputs_.clear();
    reset_pool_states_();

    /* Önceki run'dan kalan intermediate tensor'leri temizle. */
    for (auto& t : intermediate_) {
        t = Tensor{};
    }

    if (!base_ref_counts_.empty()) {
        std::memcpy(current_ref_counts_.data(), base_ref_counts_.data(),
                    base_ref_counts_.size() * sizeof(int32_t));
    }

    /* NHWC giriş dönüşümü */
    std::vector<Tensor> owned_inputs_;
    owned_inputs_.reserve(external_inputs_.size());
    std::vector<const Tensor*> nhwc_inputs(external_inputs_.size(), nullptr);

    for (size_t i = 0; i < external_inputs_.size(); ++i) {
        const Tensor* t = external_inputs_[i];
        if (t && !t->is_empty() && t->rank() == 4 && t->dtype() == DType::F32) {
            Tensor nhwc = optimizer::nchw_to_nhwc(*t);
            if (!nhwc.is_empty()) {
                owned_inputs_.push_back(std::move(nhwc));
                nhwc_inputs[i] = &owned_inputs_.back();
                continue;
            }
        }
        nhwc_inputs[i] = t;
    }

    auto saved_ext_inputs = external_inputs_;
    external_inputs_ = std::move(nhwc_inputs);

    const Graph& g = model_->graph();

#ifdef ENGINE_PROFILING
    FILE* csv = get_profile_csv();
    int64_t node_counter = 0;
#endif

    for (const Node& node : g.nodes) {
        current_node_         = &node;
        current_node_out_idx_ = 0;

#ifdef ENGINE_PROFILING
        const auto t0 = std::chrono::high_resolution_clock::now();
        const bool ok = run_node(node);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (csv) {
            char dims_buf[96] = "";
            if (!node.outputs.empty() && node.outputs[0] >= 0) {
                const size_t oi = static_cast<size_t>(node.outputs[0]);
                if (oi < intermediate_.size() && !intermediate_[oi].is_empty()) {
                    const Shape& s = intermediate_[oi].shape();
                    int pos = 0;
                    pos += std::snprintf(dims_buf + pos, sizeof(dims_buf) - (size_t)pos, "[");
                    for (int32_t d = 0; d < s.rank(); ++d) {
                        pos += std::snprintf(dims_buf + pos, sizeof(dims_buf) - (size_t)pos,
                                             "%s%lld", (d ? ";" : ""), (long long)s.dim(d));
                    }
                    std::snprintf(dims_buf + pos, sizeof(dims_buf) - (size_t)pos, "]");
                }
            }
            std::fprintf(csv, "%lld,%s,\"%s\",%.6f\n",
                         (long long)node_counter, op_type_name(node.op), dims_buf, ms);
        }

        if (profile_fn_) {
            char buf[256];
            int pos = std::snprintf(buf, sizeof(buf), "node_%lld_op%d '%s' ",
                                 (long long)node_counter, (int)node.op,
                        node.name.c_str());
            if (!node.outputs.empty() && node.outputs[0] >= 0) {
                const size_t oi = static_cast<size_t>(node.outputs[0]);
                if (oi < intermediate_.size() && !intermediate_[oi].is_empty()) {
                    const Shape& s = intermediate_[oi].shape();
                    pos += std::snprintf(buf + pos, sizeof(buf) - (size_t)pos, "[");
                    for (int32_t d = 0; d < s.rank() && pos < 110; ++d) {
                        pos += std::snprintf(buf + pos, sizeof(buf) - (size_t)pos,
                                             "%s%lld", (d ? "," : ""), (long long)s.dim(d));
                    }
                    if (pos < 118) std::snprintf(buf + pos, sizeof(buf) - (size_t)pos, "]");
                }
            }
            profile_fn_(buf, ms, profile_user_);
        }
#else
        const bool ok = run_node(node);
#endif

        if (!ok) {
#ifdef ENGINE_PROFILING
            std::fprintf(stderr, "[FAIL] node %lld op=%d name='%s'\n",
                         (long long)node_counter, (int)node.op, node.name.c_str());
#else
            std::fprintf(stderr, "[FAIL] op=%d name='%s'\n",
                         (int)node.op, node.name.c_str());
#endif
            external_inputs_ = std::move(saved_ext_inputs);
            current_node_ = nullptr;
            return false;
        }

        /* ============================================================
         *  RUNTIME DEBUG HOOK — env var kontrollü, derleme olmadan
         * ============================================================ */
        if (debug_enabled(node.name.c_str())) {
            FILE* log = debug_get_log();
            std::fprintf(log, "\n[%s] op=%d\n",
                         node.name.c_str(), (int)node.op);

            /* Input shapes + first4 (level >= 1) */
            if (debug_level() >= 1) {
                for (size_t i = 0; i < node.inputs.size(); ++i) {
                    const Tensor* t = resolve_tensor(*model_, node.inputs[i],
                                                     external_inputs_, intermediate_);
                    if (t && !t->is_empty()) {
                        std::fprintf(log, "  in[%zu]", i);
                        debug_print_tensor_brief(log, *t);
                        std::fprintf(log, "\n");
                    }
                }
            }

            /* Output shape + first4 */
            if (!node.outputs.empty() && node.outputs[0] >= 0) {
                const size_t oi = static_cast<size_t>(node.outputs[0]);
                if (oi < intermediate_.size() && !intermediate_[oi].is_empty()) {
                    std::fprintf(log, "  out");
                    debug_print_tensor_brief(log, intermediate_[oi]);
                    std::fprintf(log, "\n");
                }
            }
            std::fflush(log);
        }

#ifdef ENGINE_PROFILING
        ++node_counter;
#endif

        for (int32_t in_idx : node.inputs) {
            if (in_idx >= 0 && in_idx < static_cast<int32_t>(current_ref_counts_.size())) {
                const size_t u_idx = static_cast<size_t>(in_idx);
                current_ref_counts_[u_idx]--;
                if (current_ref_counts_[u_idx] == 0) {
                    free_to_pool(const_cast<void*>(intermediate_[u_idx].data()));
                    intermediate_[u_idx] = Tensor{};
                }
            }
        }
    }

    current_node_ = nullptr;

#ifdef ENGINE_PROFILING
    if (csv) std::fflush(csv);
#endif

    for (int32_t out_i : g.graph_outputs) {
        if (out_i >= 0 && out_i < static_cast<int32_t>(intermediate_.size())) {
            outputs_.push_back(std::move(intermediate_[static_cast<size_t>(out_i)]));
        }
    }

    /* NHWC çıkış dönüşümü */
    for (auto& out_t : outputs_) {
        if (!out_t.is_empty() && out_t.rank() == 4 && out_t.dtype() == DType::F32) {
            Tensor nchw = optimizer::nhwc_to_nchw(out_t);
            if (!nchw.is_empty()) {
                out_t = std::move(nchw);
            }
        }
    }

    external_inputs_ = std::move(saved_ext_inputs);

    if (!has_static_plan_ && !tensor_sizes_.empty()) {
        init_static_memory_plan_();
    }
    const auto t_end = std::chrono::high_resolution_clock::now();
    stats_.last_run_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    stats_.runs++;
    return true;
}

/* ============================================================
 *  RUN_NODE
 * ============================================================ */
bool Session::run_node(const Node& node) noexcept {
    if (node.op == OpType::Input || node.op == OpType::Output) {
        return true;
    }
    const Model& m = *model_;

    constexpr size_t kStackIns = 8;
    const Tensor* ins_stack[kStackIns];
    std::vector<const Tensor*> ins_heap;
    const Tensor** ins_ptr;

    const size_t n_ins = node.inputs.size();
    if (n_ins <= kStackIns) {
        ins_ptr = ins_stack;
    } else {
        ins_heap.resize(n_ins);
        ins_ptr = ins_heap.data();
    }

    for (size_t i = 0; i < n_ins; ++i) {
        int32_t in_idx = node.inputs[i];

        /* WORKAROUND: converter self-reference */
        if (!node.outputs.empty() && in_idx == node.outputs[0] && in_idx > 0) {
            in_idx = in_idx - 1;
        }

        const Tensor* t = resolve_tensor(m, in_idx, external_inputs_, intermediate_);
        if (!t) {
            std::fprintf(stderr, "[MISS] '%s' op=%d in[%zu] idx=%d\n",
                         node.name.c_str(), (int)node.op, i, in_idx);
            return false;
        }
        ins_ptr[i] = t;
    }

    switch (node.op) {

#include "session_parts/session_ops_nn.inc"
#include "session_parts/session_ops_math.inc"
#include "session_parts/session_ops_shape.inc"

    case OpType::Input:
    case OpType::Output:
        return true;
    default:
        return false;
    }
}

} /* namespace engine */