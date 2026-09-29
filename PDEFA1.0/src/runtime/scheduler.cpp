/* ============================================================================
 *  src/runtime/scheduler.cpp
 *  Engine-AI — Scheduler implementasyonu
 *
 *  Kritik nokta: seviye bazlı paralellik.
 *  Her seviyedeki tüm node'lar aynı anda çalışabilir.
 *  Bir seviye bitmeden sonrakine geçilmez (basit, doğru, hızlı).
 * ========================================================================== */

#include "engine/runtime/scheduler.hpp"
#include "engine/runtime/thread_pool.hpp"

#include <algorithm>
#include <cstring>
#include <queue>

namespace engine {
namespace runtime {

/* ============================================================================
 *  EXECUTION PLAN — TOPOLOJİK ANALİZ
 * ========================================================================== */

ExecutionPlan ExecutionPlan::build(const Graph& g) noexcept {
    ExecutionPlan p;
    const int32_t n = static_cast<int32_t>(g.nodes.size());
    p.num_nodes = n;
    p.indegree.assign(static_cast<size_t>(n), 0);
    p.dependents.assign(static_cast<size_t>(n), {});

    /* Bağımlılık grafını kur:
     *   node i → node j bağımlılığı ne zaman?
     *   j, i'nin çıktısını input olarak kullanıyorsa
     *
     * Ama bizim node çıktıları intermediate index'lerle ifade ediliyor.
     * Bu yüzden önce intermediate index → üreten node eşlemesi yapalım. */
    std::vector<int32_t> producer(64, -1);   /* intermediate index → node index */
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t out_idx : g.nodes[static_cast<size_t>(i)].outputs) {
            if (out_idx < 0) continue;
            if (static_cast<size_t>(out_idx) >= producer.size()) {
                producer.resize(static_cast<size_t>(out_idx) + 1, -1);
            }
            producer[static_cast<size_t>(out_idx)] = i;
        }
    }

    /* Her node'un input'ları için producer'a bağımlı */
    for (int32_t i = 0; i < n; ++i) {
        const Node& node = g.nodes[static_cast<size_t>(i)];
        for (int32_t in_idx : node.inputs) {
            if (in_idx < 0) continue;   /* external input */
            if (static_cast<size_t>(in_idx) >= producer.size()) continue;
            const int32_t parent = producer[static_cast<size_t>(in_idx)];
            if (parent < 0 || parent == i) continue;
            p.dependents[static_cast<size_t>(parent)].push_back(i);
            p.indegree[static_cast<size_t>(i)]++;
        }
    }

    /* Deduplicate: aynı node iki kez bağımlı olabilir (aynı output'u iki input olarak kullanan) */
    for (int32_t i = 0; i < n; ++i) {
        auto& deps = p.dependents[static_cast<size_t>(i)];
        std::sort(deps.begin(), deps.end());
        deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
    }

    /* Seviye bazlı topolojik sıralama (Kahn's algorithm) */
    std::vector<int32_t> in_deg = p.indegree;
    std::vector<int32_t> current_level;
    for (int32_t i = 0; i < n; ++i) {
        if (in_deg[static_cast<size_t>(i)] == 0) current_level.push_back(i);
    }

    int32_t processed = 0;
    while (!current_level.empty()) {
        p.levels.push_back(current_level);
        if (static_cast<int32_t>(current_level.size()) > p.max_parallel) {
            p.max_parallel = static_cast<int32_t>(current_level.size());
        }
        processed += static_cast<int32_t>(current_level.size());

        std::vector<int32_t> next_level;
        for (int32_t node : current_level) {
            for (int32_t dep : p.dependents[static_cast<size_t>(node)]) {
                if (--in_deg[static_cast<size_t>(dep)] == 0) {
                    next_level.push_back(dep);
                }
            }
        }
        current_level = std::move(next_level);
    }

    p.depth = static_cast<int32_t>(p.levels.size());

    /* Cycle detection */
    if (processed != n) {
        p.levels.clear();
        p.num_nodes = 0;
        p.depth = 0;
        p.max_parallel = 0;
    }

    return p;
}

/* ============================================================================
 *  SCHEDULER
 * ========================================================================== */

Scheduler::Scheduler(const ExecutionPlan& plan) noexcept
    : plan_(plan)
{
    if (plan.num_nodes > 0) {
        remaining_deps_ = std::make_unique<std::atomic<int32_t>[]>(
            static_cast<size_t>(plan.num_nodes));
    }
    reset();
}

void Scheduler::reset() noexcept {
    for (int32_t i = 0; i < plan_.num_nodes; ++i) {
        remaining_deps_[static_cast<size_t>(i)].store(
            plan_.indegree[static_cast<size_t>(i)],
            std::memory_order_relaxed);
    }
    failed_.store(false, std::memory_order_relaxed);
}

bool Scheduler::execute(
    const std::function<bool(int32_t)>& executor) noexcept
{
    if (plan_.num_nodes == 0) return false;

    auto& pool = GlobalPool::get();

    /* Seviye seviye çalıştır */
    for (const auto& level : plan_.levels) {
        if (failed_.load(std::memory_order_acquire)) return false;

        const int32_t level_size = static_cast<int32_t>(level.size());

        if (level_size == 1) {
            /* Tek node — direkt çalıştır */
            if (!executor(level[0])) {
                failed_.store(true, std::memory_order_release);
                return false;
            }
        } else {
            /* Çoklu node — paralel çalıştır */
            std::atomic<int32_t> remaining{level_size};
            std::mutex mtx;
            std::condition_variable cv;

            for (int32_t node : level) {
                pool.submit([&, node]() {
                    bool ok = false;
                    try {
                        ok = executor(node);
                    } catch (...) {
                        ok = false;
                    }
                    if (!ok) {
                        failed_.store(true, std::memory_order_release);
                    }
                    if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> lk(mtx);
                        cv.notify_all();
                    }
                });
            }

            /* Seviye bitene kadar bekle */
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&]() {
                return remaining.load(std::memory_order_acquire) == 0;
            });

            if (failed_.load(std::memory_order_acquire)) return false;
        }
    }

    return !failed_.load(std::memory_order_acquire);
}

} /* namespace runtime */
} /* namespace engine */