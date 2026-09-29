#ifndef ENGINE_RUNTIME_SCHEDULER_HPP
#define ENGINE_RUNTIME_SCHEDULER_HPP

#include "engine/graph.hpp"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace engine {
namespace runtime {

struct ExecutionPlan {
    std::vector<int32_t> indegree;
    std::vector<std::vector<int32_t>> dependents;
    std::vector<std::vector<int32_t>> levels;
    int32_t num_nodes    = 0;
    int32_t max_parallel = 0;
    int32_t depth        = 0;

    static ExecutionPlan build(const Graph& g) noexcept;
};

class Scheduler {
public:
    explicit Scheduler(const ExecutionPlan& plan) noexcept;
    ~Scheduler() = default;

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    [[nodiscard]] bool execute(
        const std::function<bool(int32_t)>& executor) noexcept;

    void reset() noexcept;

    [[nodiscard]] const ExecutionPlan& plan() const noexcept { return plan_; }

private:
    const ExecutionPlan& plan_;
    /* atomic'ler unique_ptr dizisinde — vector moveable sorunu yok */
    std::unique_ptr<std::atomic<int32_t>[]> remaining_deps_;
    std::atomic<bool> failed_{false};
};

} /* namespace runtime */
} /* namespace engine */

#endif