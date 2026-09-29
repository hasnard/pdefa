#ifndef ENGINE_RUNTIME_THREAD_POOL_HPP
#define ENGINE_RUNTIME_THREAD_POOL_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>

#include <mutex>
#include <thread>
#include <vector>

namespace engine {
namespace runtime {

class ThreadPool {
public:
    explicit ThreadPool(int32_t n_threads = 0) noexcept;
    ~ThreadPool() noexcept;

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void submit(std::function<void()> task);
    void wait_all();

    [[nodiscard]] int32_t num_threads() const noexcept { return num_workers_; }
    [[nodiscard]] bool in_worker() const noexcept;

    void parallel_for(int64_t begin, int64_t end,
                      const std::function<void(int64_t, int64_t)>& body,
                      int64_t grain_size = 0);

    void parallel_for_2d(int64_t b0, int64_t e0,
                         int64_t b1, int64_t e1,
                         const std::function<void(int64_t, int64_t, int64_t, int64_t)>& body);

private:
    struct Task {
        std::function<void()> fn;
    };

    /* Worker — mutable, movable (std::thread movable) */
    struct alignas(64) Worker {
        std::thread thread;
        int32_t     id = -1;
    };

    /* unique_ptr ile tut — vector reallocation move gerektirmez */
    std::vector<Worker>     workers_;
    int32_t                              num_workers_ = 0;
    std::atomic<bool>                    stop_{false};

    std::mutex              queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<Task>       tasks_;
    size_t                  task_head_ = 0;
    std::atomic<int64_t>    pending_tasks_{0};

    void worker_loop(int32_t worker_id) noexcept;
    bool try_pop(Task& out) noexcept;
    static void set_affinity(int32_t worker_id, int32_t num_workers) noexcept;
};

class GlobalPool {
public:
    [[nodiscard]] static ThreadPool& get() noexcept;
    static void shutdown() noexcept;
    static void set_num_threads(int32_t n) noexcept;
    GlobalPool() = delete;
};

inline void parallel_for(int64_t begin, int64_t end,
                         const std::function<void(int64_t, int64_t)>& body,
                         int64_t grain_size = 0)
{
    GlobalPool::get().parallel_for(begin, end, body, grain_size);
}

} /* namespace runtime */
} /* namespace engine */

#endif