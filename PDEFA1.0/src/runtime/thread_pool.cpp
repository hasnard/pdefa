#if defined(_MSC_VER)
#  include <windows.h>
#  include <eh.h>
#  include <stdexcept>
#endif


#include "engine/runtime/thread_pool.hpp"
#include "engine/runtime/cpu_info.hpp"
#include "engine/runtime/affinity.hpp"

#include <algorithm>
#include <chrono>

namespace engine {
namespace runtime {

ThreadPool::ThreadPool(int32_t n_threads) noexcept {
    if (n_threads <= 0) {
        n_threads = Cpu::topology().recommended_threads();
        if (n_threads <= 0) n_threads = 1;
    }
    num_workers_ = n_threads;
    workers_.reserve(static_cast<size_t>(num_workers_));

    for (int32_t i = 0; i < num_workers_; ++i) {
        Worker w;
        w.id = i;
        w.thread = std::thread([this, i]() { worker_loop(i); });
        workers_.push_back(std::move(w));
    }
}

ThreadPool::~ThreadPool() noexcept {
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        stop_.store(true, std::memory_order_release);
    }
    queue_cv_.notify_all();
    for (auto& w : workers_) {
        if (w.thread.joinable()) w.thread.join();
    }
}

void ThreadPool::worker_loop(int32_t worker_id) noexcept {
    set_affinity(worker_id, num_workers_);

#if defined(_MSC_VER)
    /* KRİTİK: Windows SEH (access violation, divide-by-zero, vs.)
     * varsayılan olarak C++ `catch(...)` tarafından yakalanmaz ve
     * worker thread'i sessizce öldürür → pending_tasks_ hiç azalmaz →
     * `parallel_for` sonsuza kadar bekler → hang.
     *
     * _set_se_translator her thread için ayrı çağrılmalıdır; SEH'i
     * C++ exception'a çevirir, böylece aşağıdaki try/catch yakalar,
     * RAII guard pending_tasks_'i düşürür ve worker hayatta kalır. */
    _set_se_translator([](unsigned int /*code*/, EXCEPTION_POINTERS* /*ep*/) {
        throw std::runtime_error("SEH exception in worker");
    });
#endif

    while (true) {
        Task task;
        if (!try_pop(task)) {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            queue_cv_.wait(lk, [this]() {
                return stop_.load(std::memory_order_acquire)
                    || task_head_ < tasks_.size();
            });
            if (stop_.load(std::memory_order_acquire)
                && task_head_ >= tasks_.size()) break;
            continue;
        }

        /* RAII guard: task crash etse (C++ ex. veya SEH→C++ çevrilmiş)
         * bile pending_tasks_ azalsın → parallel_for deadlock olmaz. */
        struct PendingGuard {
            std::atomic<int64_t>* p;
            ~PendingGuard() { p->fetch_sub(1, std::memory_order_acq_rel); }
        } guard{&pending_tasks_};

        try {
            task.fn();
        } catch (...) {
            /* Sessizce yut — worker bir sonraki task'a geçsin.
             * Not: SEH fix'i ile AV artık buraya düşer. */
        }
        queue_cv_.notify_all();
    }
}

void ThreadPool::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        tasks_.push_back(Task{std::move(task)});
        pending_tasks_.fetch_add(1, std::memory_order_acq_rel);
    }
    queue_cv_.notify_one();
}

bool ThreadPool::try_pop(Task& out) noexcept {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    if (task_head_ >= tasks_.size()) {
        if (task_head_ > 1024) {
            tasks_.clear();
            task_head_ = 0;
        }
        return false;
    }
    out = std::move(tasks_[task_head_++]);
    if (task_head_ > 1024 && task_head_ * 2 > tasks_.size()) {
        tasks_.erase(tasks_.begin(),
                     tasks_.begin() + static_cast<std::ptrdiff_t>(task_head_));
        task_head_ = 0;
    }
    return true;
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lk(queue_mutex_);
    queue_cv_.wait(lk, [this]() {
        return pending_tasks_.load(std::memory_order_acquire) == 0;
    });
}

bool ThreadPool::in_worker() const noexcept {
    const auto this_id = std::this_thread::get_id();
    for (const auto& w : workers_) {
        if (w.thread.get_id() == this_id) return true;
    }
    return false;
}

void ThreadPool::parallel_for(int64_t begin, int64_t end,
                              const std::function<void(int64_t, int64_t)>& body,
                              int64_t grain_size)
{
    const int64_t total = end - begin;
    if (total <= 0) return;

    if (grain_size <= 0) {
        const int64_t target = static_cast<int64_t>(num_workers_) * 4;
        grain_size = (total + target - 1) / target;
        if (grain_size < 1) grain_size = 1;
    }

    if (total <= grain_size) {
        body(begin, end);
        return;
    }

    const int64_t num_chunks = (total + grain_size - 1) / grain_size;
    std::atomic<int64_t> remaining{num_chunks};
    std::mutex done_mutex;
    std::condition_variable done_cv;

    for (int64_t c = 0; c < num_chunks; ++c) {
        const int64_t cb = begin + c * grain_size;
        const int64_t ce = std::min(cb + grain_size, end);
        submit([&body, &remaining, &done_mutex, &done_cv, cb, ce]() {
            body(cb, ce);
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(done_mutex);
                done_cv.notify_all();
            }
        });
    }

    std::unique_lock<std::mutex> lk(done_mutex);
    done_cv.wait(lk, [&remaining]() {
        return remaining.load(std::memory_order_acquire) == 0;
    });
}

void ThreadPool::parallel_for_2d(int64_t b0, int64_t e0,
                                 int64_t b1, int64_t e1,
                                 const std::function<void(int64_t, int64_t, int64_t, int64_t)>& body)
{
    parallel_for(b0, e0, [&](int64_t i0, int64_t i1) {
        body(i0, i1, b1, e1);
    });
}

void ThreadPool::set_affinity(int32_t worker_id, int32_t num_workers) noexcept {
    (void)worker_id;
    (void)num_workers;
    /* Pinning KAPALI. Windows scheduler daha iyi karar veriyor. */
}

namespace {
std::unique_ptr<ThreadPool> g_global_pool;
std::mutex                  g_global_mutex;
int32_t                     g_configured_threads = 0;
}

ThreadPool& GlobalPool::get() noexcept {
    std::lock_guard<std::mutex> lk(g_global_mutex);
    if (!g_global_pool) {
        g_global_pool = std::make_unique<ThreadPool>(g_configured_threads);
    }
    return *g_global_pool;
}

void GlobalPool::shutdown() noexcept {
    std::lock_guard<std::mutex> lk(g_global_mutex);
    g_global_pool.reset();
}

void GlobalPool::set_num_threads(int32_t n) noexcept {
    std::lock_guard<std::mutex> lk(g_global_mutex);
    if (g_global_pool) g_global_pool.reset();
    g_configured_threads = n;
}

} /* namespace runtime */
} /* namespace engine */