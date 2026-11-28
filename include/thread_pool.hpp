#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
private:
    using Task = std::function<void()>;

    std::queue<Task> tasks_;
    std::mutex mutex_;

    std::condition_variable cv_work_;
    std::condition_variable cv_done_;

    std::size_t unfinished_ = 0;
    bool stopping_ = false;

    std::exception_ptr first_error_;
    std::vector<std::jthread> workers_;

    void worker_loop();

public:
    explicit ThreadPool(std::size_t thread_count);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::size_t worker_count() const noexcept { return workers_.size(); }

    void submit(Task task);
    void wait();
};

