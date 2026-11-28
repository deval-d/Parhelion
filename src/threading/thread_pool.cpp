#include "thread_pool.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

ThreadPool::ThreadPool(std::size_t thread_count) {
    if (thread_count == 0)
        throw std::invalid_argument("thread count must be positive");

    try {
        workers_.reserve(thread_count);

        for (std::size_t i = 0; i < thread_count; ++i) {
            workers_.emplace_back([&] {
                worker_loop();
            });
        }
    } catch (...) {
        // cleanup
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }

        cv_work_.notify_all();
        workers_.clear();
        throw;
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }

    cv_work_.notify_all();

    // jthread destructors join all workers
    workers_.clear();
}

void ThreadPool::submit(Task task) {
    if (!task)
        throw std::invalid_argument("empty task");

    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (stopping_)
            throw std::runtime_error("thread pool is stopping");
 
        if (unfinished_ == std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("too many tasks, holy shit");

        tasks_.push(std::move(task));
        ++unfinished_;
    }

    cv_work_.notify_one();
}

void ThreadPool::worker_loop() {
    while (true) {
        Task task;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            cv_work_.wait(lock, [this] {
                return stopping_ || !tasks_.empty();
            });

            if (stopping_ && tasks_.empty())
                return;

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        std::exception_ptr error;

        try {
            task();
        } catch (...) {
            error = std::current_exception();
        }

        bool done = false;

        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (error && !first_error_)
                first_error_ = error;

            --unfinished_;
            done = unfinished_ == 0;
        }

        if (done)
            cv_done_.notify_all();
    }
}

void ThreadPool::wait() {
    std::exception_ptr error;

    {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_done_.wait(lock, [&] {
            return unfinished_ == 0;
        });

        error = first_error_;
        first_error_ = nullptr;
    }

    if (error)
        std::rethrow_exception(error);
}

