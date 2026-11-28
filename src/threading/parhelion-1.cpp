#include "parallel-parhelion-1.hpp"
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <limits>

namespace {

struct RowPolicy {
    std::size_t samples_per_task;
    std::size_t task_limit;
    std::size_t minimum_tasks = 2;
};

// conservative starting points from the packed rgba8 benchmark on apple m4.
constexpr RowPolicy brighten_policy{4 * 1024 * 1024, 8, 4};
constexpr RowPolicy scale_policy{768 * 1024, 8};
constexpr RowPolicy affine_policy{768 * 1024, 8};
constexpr RowPolicy threshold_policy{1536 * 1024, 8};
constexpr RowPolicy blend_policy{768 * 1024, 8};
constexpr RowPolicy grayscale_policy{256 * 1024, 8};
constexpr RowPolicy premultiply_alpha_policy{1024 * 1024, 4};

template <typename Operation, typename View, typename... Views>
void parallel_rows(ThreadPool& pool, RowPolicy policy, Operation operation, View view, Views... others) {
    const auto height = view.get_height();
    const auto width = view.get_width();
    if (((others.get_height() != height || others.get_width() != width) || ...))
        throw std::invalid_argument("frame dimensions must match");

    const auto channels = std::min<std::size_t>(view.get_n_channels(), 3);
    const auto pixels = width * height;
    const auto limit = std::numeric_limits<std::size_t>::max();
    const auto samples = pixels > limit / channels ? limit : pixels * channels;
    const auto tasks = std::min({pool.worker_count(), height, policy.task_limit,
        samples / policy.samples_per_task});
    if (tasks < policy.minimum_tasks) {
        operation(view, others...);
        return;
    }

    const auto rows = height / tasks;
    const auto remainder = height % tasks;
    std::size_t y = 0;
    try {
        for (std::size_t i = 0; i < tasks; ++i) {
            const auto count = rows + (i < remainder);
            pool.submit([operation, band = view.subrows(y, count),
                         ... bands = others.subrows(y, count)] {
                operation(band, bands...);
            });
            y += count;
        }
    } catch (...) {
        // drain submitted bands before the caller can release their storage.
        const auto error = std::current_exception();
        try {
            pool.wait();
        } catch (...) {
            // preserve the submission error.
        }
        std::rethrow_exception(error);
    }
    pool.wait();
}

} // namespace

void parallel_fill(ThreadPool&, FrameViewMut frame, Color::Gray color) {
    fill(frame, color);
}

void parallel_fill(ThreadPool&, FrameViewMut frame, Color::RGB color) {
    fill(frame, color);
}

void parallel_fill(ThreadPool&, FrameViewMut frame, Color::RGBA color) {
    fill(frame, color);
}

void parallel_copy(ThreadPool&, FrameViewRef src, FrameViewMut dst) {
    copy(src, dst);
}

void parallel_brighten(ThreadPool& pool, FrameViewMut frame, int amount) {
    parallel_rows(pool, brighten_policy, [amount](FrameViewMut frame) {
        brighten(frame, amount);
    }, frame);
}

void parallel_brighten(ThreadPool& pool, FrameViewMut frame, int r_amount, int g_amount, int b_amount) {
    parallel_rows(pool, brighten_policy, [r_amount, g_amount, b_amount](FrameViewMut frame) {
        brighten(frame, r_amount, g_amount, b_amount);
    }, frame);
}

void parallel_scale(ThreadPool& pool, FrameViewMut frame, float alpha) {
    parallel_rows(pool, scale_policy, [alpha](FrameViewMut frame) {
        scale(frame, alpha);
    }, frame);
}

void parallel_scale(ThreadPool& pool, FrameViewMut frame, float r_alpha, float g_alpha, float b_alpha) {
    parallel_rows(pool, scale_policy, [r_alpha, g_alpha, b_alpha](FrameViewMut frame) {
        scale(frame, r_alpha, g_alpha, b_alpha);
    }, frame);
}

void parallel_affine(ThreadPool& pool, FrameViewMut frame, float alpha, float beta) {
    parallel_rows(pool, affine_policy, [alpha, beta](FrameViewMut frame) {
        affine(frame, alpha, beta);
    }, frame);
}

void parallel_affine(
    ThreadPool& pool, FrameViewMut frame,
    float r_alpha, float g_alpha, float b_alpha, float r_beta, float g_beta, float b_beta
) {
    parallel_rows(pool, affine_policy, [r_alpha, g_alpha, b_alpha, r_beta, g_beta, b_beta](FrameViewMut frame) {
        affine(frame, r_alpha, g_alpha, b_alpha, r_beta, g_beta, b_beta);
    }, frame);
}

void parallel_threshold(ThreadPool& pool, FrameViewMut frame, std::uint8_t t) {
    parallel_rows(pool, threshold_policy, [t](FrameViewMut frame) {
        threshold(frame, t);
    }, frame);
}

void parallel_threshold(
    ThreadPool& pool, FrameViewMut frame,
    std::uint8_t rt, std::uint8_t gt, std::uint8_t bt
) {
    parallel_rows(pool, threshold_policy, [rt, gt, bt](FrameViewMut frame) {
        threshold(frame, rt, gt, bt);
    }, frame);
}

void parallel_blend(
    ThreadPool& pool, FrameViewRef src1, FrameViewRef src2, FrameViewMut dst,
    float alpha
) {
    parallel_rows(pool, blend_policy, [alpha](FrameViewRef src1, FrameViewRef src2, FrameViewMut dst) {
        blend(src1, src2, dst, alpha);
    }, src1, src2, dst);
}

void parallel_blend(
    ThreadPool& pool, FrameViewRef src1, FrameViewRef src2, FrameViewMut dst,
    float r_alpha, float g_alpha, float b_alpha
) {
    parallel_rows(pool, blend_policy, [r_alpha, g_alpha, b_alpha](FrameViewRef src1, FrameViewRef src2, FrameViewMut dst) {
        blend(src1, src2, dst, r_alpha, g_alpha, b_alpha);
    }, src1, src2, dst);
}

void parallel_grayscale(ThreadPool& pool, FrameViewRef src, FrameViewMut dst) {
    parallel_rows(pool, grayscale_policy, [](FrameViewRef src, FrameViewMut dst) {
        grayscale(src, dst);
    }, src, dst);
}

void parallel_premultiply_alpha(ThreadPool& pool, FrameViewMut frame) {
    parallel_rows(pool, premultiply_alpha_policy, [](FrameViewMut frame) {
        premultiply_alpha(frame);
    }, frame);
}
