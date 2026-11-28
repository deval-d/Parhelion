#pragma once
#include "parhelion-1.hpp"
#include "thread_pool.hpp"

/// these synchronous wrappers use the matching serial kernel's input contract.
/// task counts depend on the kernel and image size; small jobs run on the caller.
/// fill and copy currently run on the caller.
/// the pool must be idle on entry and reserved for this call until it returns.
/// call from outside the pool's workers.

void parallel_fill(ThreadPool& pool, FrameViewMut frame, Color::Gray color);

void parallel_fill(ThreadPool& pool, FrameViewMut frame, Color::RGB color);

void parallel_fill(ThreadPool& pool, FrameViewMut frame, Color::RGBA color);

void parallel_copy(ThreadPool& pool, FrameViewRef src, FrameViewMut dst);

void parallel_brighten(ThreadPool& pool, FrameViewMut frame, int amount);

void parallel_brighten(ThreadPool& pool, FrameViewMut frame, int r_amount, int g_amount, int b_amount);

void parallel_scale(ThreadPool& pool, FrameViewMut frame, float alpha);

void parallel_scale(ThreadPool& pool, FrameViewMut frame, float r_alpha, float g_alpha, float b_alpha);

void parallel_affine(ThreadPool& pool, FrameViewMut frame, float alpha, float beta);

void parallel_affine(
    ThreadPool& pool, FrameViewMut frame,
    float r_alpha, float g_alpha, float b_alpha, float r_beta, float g_beta, float b_beta
);

void parallel_threshold(ThreadPool& pool, FrameViewMut frame, std::uint8_t t);

void parallel_threshold(
    ThreadPool& pool, FrameViewMut frame,
    std::uint8_t rt, std::uint8_t gt, std::uint8_t bt
);

void parallel_blend(
    ThreadPool& pool, FrameViewRef src1, FrameViewRef src2, FrameViewMut dst,
    float alpha
);

void parallel_blend(
    ThreadPool& pool, FrameViewRef src1, FrameViewRef src2, FrameViewMut dst,
    float r_alpha, float g_alpha, float b_alpha
);

void parallel_grayscale(ThreadPool& pool, FrameViewRef src, FrameViewMut dst);

void parallel_premultiply_alpha(ThreadPool& pool, FrameViewMut frame);
