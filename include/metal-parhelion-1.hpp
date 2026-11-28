#pragma once
#include "parhelion-1.hpp"
#include <memory>

/// owns a metal device, command queue, and compiled pointwise pipelines.
class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend struct MetalPointwise;
    friend class MetalFrame;
};

/// owns shared metal storage; cpu views borrow its bytes for the frame's lifetime.
class MetalFrame {
public:
    MetalFrame(MetalContext& context, std::size_t width, std::size_t height,
        PixelFormat format, FrameLayout layout);
    ~MetalFrame();

    MetalFrame(const MetalFrame&) = delete;
    MetalFrame& operator=(const MetalFrame&) = delete;

    /// call only while gpu work on this frame has finished.
    FrameViewRef view_ref() const;
    FrameViewMut view_mut();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend struct MetalPointwise;
};

/// synchronous macos operations with the same view contracts as the cpu kernels.
/// copies host views into shared metal buffers and waits before updating the destination.

void metal_fill(MetalContext& context, FrameViewMut frame, Color::Gray color);

void metal_fill(MetalContext& context, FrameViewMut frame, Color::RGB color);

void metal_fill(MetalContext& context, FrameViewMut frame, Color::RGBA color);

void metal_copy(MetalContext& context, FrameViewRef src, FrameViewMut dst);

void metal_brighten(MetalContext& context, FrameViewMut frame, int amount);

void metal_brighten(MetalContext& context, FrameViewMut frame, int r_amount, int g_amount, int b_amount);

void metal_scale(MetalContext& context, FrameViewMut frame, float alpha);

void metal_scale(MetalContext& context, FrameViewMut frame, float r_alpha, float g_alpha, float b_alpha);

void metal_affine(MetalContext& context, FrameViewMut frame, float alpha, float beta);

void metal_affine(
    MetalContext& context, FrameViewMut frame,
    float r_alpha, float g_alpha, float b_alpha, float r_beta, float g_beta, float b_beta
);

void metal_threshold(MetalContext& context, FrameViewMut frame, std::uint8_t t);

void metal_threshold(
    MetalContext& context, FrameViewMut frame,
    std::uint8_t rt, std::uint8_t gt, std::uint8_t bt
);

void metal_blend(
    MetalContext& context, FrameViewRef src1, FrameViewRef src2, FrameViewMut dst,
    float alpha
);

void metal_blend(
    MetalContext& context, FrameViewRef src1, FrameViewRef src2, FrameViewMut dst,
    float r_alpha, float g_alpha, float b_alpha
);

void metal_grayscale(MetalContext& context, FrameViewRef src, FrameViewMut dst);

void metal_premultiply_alpha(MetalContext& context, FrameViewMut frame);

/// shared-frame overloads dispatch directly and wait before returning.

void metal_fill(MetalContext& context, MetalFrame& frame, Color::Gray color);

void metal_fill(MetalContext& context, MetalFrame& frame, Color::RGB color);

void metal_fill(MetalContext& context, MetalFrame& frame, Color::RGBA color);

void metal_copy(MetalContext& context, const MetalFrame& src, MetalFrame& dst);

void metal_brighten(MetalContext& context, MetalFrame& frame, int amount);

void metal_brighten(MetalContext& context, MetalFrame& frame, int r_amount, int g_amount, int b_amount);

void metal_scale(MetalContext& context, MetalFrame& frame, float alpha);

void metal_scale(MetalContext& context, MetalFrame& frame, float r_alpha, float g_alpha, float b_alpha);

void metal_affine(MetalContext& context, MetalFrame& frame, float alpha, float beta);

void metal_affine(
    MetalContext& context, MetalFrame& frame,
    float r_alpha, float g_alpha, float b_alpha, float r_beta, float g_beta, float b_beta
);

void metal_threshold(MetalContext& context, MetalFrame& frame, std::uint8_t t);

void metal_threshold(
    MetalContext& context, MetalFrame& frame,
    std::uint8_t rt, std::uint8_t gt, std::uint8_t bt
);

void metal_blend(
    MetalContext& context, const MetalFrame& src1, const MetalFrame& src2, MetalFrame& dst,
    float alpha
);

void metal_blend(
    MetalContext& context, const MetalFrame& src1, const MetalFrame& src2, MetalFrame& dst,
    float r_alpha, float g_alpha, float b_alpha
);

void metal_grayscale(MetalContext& context, const MetalFrame& src, MetalFrame& dst);

void metal_premultiply_alpha(MetalContext& context, MetalFrame& frame);
