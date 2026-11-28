#pragma once
#include "color_types.hpp"
#include "image_types.hpp"
#include <array>
#include <cstdint>
#include <utility>

/// kernels require nonempty views with valid strides and sufficient storage.
/// views must be planar unless stated otherwise; results saturate to [0, 255].

/// fills a single channel with a constant value; either layout is supported.
void fill(FrameViewMut frame, Color::Gray color);

/// fills the color channels of an rgb or rgba view; preserves alpha.
void fill(FrameViewMut frame, Color::RGB color);

/// fills all channels of an rgba view.
void fill(FrameViewMut frame, Color::RGBA color);

/// copies matching formats, layouts and dimensions; supports either layout.
/// source and destination must not overlap.
void copy(FrameViewRef src, FrameViewMut dst);

/// adds an offset in [-255, 255] to each color channel; preserves alpha.
void brighten(FrameViewMut frame, int amount);

/// adds separate offsets in [-255, 255] to rgb channels; preserves alpha.
void brighten(FrameViewMut frame, int r_amount, int g_amount, int b_amount);

/// multiplies color channels by a finite, nonnegative scale; truncates and preserves alpha.
void scale(FrameViewMut frame, float alpha);

/// scales rgb channels separately; scales must be finite and nonnegative; preserves alpha.
void scale(FrameViewMut frame, float r_alpha, float g_alpha, float b_alpha);

/// applies alpha * value + beta to color channels; truncates and preserves alpha.
/// alpha must be finite and nonnegative; beta must be finite.
void affine(FrameViewMut frame, float alpha, float beta);

/// applies separate scales and offsets to rgb channels; truncates and preserves alpha.
/// scales must be finite and nonnegative; offsets must be finite.
void affine(
    FrameViewMut frame,
    float r_alpha, float g_alpha, float b_alpha,
    float r_beta, float g_beta, float b_beta
);

/// sets color values above t to 255 and the rest to 0; preserves alpha.
void threshold(FrameViewMut frame, std::uint8_t t);

/// thresholds rgb channels separately; preserves alpha.
void threshold(FrameViewMut frame, std::uint8_t rt, std::uint8_t gt, std::uint8_t bt);

/// blends color channels as alpha * src1 + (1 - alpha) * src2; truncates.
/// formats and dimensions must match; alpha must be finite and in [0, 1].
/// preserves destination alpha; exact in-place output is allowed, partial overlap is not.
void blend(FrameViewRef src1, FrameViewRef src2, FrameViewMut dst, float alpha);

/// blends rgb channels with separate finite weights in [0, 1]; truncates.
/// formats and dimensions must match; preserves destination alpha.
/// exact in-place output is allowed, partial overlap is not.
void blend(
    FrameViewRef src1, FrameViewRef src2, FrameViewMut dst,
    float r_alpha, float g_alpha, float b_alpha
);

/// converts rgb or rgba to gray using weights 0.2126, 0.7152 and 0.0722; truncates.
/// dimensions must match; the gray destination may use either layout; views must not overlap.
void grayscale(FrameViewRef src, FrameViewMut dst);

/// multiplies rgba color channels by alpha / 255, rounded to nearest; preserves alpha.
void premultiply_alpha(FrameViewMut frame);

/// returns the minimum and maximum across color channels; ignores alpha.
std::pair<std::uint8_t, std::uint8_t> minmax_global(FrameViewRef frame);

/// returns rgb minima and maxima for an rgb or rgba view; ignores alpha.
std::pair<std::array<std::uint8_t, 3>, std::array<std::uint8_t, 3>>
minmax_per_channel(FrameViewRef frame);

/// counts each byte value in a gray view.
std::array<std::uint64_t, 256> histogram_gray(FrameViewRef frame);

/// counts each byte value separately in the channels of an rgb view.
std::array<std::array<std::uint64_t, 256>, 3> histogram_rgb(FrameViewRef frame);

/// counts each byte value separately in all channels of an rgba view.
std::array<std::array<std::uint64_t, 256>, 4> histogram_rgba(FrameViewRef frame);
