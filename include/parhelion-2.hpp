#pragma once
#include "image_types.hpp"
#include <cstddef>
#include <span>

/// kernels require nonempty planar views with valid strides and sufficient storage.
/// formats and dimensions must match; enclosing buffer spans must not overlap.
/// all channels, including alpha, are processed; borders replicate the view edges.
/// results saturate to [0, 255].

/// applies a square box blur with radius in [1, 7]; truncates the average.
void box_blur(FrameViewRef src, FrameViewMut dst, std::size_t radius);

/// applies a separable gaussian blur, rounded to nearest; radius 0 copies the input.
/// sigma must be finite and positive.
void gaussian_blur(FrameViewRef src, FrameViewMut dst, float sigma, std::size_t radius);

/// convolves with finite row-major weights, flipped along both axes; rounds to nearest.
/// kernel dimensions must be odd and nonzero, and their product must equal the weight count.
void convolution(
    FrameViewRef src, FrameViewMut dst, std::span<const float> kernel,
    std::size_t kernel_width, std::size_t kernel_height
);

/// applies the cross-shaped sharpening kernel with center 5 and neighbors -1.
void sharpen(FrameViewRef src, FrameViewMut dst);

/// applies the horizontal sobel derivative; negative responses become 0.
void sobel_x(FrameViewRef src, FrameViewMut dst);

/// applies the vertical sobel derivative; negative responses become 0.
void sobel_y(FrameViewRef src, FrameViewMut dst);

/// applies the cross-shaped laplacian with center -4 and neighbors 1; negatives become 0.
void laplacian(FrameViewRef src, FrameViewMut dst);
