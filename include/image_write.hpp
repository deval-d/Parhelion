#pragma once
#include "image_types.hpp"
#include <stb_image_write.h>
#include <limits>
#include <stdexcept>

inline void image_write(const char* path, const Frame& frame) {
    if (path == nullptr)
        throw std::invalid_argument("image path is null");

    if (frame.get_frame_layout() != Interleaved)
        throw std::invalid_argument("image writing requires interleaved pixels");

    int channels;
    switch (frame.get_pixel_format()) {
        case Gray: channels = 1; break;
        case RGB:  channels = 3; break;
        case RGBA: channels = 4; break;
        default:
            throw std::invalid_argument("unsupported image pixel format");
    }

    const std::size_t w = frame.get_width();
    const std::size_t h = frame.get_height();
    const std::size_t max_int = std::numeric_limits<int>::max();

    if (w == 0 || h == 0)
        throw std::invalid_argument("invalid image dimensions");

    if (w > max_int / channels || h > max_int)
        throw std::length_error("image is too large");

    const std::size_t stride = w * channels;
    if (h > std::numeric_limits<std::size_t>::max() / stride)
        throw std::length_error("image is too large");

    const auto pixels = frame.view_ref().get_span();
    if (pixels.size() < stride * h)
        throw std::invalid_argument("image buffer is too small");

    if (!stbi_write_png(
        path,
        static_cast<int>(w),
        static_cast<int>(h),
        channels,
        pixels.data(),
        static_cast<int>(stride)
    ))
        throw std::runtime_error("failed to write image");
}
