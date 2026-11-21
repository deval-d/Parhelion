#pragma once 
#include "image_types.hpp"
#include <stb_image.h>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

inline Frame image_load(const char* path) {
    if (path == nullptr)
        throw std::invalid_argument("image path is null");

    int width = 0;
    int height = 0;
    int original_channels = 0;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load(path, &width, &height, &original_channels, 0),
        stbi_image_free
    );

    if (!pixels)
        throw std::runtime_error("failed to load image");

    if (width <= 0 || height <= 0)
        throw std::runtime_error("invalid image dimensions");

    PixelFormat format;
    switch (original_channels) {
        case 1: format = PixelFormat::Gray; break;
        case 3: format = PixelFormat::RGB;  break;
        case 4: format = PixelFormat::RGBA; break;
        default:
            throw std::runtime_error("unsupported image channel count");
    }

    const std::size_t w = static_cast<std::size_t>(width);
    const std::size_t h = static_cast<std::size_t>(height);
    const std::size_t channels = static_cast<std::size_t>(original_channels);

    if (w > std::numeric_limits<std::size_t>::max() / channels / h)
        throw std::length_error("image is too large");

    std::vector<std::byte> buffer(w * h * channels);
    std::memcpy(buffer.data(), pixels.get(), buffer.size());

    return Frame(w, h, format, FrameLayout::Interleaved, std::move(buffer));
}
