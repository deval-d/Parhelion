#include "parhelion-2.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <iostream>
#include <stdexcept>

void check_frame(const Frame& frame, const std::vector<std::byte>& expected,
    const std::vector<bool>& approximate = {}) {
    const auto pixels = frame.view_ref().get_span();
    if (pixels.size() != expected.size())
        throw std::runtime_error("incorrect buffer size");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const int difference = std::abs(std::to_integer<int>(pixels[i]) - std::to_integer<int>(expected[i]));
        const int tolerance = approximate.empty() ? 0 : approximate[i];
        if (difference > tolerance) {
            std::cerr << "byte " << i
                      << ": received " << std::to_integer<int>(pixels[i])
                      << ", expected " << std::to_integer<int>(expected[i]) << '\n';
            throw std::runtime_error("test failed");
        }
    }
}

void test(
    PixelFormat format,
    std::size_t channels,
    std::size_t width,
    std::size_t height,
    std::size_t radius,
    float sigma,
    bool cropped,
    int pattern
) {
    const std::size_t src_width = cropped ? width + 6 : width;
    const std::size_t src_height = cropped ? height + 4 : height;
    const std::size_t dst_width = cropped ? width + 10 : width;
    const std::size_t dst_height = cropped ? height + 6 : height;
    const Coordinate src_xy(cropped ? 3 : 0, cropped ? 2 : 0);
    const Coordinate dst_xy(cropped ? 5 : 0, cropped ? 3 : 0);
    const std::size_t src_n = src_width * src_height;
    const std::size_t dst_n = dst_width * dst_height;

    std::vector<std::byte> original(src_n * channels, std::byte{137});
    for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const int value = pattern == 0 ? 0 : pattern == 1 ? 255
                    : pattern == 2 ? (c * 59 + y * 37 + x * 19) % 256
                    : (x == width / 2 && y == height / 2 ? 255 : 0);
                original[c * src_n + (src_xy.y + y) * src_width + src_xy.x + x] = std::byte(value);
            }
        }
    }
    std::vector<std::byte> expected(dst_n * channels, std::byte{79});
    Frame src(src_width, src_height, format, FrameLayout::Planar, original);
    Frame dst(dst_width, dst_height, format, FrameLayout::Planar, expected);
    const auto src_view = cropped ? src.view_ref(src_xy, width, height) : src.view_ref();
    const auto dst_view = cropped ? dst.view_mut(dst_xy, width, height) : dst.view_mut();

    std::vector<double> weights(radius + 1);
    double total = 0.0;
    for (std::size_t k = 0; k <= radius; ++k) {
        const double distance = double(k) / sigma;
        weights[k] = std::exp(-0.5 * distance * distance);
        total += (k == 0 ? 1 : 2) * weights[k];
    }
    for (double& weight : weights)
        weight /= total;
    std::vector<bool> approximate(expected.size(), false);
    for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                double sum = 0.0;
                for (std::ptrdiff_t dy = -std::ptrdiff_t(radius); dy <= std::ptrdiff_t(radius); ++dy) {
                    for (std::ptrdiff_t dx = -std::ptrdiff_t(radius); dx <= std::ptrdiff_t(radius); ++dx) {
                        const auto yy = std::clamp<std::ptrdiff_t>(std::ptrdiff_t(y) + dy, 0, height - 1);
                        const auto xx = std::clamp<std::ptrdiff_t>(std::ptrdiff_t(x) + dx, 0, width - 1);
                        const auto index = c * src_n + (src_xy.y + yy) * src_width + src_xy.x + xx;
                        sum += weights[std::abs(dy)] * weights[std::abs(dx)] * std::to_integer<unsigned>(original[index]);
                    }
                }
                const auto index = c * dst_n + (dst_xy.y + y) * dst_width + dst_xy.x + x;
                expected[index] = std::byte(int(std::nearbyint(std::clamp(sum, 0.0, 255.0))));
                approximate[index] = pattern >= 2 && radius > 0;
            }
        }
    }
    gaussian_blur(src_view, dst_view, sigma, radius);
    try {
        check_frame(src, original);
        check_frame(dst, expected, approximate);
    } catch (const std::runtime_error&) {
        std::cerr << "Gaussian blur: " << width << " x " << height
                  << ", channels " << channels << ", radius " << radius
                  << ", sigma " << sigma << ", cropped " << cropped << ", pattern " << pattern << '\n';
        throw;
    }
}

void test_sizes(PixelFormat format, std::size_t channels) {
    for (Coordinate size : {
        Coordinate(1, 1), Coordinate(1, 17), Coordinate(17, 1),
        Coordinate(2, 3), Coordinate(7, 9), Coordinate(8, 9), Coordinate(9, 9),
        Coordinate(15, 17), Coordinate(16, 17), Coordinate(17, 17),
        Coordinate(31, 3), Coordinate(32, 3), Coordinate(33, 3), Coordinate(65, 17)
    }) {
        for (std::size_t radius : {0, 1, 2, 4, 7, 12})
            for (float sigma : {0.01f, 1.5f, 4.0f})
                for (bool cropped : {false, true})
                    for (int pattern : {0, 1, 2, 3})
                        test(format, channels, size.x, size.y, radius, sigma, cropped, pattern);
    }
}

template <typename Action>
void check_invalid(Frame& src, Frame& dst, Action action) {
    const auto src_pixels = src.view_ref().get_span();
    const auto dst_pixels = dst.view_ref().get_span();
    const std::vector<std::byte> original(src_pixels.begin(), src_pixels.end());
    const std::vector<std::byte> expected(dst_pixels.begin(), dst_pixels.end());
    bool rejected = false;
    try {
        action();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("invalid Gaussian blur input was accepted");
    check_frame(src, original);
    check_frame(dst, expected);
}

int invalid_inputs(void) {
    Frame src(17, 9, PixelFormat::Gray, FrameLayout::Planar,
        std::vector<std::byte>(17 * 9, std::byte{100}));
    Frame dst(17, 9, PixelFormat::Gray, FrameLayout::Planar,
        std::vector<std::byte>(17 * 9, std::byte{79}));
    for (float sigma : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
        check_invalid(src, dst, [&] { gaussian_blur(src.view_ref(), dst.view_mut(), sigma, 2); });
    check_invalid(src, src, [&] { gaussian_blur(src.view_ref(), src.view_mut(), 1.5f, 2); });
    check_invalid(src, dst, [&] {
        gaussian_blur(src.view_ref(Coordinate(0, 0), 16, 9), dst.view_mut(), 1.5f, 2);
    });
    Frame rgb(17, 9, PixelFormat::RGB, FrameLayout::Planar,
        std::vector<std::byte>(17 * 9 * 3, std::byte{100}));
    check_invalid(rgb, dst, [&] { gaussian_blur(rgb.view_ref(), dst.view_mut(), 1.5f, 2); });
    Frame interleaved(17, 9, PixelFormat::Gray, FrameLayout::Interleaved,
        std::vector<std::byte>(17 * 9, std::byte{100}));
    check_invalid(interleaved, dst, [&] { gaussian_blur(interleaved.view_ref(), dst.view_mut(), 1.5f, 2); });
    check_invalid(src, interleaved, [&] { gaussian_blur(src.view_ref(), interleaved.view_mut(), 1.5f, 2); });
    return 0;
}

int gray8(void) {
    test_sizes(PixelFormat::Gray, 1);
    return 0;
}

int rgb8(void) {
    test_sizes(PixelFormat::RGB, 3);
    return 0;
}

int rgba8(void) {
    test_sizes(PixelFormat::RGBA, 4);
    return 0;
}

int main(void) {
    int status = 0;
    status |= gray8();
    status |= rgb8();
    status |= rgba8();
    status |= invalid_inputs();
    return status;
}
