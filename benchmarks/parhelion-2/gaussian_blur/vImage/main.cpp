#include "parhelion-2.hpp"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

using Clock = std::chrono::steady_clock;

template <typename Operation>
double measure(Operation operation, int repeats) {
    const auto start = Clock::now();
    for (int i = 0; i < repeats; ++i)
        operation();
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count() / repeats;
}

template <typename Parhelion, typename Library>
void compare(
    const char* format,
    std::size_t width,
    std::size_t height,
    std::size_t radius,
    float sigma,
    bool cropped,
    Parhelion parhelion,
    Library library
) {
    std::array<double, 31> ours;
    std::array<double, 31> theirs;
    const int repeats = width * height <= 65536 ? 16 : 1;
    for (int i = -5; i < 31; ++i) {
        double a;
        double b;
        if (i % 2 == 0) {
            a = measure(parhelion, repeats);
            b = measure(library, repeats);
        } else {
            b = measure(library, repeats);
            a = measure(parhelion, repeats);
        }
        if (i >= 0) {
            ours[i] = a;
            theirs[i] = b;
        }
    }
    std::sort(ours.begin(), ours.end());
    std::sort(theirs.begin(), theirs.end());
    const std::string size = std::to_string(width) + "x" + std::to_string(height);
    std::cout << std::left << std::setw(7) << format
              << std::setw(13) << size << std::setw(8) << radius << std::setw(8) << sigma
              << std::setw(10) << (cropped ? "cropped" : "packed")
              << std::right << std::setw(14) << ours[15]
              << std::setw(14) << theirs[15]
              << std::setw(10) << theirs[15] / ours[15] << std::endl;
}

void check(const Frame& frame, const std::vector<std::byte>& expected,
    int tolerance = 0, const std::vector<bool>& active = {}) {
    const auto pixels = frame.view_ref().get_span();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const int difference = std::abs(std::to_integer<int>(pixels[i]) - std::to_integer<int>(expected[i]));
        const int allowed = active.empty() || !active[i] ? 0 : tolerance;
        if (difference > allowed) {
            std::cerr << "byte " << i
                      << ": received " << std::to_integer<int>(pixels[i])
                      << ", expected " << std::to_integer<int>(expected[i]) << '\n';
            throw std::runtime_error("benchmark correctness check failed");
        }
    }
}

void reference(FrameViewRef src, FrameViewMut dst, float sigma, std::size_t radius) {
    const std::size_t width = src.get_width();
    const std::size_t height = src.get_height();
    std::vector<double> weights(radius + 1);
    double total = 0.0;
    for (std::size_t k = 0; k <= radius; ++k) {
        const double distance = double(k) / sigma;
        weights[k] = std::exp(-0.5 * distance * distance);
        total += (k == 0 ? 1 : 2) * weights[k];
    }
    for (double& weight : weights)
        weight /= total;
    std::vector<double> rows(width * height);
    for (std::size_t c = 0; c < src.get_n_channels(); ++c) {
        const auto sc = src.channel_ref(c);
        const auto dc = dst.channel_mut(c);
        const auto input = sc.get_span();
        const auto output = dc.get_span();
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                double sum = weights[0] * std::to_integer<int>(input[y * sc.get_row_stride() + x]);
                for (std::size_t k = 1; k <= radius; ++k) {
                    const auto top = k > y ? 0 : y - k;
                    const auto bottom = k >= height - y ? height - 1 : y + k;
                    sum += weights[k] * (std::to_integer<int>(input[top * sc.get_row_stride() + x])
                        + std::to_integer<int>(input[bottom * sc.get_row_stride() + x]));
                }
                rows[y * width + x] = sum;
            }
        }
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                double sum = weights[0] * rows[y * width + x];
                for (std::size_t k = 1; k <= radius; ++k) {
                    const auto left = k > x ? 0 : x - k;
                    const auto right = k >= width - x ? width - 1 : x + k;
                    sum += weights[k] * (rows[y * width + left] + rows[y * width + right]);
                }
                output[y * dc.get_row_stride() + x] = std::byte(int(std::nearbyint(std::clamp(sum, 0.0, 255.0))));
            }
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
    bool cropped
) {
    const std::size_t src_width = cropped ? width + 6 : width;
    const std::size_t src_height = cropped ? height + 4 : height;
    const std::size_t dst_width = cropped ? width + 10 : width;
    const std::size_t dst_height = cropped ? height + 6 : height;
    const Coordinate src_xy(cropped ? 3 : 0, cropped ? 2 : 0);
    const Coordinate dst_xy(cropped ? 5 : 0, cropped ? 3 : 0);
    std::vector<std::byte> original(src_width * src_height * channels);
    std::uint32_t state = 123456789;
    for (auto& pixel : original) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        pixel = std::byte(state & 255);
    }
    const std::vector<std::byte> empty(dst_width * dst_height * channels, std::byte{79});
    Frame src(src_width, src_height, format, FrameLayout::Planar, original);
    Frame dst(dst_width, dst_height, format, FrameLayout::Planar, empty);
    Frame result(dst_width, dst_height, format, FrameLayout::Planar, empty);
    const auto src_view = cropped ? src.view_ref(src_xy, width, height) : src.view_ref();
    const auto dst_view = cropped ? dst.view_mut(dst_xy, width, height) : dst.view_mut();
    const auto result_view = cropped ? result.view_mut(dst_xy, width, height) : result.view_mut();
    const std::size_t side = 2 * radius + 1;
    std::vector<float> weights(side);
    double total = 0.0;
    for (std::size_t k = 0; k < side; ++k) {
        const double distance = (double(k) - double(radius)) / sigma;
        weights[k] = std::exp(-0.5 * distance * distance);
        total += weights[k];
    }
    for (float& weight : weights)
        weight /= total;
    vImage_Buffer input[4];
    vImage_Buffer output[4];
    const vImage_Flags flags = kvImageEdgeExtend | kvImageDoNotTile;
    std::size_t scratch_size = 1;
    for (std::size_t c = 0; c < channels; ++c) {
        const auto src_channel = src_view.channel_ref(c);
        const auto dst_channel = result_view.channel_mut(c);
        input[c] = {const_cast<std::byte*>(src_channel.get_span().data()), height, width, src_channel.get_row_stride()};
        output[c] = {dst_channel.get_span().data(), height, width, dst_channel.get_row_stride()};
        const auto size = vImageSepConvolve_Planar8(&input[c], &output[c], nullptr,
            0, 0, weights.data(), side, weights.data(), side, 0.0f, 0, flags | kvImageGetTempBufferSize);
        if (size < 0)
            throw std::runtime_error("vImage scratch query failed");
        scratch_size = std::max(scratch_size, static_cast<std::size_t>(size));
    }
    std::vector<std::byte> scratch(scratch_size);
    auto ours = [&] { gaussian_blur(src_view, dst_view, sigma, radius); };
    auto theirs = [&] {
        for (std::size_t c = 0; c < channels; ++c) {
            if (vImageSepConvolve_Planar8(&input[c], &output[c], scratch.data(),
                0, 0, weights.data(), side, weights.data(), side, 0.0f, 0, flags) != kvImageNoError)
                throw std::runtime_error("vImage gaussian_blur failed");
        }
    };
    ours();
    theirs();
    Frame expected(dst_width, dst_height, format, FrameLayout::Planar, empty);
    const auto expected_view = cropped ? expected.view_mut(dst_xy, width, height) : expected.view_mut();
    reference(src_view, expected_view, sigma, radius);
    const auto expected_pixels = expected.view_ref().get_span();
    const std::vector<std::byte> library_expected(expected_pixels.begin(), expected_pixels.end());
    std::vector<bool> active(library_expected.size(), false);
    for (std::size_t c = 0; c < channels; ++c)
        for (std::size_t y = 0; y < height; ++y)
            for (std::size_t x = 0; x < width; ++x)
                active[c * dst_width * dst_height + (dst_xy.y + y) * dst_width + dst_xy.x + x] = true;
    check(dst, library_expected, 1, active);
    check(result, library_expected, 1, active);
    const auto dst_pixels = dst.view_ref().get_span();
    const std::vector<std::byte> ours_expected(dst_pixels.begin(), dst_pixels.end());
    const char* name = channels == 1 ? "Gray8" : channels == 3 ? "RGB8" : "RGBA8";
    compare(name, width, height, radius, sigma, cropped, ours, theirs);
    check(src, original);
    check(dst, ours_expected);
    check(result, library_expected, 1, active);
}

int main(void) {

    std::cout << "Single thread, median of 31 samples (us); Parhelion scratch allocation included; vImage scratch preallocated and weights precomputed\n"
              << std::left << std::setw(7) << "Format"
              << std::setw(13) << "Size" << std::setw(8) << "Radius" << std::setw(8) << "Sigma" << std::setw(10) << "View"
              << std::right << std::setw(14) << "Parhelion"
              << std::setw(14) << "vImage" << std::setw(10) << "Speedup" << '\n'
              << std::fixed << std::setprecision(3);
    for (auto config : {std::pair{1, 0.8f}, std::pair{4, 1.5f}, std::pair{7, 2.5f}}) {
        for (Coordinate size : {
            Coordinate(256, 256), Coordinate(1920, 1080), Coordinate(3840, 2160)
        }) {
            for (bool cropped : {false, true}) {
                test(PixelFormat::Gray, 1, size.x, size.y, config.first, config.second, cropped);
                test(PixelFormat::RGB, 3, size.x, size.y, config.first, config.second, cropped);
                test(PixelFormat::RGBA, 4, size.x, size.y, config.first, config.second, cropped);
            }
        }
    }
    return 0;
}
