#include "parhelion-2.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
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
    const char* operation,
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
              << std::setw(13) << size << std::setw(12) << operation
              << std::setw(10) << (cropped ? "cropped" : "packed")
              << std::right << std::setw(14) << ours[15]
              << std::setw(14) << theirs[15]
              << std::setw(10) << theirs[15] / ours[15] << std::endl;
}

void check(const Frame& frame, const std::vector<std::byte>& expected) {
    const auto pixels = frame.view_ref().get_span();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (pixels[i] != expected[i]) {
            std::cerr << "byte " << i
                      << ": received " << std::to_integer<int>(pixels[i])
                      << ", expected " << std::to_integer<int>(expected[i]) << '\n';
            throw std::runtime_error("benchmark correctness check failed");
        }
    }
}

void reference(FrameViewRef src, FrameViewMut dst, std::span<const std::int16_t> kernel) {
    const std::size_t width = src.get_width();
    const std::size_t height = src.get_height();
    for (std::size_t c = 0; c < src.get_n_channels(); ++c) {
        const auto sc = src.channel_ref(c);
        const auto dc = dst.channel_mut(c);
        const auto input = sc.get_span();
        const auto output = dc.get_span();
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                int sum = 0;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        const auto yy = std::clamp<std::ptrdiff_t>(std::ptrdiff_t(y) + ky - 1, 0, height - 1);
                        const auto xx = std::clamp<std::ptrdiff_t>(std::ptrdiff_t(x) + kx - 1, 0, width - 1);
                        sum += kernel[ky * 3 + kx] * std::to_integer<int>(input[yy * sc.get_row_stride() + xx]);
                    }
                }
                output[y * dc.get_row_stride() + x] = std::byte(std::clamp(sum, 0, 255));
            }
        }
    }
}

void test(
    PixelFormat format,
    std::size_t channels,
    std::size_t width,
    std::size_t height,
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
    const std::array<std::int16_t, 9> weights = {0, 1, 0, 1, -4, 1, 0, 1, 0};
    cv::Mat input[4];
    cv::Mat output[4];
    for (std::size_t c = 0; c < channels; ++c) {
        const auto src_channel = src_view.channel_ref(c);
        const auto dst_channel = result_view.channel_mut(c);
        input[c] = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_8UC1,
            const_cast<std::byte*>(src_channel.get_span().data()), src_channel.get_row_stride());
        output[c] = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_8UC1,
            dst_channel.get_span().data(), dst_channel.get_row_stride());
    }
    auto ours = [&] { laplacian(src_view, dst_view); };
    auto theirs = [&] {
        for (std::size_t c = 0; c < channels; ++c)
            cv::Laplacian(input[c], output[c], CV_8U, 1, 1, 0,
                cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    };
    ours();
    theirs();
    Frame expected(dst_width, dst_height, format, FrameLayout::Planar, empty);
    const auto expected_view = cropped ? expected.view_mut(dst_xy, width, height) : expected.view_mut();
    reference(src_view, expected_view, weights);
    const auto expected_pixels = expected.view_ref().get_span();
    const std::vector<std::byte> library_expected(expected_pixels.begin(), expected_pixels.end());
    check(dst, library_expected);
    check(result, library_expected);
    const auto dst_pixels = dst.view_ref().get_span();
    const std::vector<std::byte> ours_expected(dst_pixels.begin(), dst_pixels.end());
    const char* name = channels == 1 ? "Gray8" : channels == 3 ? "RGB8" : "RGBA8";
    compare(name, width, height, "laplacian", cropped, ours, theirs);
    check(src, original);
    check(dst, ours_expected);
    check(result, library_expected);
}

int main(void) {
    cv::setNumThreads(1);
    std::cout << "Single thread, median of 31 samples (us); no Parhelion scratch allocation\n"
              << std::left << std::setw(7) << "Format"
              << std::setw(13) << "Size" << std::setw(12) << "Kernel" << std::setw(10) << "View"
              << std::right << std::setw(14) << "Parhelion"
              << std::setw(14) << "OpenCV" << std::setw(10) << "Speedup" << '\n'
              << std::fixed << std::setprecision(3);
    for (Coordinate size : {
        Coordinate(256, 256), Coordinate(1920, 1080), Coordinate(3840, 2160)
    }) {
        for (bool cropped : {false, true}) {
            test(PixelFormat::Gray, 1, size.x, size.y, cropped);
            test(PixelFormat::RGB, 3, size.x, size.y, cropped);
            test(PixelFormat::RGBA, 4, size.x, size.y, cropped);
        }
    }
    return 0;
}
