#include "parhelion-2.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <chrono>
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
              << std::setw(13) << size << std::setw(8) << radius
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

void reference(
    FrameViewRef src,
    FrameViewMut dst,
    std::size_t radius,
    bool rounded
) {
    const std::size_t width = src.get_width();
    const std::size_t height = src.get_height();
    const std::size_t side = 2 * radius + 1;
    const unsigned area = side * side;
    std::vector<unsigned> rows(width * height);
    for (std::size_t c = 0; c < src.get_n_channels(); ++c) {
        const auto src_channel = src.channel_ref(c);
        const auto dst_channel = dst.channel_mut(c);
        const auto input = src_channel.get_span();
        const auto output = dst_channel.get_span();
        const auto src_stride = src_channel.get_row_stride();
        const auto dst_stride = dst_channel.get_row_stride();
        for (std::size_t y = 0; y < height; ++y) {
            const auto row = input.subspan(y * src_stride, width);
            unsigned sum = (radius + 1) * std::to_integer<unsigned>(row[0]);
            for (std::size_t k = 1; k <= radius; ++k)
                sum += std::to_integer<unsigned>(row[std::min(k, width - 1)]);
            for (std::size_t x = 0; x < width; ++x) {
                rows[y * width + x] = sum;
                const auto leaving = x > radius ? x - radius : 0;
                const auto entering = std::min(x + radius + 1, width - 1);
                sum -= std::to_integer<unsigned>(row[leaving]);
                sum += std::to_integer<unsigned>(row[entering]);
            }
        }
        for (std::size_t x = 0; x < width; ++x) {
            unsigned sum = (radius + 1) * rows[x];
            for (std::size_t k = 1; k <= radius; ++k)
                sum += rows[std::min(k, height - 1) * width + x];
            for (std::size_t y = 0; y < height; ++y) {
                output[y * dst_stride + x] = std::byte((sum + (rounded ? area / 2 : 0)) / area);
                const auto leaving = y > radius ? y - radius : 0;
                const auto entering = std::min(y + radius + 1, height - 1);
                sum -= rows[leaving * width + x];
                sum += rows[entering * width + x];
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
    auto ours = [&] { box_blur(src_view, dst_view, radius); };
    auto theirs = [&] {
        for (std::size_t c = 0; c < channels; ++c)
            cv::boxFilter(input[c], output[c], -1, cv::Size(side, side), cv::Point(-1, -1),
                true, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    };
    ours();
    theirs();
    Frame expected(dst_width, dst_height, format, FrameLayout::Planar, empty);
    const auto expected_view = cropped ? expected.view_mut(dst_xy, width, height) : expected.view_mut();
    reference(src_view, expected_view, radius, false);
    const auto expected_pixels = expected.view_ref().get_span();
    check(dst, std::vector<std::byte>(expected_pixels.begin(), expected_pixels.end()));
    reference(src_view, expected_view, radius, true);
    const std::vector<std::byte> library_expected(expected_pixels.begin(), expected_pixels.end());
    check(result, library_expected);
    const auto dst_pixels = dst.view_ref().get_span();
    const std::vector<std::byte> ours_expected(dst_pixels.begin(), dst_pixels.end());
    const char* name = channels == 1 ? "Gray8" : channels == 3 ? "RGB8" : "RGBA8";
    compare(name, width, height, radius, cropped, ours, theirs);
    check(src, original);
    check(dst, ours_expected);
    check(result, library_expected);
}

int main(void) {
    cv::setNumThreads(1);
    std::cout << "Single thread, median of 31 samples (us); Parhelion scratch allocation included\n"
              << std::left << std::setw(7) << "Format"
              << std::setw(13) << "Size" << std::setw(8) << "Radius" << std::setw(10) << "View"
              << std::right << std::setw(14) << "Parhelion"
              << std::setw(14) << "OpenCV" << std::setw(10) << "Speedup" << '\n'
              << std::fixed << std::setprecision(3);
    for (std::size_t radius = 1; radius <= 7; ++radius) {
        for (Coordinate size : {
            Coordinate(64, 64), Coordinate(256, 256), Coordinate(640, 480),
            Coordinate(1920, 1080), Coordinate(3840, 2160)
        }) {
            for (bool cropped : {false, true}) {
                test(PixelFormat::Gray, 1, size.x, size.y, radius, cropped);
                test(PixelFormat::RGB, 3, size.x, size.y, radius, cropped);
                test(PixelFormat::RGBA, 4, size.x, size.y, radius, cropped);
            }
        }
    }
    return 0;
}
