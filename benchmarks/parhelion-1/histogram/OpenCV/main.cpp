#include "image_types.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

std::array<std::uint64_t, 256> histogram_gray(FrameViewRef frame);
std::array<std::array<std::uint64_t, 256>, 3> histogram_rgb(FrameViewRef frame);
std::array<std::array<std::uint64_t, 256>, 4> histogram_rgba(FrameViewRef frame);

using Clock = std::chrono::steady_clock;

template <typename Convert>
double measure(Convert convert, int repeats) {
    const auto start = Clock::now();
    for (int i = 0; i < repeats; ++i)
        convert(i);
    const auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / repeats;
}

template <typename Parhelion, typename Library>
void compare(
    const char* format,
    std::size_t width,
    std::size_t height,
    const char* label,
    Parhelion parhelion,
    Library library
) {
    std::array<double, 31> ours;
    std::array<double, 31> theirs;
    const int repeats = width * height <= 65536 ? 128 : 2;

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
              << std::setw(13) << size << std::setw(22) << label
              << std::right << std::setw(14) << ours[15]
              << std::setw(14) << theirs[15]
              << std::setw(10) << theirs[15] / ours[15] << '\n';
}

void test(PixelFormat format, std::size_t channels, std::size_t width, std::size_t height, bool cropped, bool constant) {
    const std::size_t frame_width = cropped ? width + 6 : width;
    const std::size_t frame_height = cropped ? height + 4 : height;
    const Coordinate xy(cropped ? 3 : 0, cropped ? 2 : 0);
    const std::size_t n = frame_width * frame_height;
    std::vector<std::byte> original(n * channels);
    std::uint32_t state = 123456789;
    for (std::size_t i = 0; i < original.size(); ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        original[i] = static_cast<std::byte>(constant ? 31 + 47 * (i / n) : state & 255);
    }
    Frame frame(frame_width, frame_height, format, FrameLayout::Planar, original);
    const auto view = cropped ? frame.view_ref(xy, width, height) : frame.view_ref();
    std::array<std::array<std::uint64_t, 256>, 4> expected{};
    std::array<std::array<std::uint64_t, 256>, 4> result{};
    for (std::size_t c = 0; c < channels; ++c)
        for (std::size_t y = xy.y; y < xy.y + height; ++y)
            for (std::size_t x = xy.x; x < xy.x + width; ++x)
                ++expected[c][std::to_integer<unsigned>(original[c * n + y * frame_width + x])];
    auto ours = [&](int) {
        switch (format) {
            case PixelFormat::Gray:
                result[0] = histogram_gray(view);
                break;
            case PixelFormat::RGB: {
                const auto hist = histogram_rgb(view);
                for (std::size_t c = 0; c < 3; ++c)
                    result[c] = hist[c];
                break;
            }
            case PixelFormat::RGBA:
                result = histogram_rgba(view);
                break;
        }
    };
    cv::Mat channel[4];
    cv::Mat hist[4];
    for (std::size_t c = 0; c < channels; ++c) {
        channel[c] = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_8UC1,
            const_cast<std::byte*>(view.get_span().data()) + c * n, view.get_row_stride());
        hist[c] = cv::Mat(256, 1, CV_32F);
    }
    const int bins = 256;
    const int index = 0;
    const float range[] = {0, 256};
    const float* ranges[] = {range};
    auto theirs = [&](int) {
        for (std::size_t c = 0; c < channels; ++c)
            cv::calcHist(&channel[c], 1, &index, cv::noArray(), hist[c], 1, &bins, ranges, true, false);
    };
    auto check = [&] {
        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t i = 0; i < 256; ++i) {
                if (result[c][i] != expected[c][i] || hist[c].at<float>(static_cast<int>(i)) != expected[c][i]) {
                    std::cerr << "channel " << c << ", bin " << i << ": expected " << expected[c][i]
                              << ", Parhelion " << result[c][i] << ", library " << hist[c].at<float>(static_cast<int>(i)) << '\n';
                    throw std::runtime_error("histogram correctness check failed");
                }
            }
        }
    };
    ours(0);
    theirs(0);
    check();
    const char* name = channels == 1 ? "Gray8" : channels == 3 ? "RGB8" : "RGBA8";
    const char* label = constant ? (cropped ? "cropped constant" : "packed constant")
        : (cropped ? "cropped mixed" : "packed mixed");
    compare(name, width, height, label, ours, theirs);
    check();
    const auto pixels = frame.view_ref().get_span();
    if (!std::equal(pixels.begin(), pixels.end(), original.begin()))
        throw std::runtime_error("histogram modified the source");
}

int main(void) {
    cv::setNumThreads(1);
    std::cout << "Single thread, preallocated library outputs, median of 31 samples (us)\n"
              << std::left << std::setw(7) << "Format"
              << std::setw(13) << "Size" << std::setw(22) << "View"
              << std::right << std::setw(14) << "Parhelion"
              << std::setw(14) << "OpenCV" << std::setw(10) << "Speedup" << '\n'
              << std::fixed << std::setprecision(3);
    for (Coordinate size : {
        Coordinate(64, 64), Coordinate(256, 256), Coordinate(640, 480),
        Coordinate(1280, 720), Coordinate(1920, 1080),
        Coordinate(2560, 1440), Coordinate(3840, 2160)
    }) {
        for (bool cropped : {false, true}) {
            for (bool constant : {false, true}) {
                test(PixelFormat::Gray, 1, size.x, size.y, cropped, constant);
                test(PixelFormat::RGB, 3, size.x, size.y, cropped, constant);
                test(PixelFormat::RGBA, 4, size.x, size.y, cropped, constant);
            }
        }
    }
    return 0;
}
