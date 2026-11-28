#include "parhelion-1.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

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

void test(PixelFormat format, std::size_t channels, std::size_t width, std::size_t height, bool cropped) {
    const std::size_t frame_width = cropped ? width + 6 : width;
    const std::size_t frame_height = cropped ? height + 4 : height;
    const Coordinate xy(cropped ? 3 : 0, cropped ? 2 : 0);
    const std::size_t n = frame_width * frame_height;
    const std::size_t color_channels = channels == 4 ? 3 : channels;
    std::vector<std::byte> original(n * channels);
    for (std::size_t i = 0; i < original.size(); ++i)
        original[i] = i % 2 == 0 ? std::byte{0} : std::byte{255};
    std::array<uint8_t, 3> minima{255, 255, 255};
    std::array<uint8_t, 3> maxima{0, 0, 0};
    for (std::size_t c = 0; c < color_channels; ++c) {
        for (std::size_t y = xy.y; y < xy.y + height; ++y) {
            for (std::size_t x = xy.x; x < xy.x + width; ++x) {
                const auto value = static_cast<uint8_t>(10 + c * 60 + (x * 7 + y * 13) % 40);
                original[c * n + y * frame_width + x] = static_cast<std::byte>(value);
                minima[c] = std::min(minima[c], value);
                maxima[c] = std::max(maxima[c], value);
            }
        }
    }
    Frame frame(frame_width, frame_height, format, FrameLayout::Planar, original);
    const auto view = cropped ? frame.view_ref(xy, width, height) : frame.view_ref();
    cv::Mat channel[3];
    for (std::size_t c = 0; c < color_channels; ++c)
        channel[c] = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_8UC1,
            const_cast<std::byte*>(view.get_span().data()) + c * n, view.get_row_stride());
    const bool packed = view.get_row_stride() == width && view.get_channel_stride() == width * height;
    cv::Mat combined;
    if (packed)
        combined = cv::Mat(static_cast<int>(height * color_channels), static_cast<int>(width), CV_8UC1,
            const_cast<std::byte*>(view.get_span().data()), view.get_row_stride());
    auto cv_global = [&] {
        double lo = 255;
        double hi = 0;
        if (packed) {
            cv::minMaxLoc(combined, &lo, &hi);
        } else {
            for (std::size_t c = 0; c < color_channels; ++c) {
                double cmin;
                double cmax;
                cv::minMaxLoc(channel[c], &cmin, &cmax);
                lo = std::min(lo, cmin);
                hi = std::max(hi, cmax);
            }
        }
        return std::pair<uint8_t, uint8_t>{static_cast<uint8_t>(lo), static_cast<uint8_t>(hi)};
    };
    auto cv_per_channel = [&] {
        std::array<uint8_t, 3> lo;
        std::array<uint8_t, 3> hi;
        for (std::size_t c = 0; c < 3; ++c) {
            double cmin;
            double cmax;
            cv::minMaxLoc(channel[c], &cmin, &cmax);
            lo[c] = static_cast<uint8_t>(cmin);
            hi[c] = static_cast<uint8_t>(cmax);
        }
        return std::pair{lo, hi};
    };

    const std::pair<uint8_t, uint8_t> expected{
        *std::min_element(minima.begin(), minima.end()),
        *std::max_element(maxima.begin(), maxima.end())
    };
    if (minmax_global(view) != expected || cv_global() != expected)
        throw std::runtime_error("global minmax correctness check failed");
    volatile std::uint64_t result = 0;
    auto ours = [&](int) {
        const auto value = minmax_global(view);
        result = std::uint64_t(value.first) | (std::uint64_t(value.second) << 8);
    };
    auto theirs = [&](int) {
        const auto value = cv_global();
        result = std::uint64_t(value.first) | (std::uint64_t(value.second) << 8);
    };
    const char* name = channels == 1 ? "Gray8" : channels == 3 ? "RGB8" : "RGBA8";
    compare(name, width, height, cropped ? "global cropped" : "global packed", ours, theirs);

    if (channels > 1) {
        const std::pair expected{minima, maxima};
        if (minmax_per_channel(view) != expected || cv_per_channel() != expected)
            throw std::runtime_error("per-channel minmax correctness check failed");
        auto ours = [&](int) {
            const auto value = minmax_per_channel(view);
            std::uint64_t packed = 0;
            for (std::size_t c = 0; c < 3; ++c) {
                packed |= std::uint64_t(value.first[c]) << (8 * c);
                packed |= std::uint64_t(value.second[c]) << (24 + 8 * c);
            }
            result = packed;
        };
        auto theirs = [&](int) {
            const auto value = cv_per_channel();
            std::uint64_t packed = 0;
            for (std::size_t c = 0; c < 3; ++c) {
                packed |= std::uint64_t(value.first[c]) << (8 * c);
                packed |= std::uint64_t(value.second[c]) << (24 + 8 * c);
            }
            result = packed;
        };
        compare(name, width, height, cropped ? "per-channel cropped" : "per-channel packed", ours, theirs);
    }
    const auto pixels = frame.view_ref().get_span();
    if (!std::equal(pixels.begin(), pixels.end(), original.begin()))
        throw std::runtime_error("minmax modified the source");
    (void)result;
}

int main(void) {
    cv::setNumThreads(1);
    std::cout << "Single thread, preallocated, median of 31 samples (us)\n"
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
            test(PixelFormat::Gray, 1, size.x, size.y, cropped);
            test(PixelFormat::RGB, 3, size.x, size.y, cropped);
            test(PixelFormat::RGBA, 4, size.x, size.y, cropped);
        }
    }
    return 0;
}
