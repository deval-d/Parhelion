#include "image_types.hpp"
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

void premultiply_alpha(FrameViewMut frame);

using Clock = std::chrono::steady_clock;

template <typename Convert, typename Reset>
double measure(Convert convert, Reset reset, int repeats) {
    double elapsed = 0;
    for (int i = 0; i < repeats; ++i) {
        reset();
        const auto start = Clock::now();
        convert(i);
        const auto end = Clock::now();
        elapsed += std::chrono::duration<double, std::micro>(end - start).count();
    }
    return elapsed / repeats;
}

template <typename Parhelion, typename Library, typename ResetParhelion, typename ResetLibrary>
void compare(
    const char* format,
    std::size_t width,
    std::size_t height,
    const char* label,
    Parhelion parhelion,
    Library library,
    ResetParhelion reset_ours,
    ResetLibrary reset_theirs
) {
    std::array<double, 31> ours;
    std::array<double, 31> theirs;
    const int repeats = width * height <= 65536 ? 128 : 2;

    for (int i = -5; i < 31; ++i) {
        double a;
        double b;
        if (i % 2 == 0) {
            a = measure(parhelion, reset_ours, repeats);
            b = measure(library, reset_theirs, repeats);
        } else {
            b = measure(library, reset_theirs, repeats);
            a = measure(parhelion, reset_ours, repeats);
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

void test(std::size_t width, std::size_t height, bool cropped) {
    const std::size_t frame_width = cropped ? width + 6 : width;
    const std::size_t frame_height = cropped ? height + 4 : height;
    const Coordinate xy(cropped ? 3 : 0, cropped ? 2 : 0);
    const std::size_t n = frame_width * frame_height;
    std::vector<std::byte> original(4 * n);
    for (std::size_t i = 0; i < original.size(); ++i)
        original[i] = static_cast<std::byte>((i * 7 + i / 251 + 13) % 256);
    Frame frame(frame_width, frame_height, PixelFormat::RGBA, FrameLayout::Planar, original);
    const auto view = cropped ? frame.view_mut(xy, width, height) : frame.view_mut();
    auto pixels = frame.view_mut().get_span();
    std::vector<std::uint8_t> rgba(4 * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t c = 0; c < 4; ++c)
            rgba[4 * i + c] = std::to_integer<std::uint8_t>(original[c * n + i]);
    auto packed_original = rgba;
    cv::Mat channel(static_cast<int>(height), static_cast<int>(width), CV_8UC4,
        rgba.data() + 4 * (xy.y * frame_width + xy.x), 4 * frame_width);
    auto ours = [&](int) { premultiply_alpha(view); };
    auto theirs = [&](int) { cv::cvtColor(channel, channel, cv::COLOR_RGBA2mRGBA); };
    auto reset_ours = [&] { std::copy(original.begin(), original.end(), pixels.begin()); };
    auto reset_theirs = [&] { std::copy(packed_original.begin(), packed_original.end(), rgba.begin()); };

    auto expected = original;
    auto rounded = original;
    for (std::size_t y = xy.y; y < xy.y + height; ++y) {
        for (std::size_t x = xy.x; x < xy.x + width; ++x) {
            const std::size_t i = y * frame_width + x;
            const int alpha = std::to_integer<int>(original[3 * n + i]);
            for (std::size_t c = 0; c < 3; ++c) {
                const int value = std::to_integer<int>(original[c * n + i]);
                expected[c * n + i] = static_cast<std::byte>((value * alpha + 127) / 255);
                rounded[c * n + i] = static_cast<std::byte>((value * alpha + 127) / 255);
            }
        }
    }
    ours(0);
    check(frame, expected);
    theirs(0);
    for (std::size_t i = 0; i < n; ++i) {
        const bool inside = i / frame_width >= xy.y && i / frame_width < xy.y + height &&
            i % frame_width >= xy.x && i % frame_width < xy.x + width;
        for (std::size_t c = 0; c < 4; ++c) {
            const int difference = std::abs(int(rgba[4 * i + c]) - std::to_integer<int>(rounded[c * n + i]));
            if (difference > (inside && c < 3 ? 1 : 0))
                throw std::runtime_error("RGBA correctness check failed");
        }
    }
    const char* label = cropped ? "cropped" : "packed";
    compare("RGBA8", width, height, label, ours, theirs, reset_ours, reset_theirs);
    check(frame, expected);
}

int main(void) {
    cv::setNumThreads(1);
    std::cout << "Parhelion NEON planar RGBA, integer rounding to nearest; library rounds to nearest; alpha preserved\n";
    std::cout << "OpenCV cvtColor RGBA2mRGBA; color rounding tolerance 1; interleaved input prepared outside timing\n";
    std::cout << "Single thread, preallocated, buffers restored outside each timed call; median of 31 samples (us)\n"
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
            test(size.x, size.y, cropped);
        }
    }
    return 0;
}
