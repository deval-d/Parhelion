#include "../../../threading/mode.hpp"
#include "parhelion-1.hpp"
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

void test(std::size_t width, std::size_t height, bool cropped, bool interleaved) {
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
    std::vector<std::byte> planar = original;
    vImage_Buffer channel[4];
    for (std::size_t c = 0; c < 4; ++c)
        channel[c] = {planar.data() + c * n + xy.y * frame_width + xy.x,
            height, width, frame_width};
    vImage_Buffer rgba_channel = {rgba.data() + 4 * (xy.y * frame_width + xy.x),
        height, width, 4 * frame_width};
    auto ours = [&](int) { PARHELION_BENCH_CALL(premultiply_alpha, view); };
    auto theirs = [&](int) {
        if (interleaved) {
            if (vImagePremultiplyData_RGBA8888(&rgba_channel, &rgba_channel, (benchmark_threads == 1 ? kvImageDoNotTile : kvImageNoFlags)) != kvImageNoError)
                throw std::runtime_error("vImage premultiply failed");
        } else {
            for (std::size_t c = 0; c < 3; ++c)
                if (vImagePremultiplyData_Planar8(&channel[c], &channel[3], &channel[c], (benchmark_threads == 1 ? kvImageDoNotTile : kvImageNoFlags)) != kvImageNoError)
                    throw std::runtime_error("vImage premultiply failed");
        }
    };
    auto reset_ours = [&] { std::copy(original.begin(), original.end(), pixels.begin()); };
    auto reset_theirs = [&] {
        if (interleaved)
            std::copy(packed_original.begin(), packed_original.end(), rgba.begin());
        else
            std::copy(original.begin(), original.end(), planar.begin());
    };

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
    if (!interleaved) {
        if (planar != rounded)
            throw std::runtime_error("vImage planar correctness check failed");
    } else {
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t c = 0; c < 4; ++c)
            if (rgba[4 * i + c] != std::to_integer<std::uint8_t>(rounded[c * n + i]))
                throw std::runtime_error("RGBA correctness check failed");
    }
    const char* label = interleaved ? (cropped ? "cropped vs RGBA" : "packed vs RGBA")
        : (cropped ? "cropped vs planar" : "packed vs planar");
    compare("RGBA8", width, height, label, ours, theirs, reset_ours, reset_theirs);
    check(frame, expected);
}

int main(void) {
#if defined(PARHELION_THREADED_BENCHMARK) || defined(PARHELION_METAL_BENCHMARK)
    std::cout << "vImage: native tiling enabled\n";
#endif
    std::cout << "layouts prepared outside timing\n";
    std::cout << benchmark_mode << "preallocated, buffers restored outside each timed call; median of 31 samples (us)\n"
              << std::left << std::setw(7) << "Format"
              << std::setw(13) << "Size" << std::setw(22) << "View"
              << std::right << std::setw(14) << "Parhelion"
              << std::setw(14) << "vImage" << std::setw(10) << "Speedup" << '\n'
              << std::fixed << std::setprecision(3);
    for (Coordinate size : {
        Coordinate(64, 64), Coordinate(256, 256), Coordinate(640, 480),
        Coordinate(1280, 720), Coordinate(1920, 1080),
        Coordinate(2560, 1440), Coordinate(3840, 2160)
    }) {
#if defined(PARHELION_THREADED_BENCHMARK) || defined(PARHELION_METAL_BENCHMARK)
        if (size.x < 1920)
            continue;
#endif
        for (bool cropped : {false, true}) {
            for (bool interleaved : {false, true})
                test(size.x, size.y, cropped, interleaved);
        }
    }
    return 0;
}
