#include "../../../threading/mode.hpp"
#include "parhelion-1.hpp"
#include <Accelerate/Accelerate.h>
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

void test(PixelFormat format, std::size_t channels, std::size_t width, std::size_t height, bool cropped) {
    const std::size_t frame_width = cropped ? width + 6 : width;
    const std::size_t frame_height = cropped ? height + 4 : height;
    const Coordinate xy(cropped ? 3 : 0, cropped ? 2 : 0);
    const std::size_t n = frame_width * frame_height;
    std::vector<std::byte> original(n * channels);
    for (std::size_t i = 0; i < original.size(); ++i)
        original[i] = static_cast<std::byte>((i * 7 + i / 251) % 256);

    Frame frame(frame_width, frame_height, format, FrameLayout::Planar, original);
    const auto view = cropped ? frame.view_mut(xy, width, height) : frame.view_mut();
    auto pixels = frame.view_mut().get_span();
    vImage_Buffer channel[4];
    for (std::size_t c = 0; c < channels; ++c)
        channel[c] = {view.get_span().data() + c * n, height, width, view.get_row_stride()};
    std::array<Pixel_8, 256> table[2];
    for (int value = 0; value < 256; ++value) {
        table[0][value] = static_cast<Pixel_8>(value > 127 ? 255 : 0);
        table[1][value] = static_cast<Pixel_8>(value > 128 ? 255 : 0);
    }
    auto ours = [&](int i) { PARHELION_BENCH_CALL(threshold, view, static_cast<std::uint8_t>(127 + i % 2)); };
    auto theirs = [&](int i) {
        for (std::size_t c = 0; c < (channels == 4 ? 3 : channels); ++c) {
            if (vImageTableLookUp_Planar8(&channel[c], &channel[c], table[i % 2].data(), (benchmark_threads == 1 ? kvImageDoNotTile : kvImageNoFlags)) != kvImageNoError)
                throw std::runtime_error("vImage threshold failed");
        }
    };

    for (int phase : {0, 1}) {
        auto expected = original;
        for (std::size_t c = 0; c < (channels == 4 ? 3 : channels); ++c) {
            for (std::size_t y = xy.y; y < xy.y + height; ++y) {
                for (std::size_t x = xy.x; x < xy.x + width; ++x) {
                    const std::size_t index = c * n + y * frame_width + x;
                    const int value = std::to_integer<int>(original[index]);
                    expected[index] = static_cast<std::byte>(value > 127 + phase ? 255 : 0);
                }
            }
        }
        std::copy(original.begin(), original.end(), pixels.begin());
        ours(phase);
        check(frame, expected);
        std::copy(original.begin(), original.end(), pixels.begin());
        theirs(phase);
        check(frame, expected);
    }
    std::copy(original.begin(), original.end(), pixels.begin());
    ours(0);
    const std::vector<std::byte> expected(pixels.begin(), pixels.end());
    const char* name = channels == 1 ? "Gray8" : channels == 3 ? "RGB8" : "RGBA8";
    compare(name, width, height, cropped ? "cropped" : "packed", ours, theirs);
    check(frame, expected);
}

int main(void) {
#if defined(PARHELION_THREADED_BENCHMARK) || defined(PARHELION_METAL_BENCHMARK)
    std::cout << "vImage: native tiling enabled\n";
#endif
    std::cout << "vImage LUTs precomputed outside timing\n";
    std::cout << benchmark_mode << "preallocated buffers, median of 31 samples (us)\n"
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
            test(PixelFormat::Gray, 1, size.x, size.y, cropped);
            test(PixelFormat::RGB, 3, size.x, size.y, cropped);
            test(PixelFormat::RGBA, 4, size.x, size.y, cropped);
        }
    }
    return 0;
}
