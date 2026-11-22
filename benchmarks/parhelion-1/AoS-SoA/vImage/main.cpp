#include <cstdint>
// Benchmark the current internal kernels with preallocated output buffers.
#include "../../../../src/AoS_SoA.cpp"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

using Clock = std::chrono::steady_clock;

template <typename Convert>
double measure(Convert convert, int repeats) {
    const auto start = Clock::now();
    for (int i = 0; i < repeats; ++i)
        convert();
    const auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / repeats;
}

template <typename Parhelion, typename Library>
void compare(
    const char* format,
    std::size_t width,
    std::size_t height,
    const char* direction,
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
              << std::setw(13) << size << std::setw(15) << direction
              << std::right << std::setw(14) << ours[15]
              << std::setw(14) << theirs[15]
              << std::setw(10) << theirs[15] / ours[15] << '\n';
}

void check(FrameViewRef view, const std::vector<std::byte>& expected) {
    const auto pixels = view.get_span();
    if (!std::equal(pixels.begin(), pixels.end(), expected.begin(), expected.end()))
        throw std::runtime_error("incorrect conversion");
}

void test(PixelFormat format, std::size_t channels, std::size_t width, std::size_t height) {
    const std::size_t n = width * height;
    std::vector<std::byte> original(n * channels);
    std::vector<std::byte> expected(n * channels);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < channels; ++c) {
            const auto value = static_cast<std::byte>((i * 7 + i / 251 + c * 53) % 256);
            original[i * channels + c] = value;
            expected[c * n + i] = value;
        }
    }

    Frame image(width, height, format, FrameLayout::Interleaved, original);
    Frame planes(width, height, format, FrameLayout::Planar,
        std::vector<std::byte>(n * channels));
    const auto image_ref = image.view_ref();
    const auto image_mut = image.view_mut();
    const auto planes_ref = planes.view_ref();
    const auto planes_mut = planes.view_mut();

    auto split = [&] {
        if (format == PixelFormat::RGB)
            rgb_to_planar(image_ref, planes_mut);
        else
            rgba_to_planar(image_ref, planes_mut);
    };
    auto merge = [&] {
        if (format == PixelFormat::RGB)
            rgb_to_interleaved(planes_ref, image_mut);
        else
            rgba_to_interleaved(planes_ref, image_mut);
    };

    vImage_Buffer pixels{image_mut.get_span().data(), height, width, width * channels};
    vImage_Buffer channel[4];
    for (std::size_t c = 0; c < channels; ++c)
        channel[c] = {planes_mut.get_span().data() + c * n, height, width, width};

    auto library_split = [&] {
        // ARGB routines preserve byte positions, so channel[0..3] holds RGBA here.
        const auto error = channels == 3
            ? vImageConvert_RGB888toPlanar8(&pixels, &channel[0], &channel[1], &channel[2], kvImageDoNotTile)
            : vImageConvert_ARGB8888toPlanar8(&pixels, &channel[0], &channel[1], &channel[2], &channel[3], kvImageDoNotTile);
        if (error != kvImageNoError)
            throw std::runtime_error("vImage split failed");
    };
    auto library_merge = [&] {
        const auto error = channels == 3
            ? vImageConvert_Planar8toRGB888(&channel[0], &channel[1], &channel[2], &pixels, kvImageDoNotTile)
            : vImageConvert_Planar8toARGB8888(&channel[0], &channel[1], &channel[2], &channel[3], &pixels, kvImageDoNotTile);
        if (error != kvImageNoError)
            throw std::runtime_error("vImage merge failed");
    };

    // Check both implementations before timing, including each channel's order.
    for (bool ours : {true, false}) {
        std::fill(planes_mut.get_span().begin(), planes_mut.get_span().end(), std::byte{0});
        if (ours) split();
        else library_split();
        check(planes_ref, expected);

        std::fill(image_mut.get_span().begin(), image_mut.get_span().end(), std::byte{0});
        if (ours) merge();
        else library_merge();
        check(image_ref, original);
    }

    const char* name = channels == 3 ? "RGB8" : "RGBA8";
    compare(name, width, height, "AoS -> SoA", split, library_split);
    compare(name, width, height, "SoA -> AoS", merge, library_merge);
    check(planes_ref, expected);
    check(image_ref, original);
}

int main(void) {

    std::cout << "Single thread, preallocated buffers, median of 31 samples (us)\n"
              << std::left << std::setw(7) << "Format"
              << std::setw(13) << "Size" << std::setw(15) << "Conversion"
              << std::right << std::setw(14) << "Parhelion"
              << std::setw(14) << "vImage" << std::setw(10) << "Speedup" << '\n'
              << std::fixed << std::setprecision(3);

    for (Coordinate size : {
        Coordinate(64, 64), Coordinate(256, 256), Coordinate(640, 480),
        Coordinate(1280, 720), Coordinate(1920, 1080),
        Coordinate(2560, 1440), Coordinate(3840, 2160)
    }) {
        test(PixelFormat::RGB, 3, size.x, size.y);
        test(PixelFormat::RGBA, 4, size.x, size.y);
    }
    return 0;
}
