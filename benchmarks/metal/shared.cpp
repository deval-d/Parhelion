#include "metal-parhelion-1.hpp"
#include "parallel-parhelion-1.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {

void initialize(FrameViewMut view, std::uint32_t seed) {
    for (auto& byte : view.get_span()) {
        seed = seed * 1664525u + 1013904223u;
        byte = std::byte(seed >> 24);
    }
}

template <typename Serial, typename Parallel, typename Gpu>
void compare(const char* name, MetalFrame& output, Serial serial, Parallel parallel, Gpu gpu) {
    using Clock = std::chrono::steady_clock;
    const auto view = output.view_mut();
    const auto bytes = view.get_span();
    initialize(view, 123);
    const std::vector<std::byte> input(bytes.begin(), bytes.end());
    serial(view);
    const std::vector<std::byte> expected(bytes.begin(), bytes.end());

    const auto operations = std::tuple{
        [&] { serial(view); },
        [&] { parallel(view); },
        [&] { gpu(view); },
        [&] { gpu(output); }
    };
    constexpr int samples = 31;
    std::array<std::array<double, samples>, 4> times{};
    auto measure = [&](auto& operation, std::size_t index, int sample) {
        std::ranges::copy(input, bytes.begin());
        const auto start = Clock::now();
        operation();
        const auto end = Clock::now();
        if (sample >= 0)
            times[index][sample] = std::chrono::duration<double, std::micro>(end - start).count();
        if (sample == -3 || sample == samples - 1) {
            if (!std::ranges::equal(bytes, expected))
                throw std::runtime_error(std::string(name) + " output differs from serial");
        }
    };
    for (int sample = -3; sample < samples; ++sample) {
        if (sample % 2 == 0) {
            measure(std::get<0>(operations), 0, sample);
            measure(std::get<1>(operations), 1, sample);
            measure(std::get<2>(operations), 2, sample);
            measure(std::get<3>(operations), 3, sample);
        } else {
            measure(std::get<3>(operations), 3, sample);
            measure(std::get<2>(operations), 2, sample);
            measure(std::get<1>(operations), 1, sample);
            measure(std::get<0>(operations), 0, sample);
        }
    }

    std::cout << std::left << std::setw(21) << name << std::right;
    for (auto& timings : times) {
        std::ranges::sort(timings);
        std::cout << std::setw(14) << timings[samples / 2];
    }
    std::cout << '\n';
}

void benchmark(MetalContext& gpu, ThreadPool& pool, Coordinate size) {
    MetalFrame source(gpu, size.x, size.y, PixelFormat::RGBA, FrameLayout::Planar);
    MetalFrame second(gpu, size.x, size.y, PixelFormat::RGBA, FrameLayout::Planar);
    MetalFrame color(gpu, size.x, size.y, PixelFormat::RGBA, FrameLayout::Planar);
    MetalFrame gray(gpu, size.x, size.y, PixelFormat::Gray, FrameLayout::Planar);
    initialize(source.view_mut(), 42);
    initialize(second.view_mut(), 71);

    // each overload uses matching inputs, with explicit staging for ordinary views.
    const auto src = source.view_ref();
    const auto src2 = second.view_ref();
    const Color::RGBA value{std::byte{17}, std::byte{91}, std::byte{203}, std::byte{255}};

    std::cout << '\n' << size.x << 'x' << size.y << '\n'
              << std::left << std::setw(21) << "kernel" << std::right
              << std::setw(14) << "cpu serial" << std::setw(14) << "cpu pool"
              << std::setw(14) << "metal staged" << std::setw(14) << "metal shared" << '\n';

    compare("fill", color,
        [&](auto dst) { fill(dst, value); },
        [&](auto dst) { parallel_fill(pool, dst, value); },
        [&](auto&& dst) { metal_fill(gpu, dst, value); });
    compare("copy", color,
        [&](auto dst) { copy(src, dst); },
        [&](auto dst) { parallel_copy(pool, src, dst); },
        [&](auto&& dst) {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(dst)>, MetalFrame>)
                metal_copy(gpu, source, dst);
            else
                metal_copy(gpu, src, dst);
        });
    compare("brighten", color,
        [](auto dst) { brighten(dst, 20); },
        [&](auto dst) { parallel_brighten(pool, dst, 20); },
        [&](auto&& dst) { metal_brighten(gpu, dst, 20); });
    compare("scale", color,
        [](auto dst) { scale(dst, 1.25f); },
        [&](auto dst) { parallel_scale(pool, dst, 1.25f); },
        [&](auto&& dst) { metal_scale(gpu, dst, 1.25f); });
    compare("affine", color,
        [](auto dst) { affine(dst, 1.25f, -10.0f); },
        [&](auto dst) { parallel_affine(pool, dst, 1.25f, -10.0f); },
        [&](auto&& dst) { metal_affine(gpu, dst, 1.25f, -10.0f); });
    compare("threshold", color,
        [](auto dst) { threshold(dst, 127); },
        [&](auto dst) { parallel_threshold(pool, dst, 127); },
        [&](auto&& dst) { metal_threshold(gpu, dst, 127); });
    compare("blend", color,
        [&](auto dst) { blend(src, src2, dst, 0.375f); },
        [&](auto dst) { parallel_blend(pool, src, src2, dst, 0.375f); },
        [&](auto&& dst) {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(dst)>, MetalFrame>)
                metal_blend(gpu, source, second, dst, 0.375f);
            else
                metal_blend(gpu, src, src2, dst, 0.375f);
        });
    compare("grayscale", gray,
        [&](auto dst) { grayscale(src, dst); },
        [&](auto dst) { parallel_grayscale(pool, src, dst); },
        [&](auto&& dst) {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(dst)>, MetalFrame>)
                metal_grayscale(gpu, source, dst);
            else
                metal_grayscale(gpu, src, dst);
        });
    compare("premultiply_alpha", color,
        [](auto dst) { premultiply_alpha(dst); },
        [&](auto dst) { parallel_premultiply_alpha(pool, dst); },
        [&](auto&& dst) { metal_premultiply_alpha(gpu, dst); });
}

} // namespace

int main() {
    MetalContext gpu;
    ThreadPool pool(8);
    std::cout << "packed rgba8, median of 31 calls (us), 3 warmups\n"
              << "cpu pool capacity 8, adaptive dispatch; all calls synchronous\n"
              << "frame setup and reset excluded; staged metal includes allocation and copies\n"
              << std::fixed << std::setprecision(2);
    for (Coordinate size : {Coordinate(256, 256), Coordinate(1920, 1080), Coordinate(3840, 2160)})
        benchmark(gpu, pool, size);
    std::cout << "\nall outputs matched serial cpu output\n";
}
