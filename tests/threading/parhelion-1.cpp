#include "parallel-parhelion-1.hpp"
#include <algorithm>
#include <stdexcept>

namespace {

Frame image(Coordinate size, PixelFormat format, FrameLayout layout = FrameLayout::Planar) {
    const auto width = size.x;
    const auto height = size.y;
    const auto channels = format == PixelFormat::Gray ? 1 : format == PixelFormat::RGB ? 3 : 4;
    std::vector<std::byte> data(width * height * channels);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = std::byte((i * 79 + i / 7) % 256);
    return Frame(width, height, format, layout, std::move(data));
}

void check(const Frame& expected, const Frame& actual) {
    if (!std::ranges::equal(expected.view_ref().get_span(), actual.view_ref().get_span()))
        throw std::runtime_error("parallel output differs from serial output");
}

void test(std::size_t workers, std::size_t height, bool cropped, Coordinate size = Coordinate(73, 11)) {
    ThreadPool pool(workers);
    const auto input = image(size, PixelFormat::RGBA);
    auto expected = image(size, PixelFormat::RGBA);
    std::ranges::fill(expected.view_mut().get_span(), std::byte{0});
    auto actual = expected;
    const Coordinate origin = cropped ? Coordinate(3, 2) : Coordinate(0, 0);
    const std::size_t width = cropped ? size.x - 8 : size.x;
    auto a = expected.view_mut(origin, width, height);
    auto b = actual.view_mut(origin, width, height);
    const auto src = input.view_ref(origin, width, height);

    copy(src, a);
    parallel_copy(pool, src, b);
    check(expected, actual);
    brighten(a, -20);
    parallel_brighten(pool, b, -20);
    check(expected, actual);
    brighten(a, 40, 20, -10);
    parallel_brighten(pool, b, 40, 20, -10);
    check(expected, actual);
    scale(a, 0.75f);
    parallel_scale(pool, b, 0.75f);
    check(expected, actual);
    scale(a, 1.5f, 0.5f, 1.25f);
    parallel_scale(pool, b, 1.5f, 0.5f, 1.25f);
    check(expected, actual);
    affine(a, 1.3f, -5.5f);
    parallel_affine(pool, b, 1.3f, -5.5f);
    check(expected, actual);
    affine(a, 0.5f, 1.25f, 1.5f, 15.0f, -3.0f, 9.0f);
    parallel_affine(pool, b, 0.5f, 1.25f, 1.5f, 15.0f, -3.0f, 9.0f);
    check(expected, actual);
    threshold(a, 100);
    parallel_threshold(pool, b, 100);
    check(expected, actual);
    copy(src, a);
    copy(src, b);
    threshold(a, 20, 100, 240);
    parallel_threshold(pool, b, 20, 100, 240);
    check(expected, actual);
    blend(src, expected.view_ref(origin, width, height), a, 0.3f);
    parallel_blend(pool, src, actual.view_ref(origin, width, height), b, 0.3f);
    check(expected, actual);
    blend(src, expected.view_ref(origin, width, height), a, 0.5f, 0.25f, 0.75f);
    parallel_blend(pool, src, actual.view_ref(origin, width, height), b, 0.5f, 0.25f, 0.75f);
    check(expected, actual);
    premultiply_alpha(a);
    parallel_premultiply_alpha(pool, b);
    check(expected, actual);

    auto gray_expected = image(size, PixelFormat::Gray, FrameLayout::Interleaved);
    auto gray_actual = gray_expected;
    auto gray_a = gray_expected.view_mut(origin, width, height);
    auto gray_b = gray_actual.view_mut(origin, width, height);
    grayscale(src, gray_a);
    parallel_grayscale(pool, src, gray_b);
    check(gray_expected, gray_actual);
    fill(gray_a, Color::Gray{std::byte{42}});
    parallel_fill(pool, gray_b, Color::Gray{std::byte{42}});
    check(gray_expected, gray_actual);
    const Color::RGB rgb{std::byte{1}, std::byte{2}, std::byte{3}};
    fill(a, rgb);
    parallel_fill(pool, b, rgb);
    check(expected, actual);
    const Color::RGBA rgba{std::byte{4}, std::byte{5}, std::byte{6}, std::byte{7}};
    fill(a, rgba);
    parallel_fill(pool, b, rgba);
    check(expected, actual);

    try {
        parallel_brighten(pool, b, 256);
        throw std::runtime_error("invalid brightness was accepted");
    } catch (const std::invalid_argument&) {
        check(expected, actual);
    }
    brighten(a, 1);
    parallel_brighten(pool, b, 1);
    check(expected, actual);
}

} // namespace

int main() {
    test(1, 11, false);
    test(3, 11, false);
    test(3, 7, true);
    test(8, 3, true);
    test(4, 2307, true, Coordinate(2569, 2311));
}
