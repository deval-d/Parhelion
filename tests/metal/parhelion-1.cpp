#include "metal-parhelion-1.hpp"
#include <algorithm>
#include <limits>
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
        throw std::runtime_error("metal output differs from serial output");
}

void test(MetalContext& context, std::size_t height, bool cropped, Coordinate size) {
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
    metal_copy(context, src, b);
    check(expected, actual);
    brighten(a, -20);
    metal_brighten(context, b, -20);
    check(expected, actual);
    brighten(a, 40, 20, -10);
    metal_brighten(context, b, 40, 20, -10);
    check(expected, actual);
    scale(a, 0.75f);
    metal_scale(context, b, 0.75f);
    check(expected, actual);
    scale(a, 1.5f, 0.5f, 1.25f);
    metal_scale(context, b, 1.5f, 0.5f, 1.25f);
    check(expected, actual);
    affine(a, 1.3f, -5.5f);
    metal_affine(context, b, 1.3f, -5.5f);
    check(expected, actual);
    affine(a, 0.5f, 1.25f, 1.5f, 15.0f, -3.0f, 9.0f);
    metal_affine(context, b, 0.5f, 1.25f, 1.5f, 15.0f, -3.0f, 9.0f);
    check(expected, actual);
    threshold(a, 100);
    metal_threshold(context, b, 100);
    check(expected, actual);
    copy(src, a);
    copy(src, b);
    threshold(a, 20, 100, 240);
    metal_threshold(context, b, 20, 100, 240);
    check(expected, actual);
    blend(src, expected.view_ref(origin, width, height), a, 0.3f);
    metal_blend(context, src, actual.view_ref(origin, width, height), b, 0.3f);
    check(expected, actual);
    blend(src, expected.view_ref(origin, width, height), a, 0.5f, 0.25f, 0.75f);
    metal_blend(context, src, actual.view_ref(origin, width, height), b, 0.5f, 0.25f, 0.75f);
    check(expected, actual);
    premultiply_alpha(a);
    metal_premultiply_alpha(context, b);
    check(expected, actual);

    auto gray_expected = image(size, PixelFormat::Gray, FrameLayout::Interleaved);
    auto gray_actual = gray_expected;
    auto gray_a = gray_expected.view_mut(origin, width, height);
    auto gray_b = gray_actual.view_mut(origin, width, height);
    grayscale(src, gray_a);
    metal_grayscale(context, src, gray_b);
    check(gray_expected, gray_actual);
    fill(gray_a, Color::Gray{std::byte{42}});
    metal_fill(context, gray_b, Color::Gray{std::byte{42}});
    check(gray_expected, gray_actual);
    const Color::RGB rgb{std::byte{1}, std::byte{2}, std::byte{3}};
    fill(a, rgb);
    metal_fill(context, b, rgb);
    check(expected, actual);
    const Color::RGBA rgba{std::byte{4}, std::byte{5}, std::byte{6}, std::byte{7}};
    fill(a, rgba);
    metal_fill(context, b, rgba);
    check(expected, actual);

    try {
        metal_brighten(context, b, 256);
        throw std::runtime_error("invalid brightness was accepted");
    } catch (const std::invalid_argument&) {
        check(expected, actual);
    }
    brighten(a, 1);
    metal_brighten(context, b, 1);
    check(expected, actual);
}

void test_layouts(MetalContext& context) {
    const Coordinate size(45, 19);
    const Coordinate src_origin(3, 2);
    const Coordinate dst_origin(5, 4);
    const auto input = image(size, PixelFormat::RGBA, FrameLayout::Interleaved);
    auto expected = image(size, PixelFormat::RGBA, FrameLayout::Interleaved);
    std::ranges::fill(expected.view_mut().get_span(), std::byte{0});
    auto actual = expected;

    const auto src = input.view_ref(src_origin, 33, 13);
    copy(src, expected.view_mut(dst_origin, 33, 13));
    metal_copy(context, src, actual.view_mut(dst_origin, 33, 13));
    check(expected, actual);

    auto gray_expected = image(size, PixelFormat::Gray);
    auto gray_actual = gray_expected;
    const auto a = gray_expected.view_mut(src_origin, 33, 13);
    const auto b = gray_actual.view_mut(src_origin, 33, 13);
    constexpr auto largest = std::numeric_limits<float>::max();
    scale(a, largest);
    metal_scale(context, b, largest);
    check(gray_expected, gray_actual);
    affine(a, largest, -largest);
    metal_affine(context, b, largest, -largest);
    check(gray_expected, gray_actual);
}

void test_shared_storage(MetalContext& context) {
    const Coordinate size(73, 11);
    auto expected = image(size, PixelFormat::RGBA);
    MetalFrame frame(context, size.x, size.y, PixelFormat::RGBA, FrameLayout::Planar);
    MetalFrame output(context, size.x, size.y, PixelFormat::RGBA, FrameLayout::Planar);

    const auto pixels = frame.view_mut().get_span();
    std::ranges::copy(expected.view_ref().get_span(), pixels.begin());
    brighten(expected.view_mut(), 31);
    metal_brighten(context, frame, 31);
    if (!std::ranges::equal(pixels, expected.view_ref().get_span()))
        throw std::runtime_error("metal did not update the borrowed cpu view");

    scale(frame.view_mut(), 0.75f);
    scale(expected.view_mut(), 0.75f);
    metal_copy(context, frame, output);
    if (!std::ranges::equal(output.view_ref().get_span(), expected.view_ref().get_span()))
        throw std::runtime_error("metal did not read the cpu update");

    blend(expected.view_ref(), expected.view_ref(), expected.view_mut(), 0.25f);
    metal_blend(context, frame, output, output, 0.25f);
    if (!std::ranges::equal(output.view_ref().get_span(), expected.view_ref().get_span()))
        throw std::runtime_error("shared-frame blend differs from serial output");
}

} // namespace

int main() {
    MetalContext context;
    test(context, 11, false, Coordinate(73, 11));
    test(context, 7, true, Coordinate(73, 11));
    test(context, 1, false, Coordinate(1, 1));
    test(context, 129, false, Coordinate(257, 129));
    test_layouts(context);
    test_shared_storage(context);
}
