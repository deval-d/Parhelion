#include "image_types.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void blend(FrameViewRef src1, FrameViewRef src2, FrameViewMut dst, float alpha);
void blend(FrameViewRef src1, FrameViewRef src2, FrameViewMut dst, float r_alpha, float g_alpha, float b_alpha);

void check_frame(const Frame& frame, const std::vector<std::byte>& expected) {
    const auto pixels = frame.view_ref().get_span();
    if (pixels.size() != expected.size()) {
        std::cerr << "buffer size: received " << pixels.size()
                  << ", expected " << expected.size() << '\n';
        throw std::runtime_error("test failed");
    }

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (pixels[i] != expected[i]) {
            std::cerr << "byte " << i
                      << ": received " << std::to_integer<int>(pixels[i])
                      << ", expected " << std::to_integer<int>(expected[i]) << '\n';
            throw std::runtime_error("test failed");
        }
    }
}

void test(
    PixelFormat format,
    std::size_t channels,
    const std::vector<float>& alphas,
    std::size_t width,
    std::size_t height,
    bool cropped,
    int in_place = 0
) {
    const std::size_t src1_width = cropped ? width + 6 : width;
    const std::size_t src2_width = cropped ? width + 10 : width;
    const std::size_t dst_width = cropped ? width + 14 : width;
    const std::size_t frame_height = cropped ? height + 4 : height;
    const Coordinate xy(cropped ? 3 : 0, cropped ? 2 : 0);
    const std::size_t n1 = src1_width * frame_height;
    const std::size_t n2 = src2_width * frame_height;

    std::vector<std::byte> a(n1 * channels);
    std::vector<std::byte> b(n2 * channels);
    std::vector<std::byte> original(dst_width * frame_height * channels, std::byte{79});
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<std::byte>((i * 7 + i / 251 + 13) % 256);
    for (std::size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<std::byte>((i * 13 + i / 127 + 17) % 256);
    Frame source1(src1_width, frame_height, format, FrameLayout::Planar, a);
    Frame source2(src2_width, frame_height, format, FrameLayout::Planar, b);
    Frame frame(dst_width, frame_height, format, FrameLayout::Planar, original);
    Frame& dst = in_place == 1 ? source1 : in_place == 2 ? source2 : frame;
    auto expected = in_place == 1 ? a : in_place == 2 ? b : original;
    const std::size_t frame_width = dst.get_width();
    const std::size_t n = frame_width * frame_height;
    const auto src1 = cropped ? source1.view_ref(xy, width, height) : source1.view_ref();
    const auto src2 = cropped ? source2.view_ref(xy, width, height) : source2.view_ref();
    const auto view = cropped ? dst.view_mut(xy, width, height) : dst.view_mut();

    if (alphas.size() == 1)
        blend(src1, src2, view, alphas[0]);
    else
        blend(src1, src2, view, alphas[0], alphas[1], alphas[2]);

    const std::size_t color_channels = channels == 4 ? 3 : channels;
    for (std::size_t c = 0; c < color_channels; ++c) {
        const float alpha = alphas.size() == 1 ? alphas[0] : alphas[c];
        const float beta = 1.0f - alpha;
        for (std::size_t y = xy.y; y < xy.y + height; ++y) {
            for (std::size_t x = xy.x; x < xy.x + width; ++x) {
                const float v1 = static_cast<float>(std::to_integer<int>(a[c * n1 + y * src1_width + x]));
                const float v2 = static_cast<float>(std::to_integer<int>(b[c * n2 + y * src2_width + x]));
                const float first = std::fma(v1, alpha, 0.0f);
                const float value = std::fma(v2, beta, first);
                expected[c * n + y * frame_width + x] = static_cast<std::byte>(value);
            }
        }
    }

    try {
        check_frame(dst, expected);
        if (in_place != 1)
            check_frame(source1, a);
        if (in_place != 2)
            check_frame(source2, b);
    } catch (const std::runtime_error&) {
        std::cerr << "blend: " << width << " x " << height
                  << ", channels " << channels << ", cropped " << cropped
                  << ", in place " << in_place << ", alphas";
        for (float alpha : alphas)
            std::cerr << ' ' << alpha;
        std::cerr << '\n';
        throw;
    }
}

void test_sizes(PixelFormat format, std::size_t channels) {
    for (Coordinate size : {
        Coordinate(1, 1), Coordinate(3, 2), Coordinate(15, 1),
        Coordinate(16, 1), Coordinate(17, 1), Coordinate(63, 1),
        Coordinate(64, 1), Coordinate(65, 1), Coordinate(127, 1),
        Coordinate(128, 1), Coordinate(129, 1), Coordinate(191, 1),
        Coordinate(192, 1), Coordinate(193, 1), Coordinate(255, 1),
        Coordinate(256, 1), Coordinate(257, 1), Coordinate(1025, 3)
    }) {
        for (bool cropped : {false, true}) {
            for (float alpha : {0.0f, 0.1f, 0.25f, 0.5f, 0.9f, 1.0f})
                test(format, channels, {alpha}, size.x, size.y, cropped);
            if (channels > 1) {
                test(format, channels, {0.25f, 0.5f, 0.75f}, size.x, size.y, cropped);
                test(format, channels, {0.0f, 0.1f, 1.0f}, size.x, size.y, cropped);
            }
        }
    }

    for (bool cropped : {false, true}) {
        for (int in_place : {1, 2}) {
            test(format, channels, {0.3f}, 257, 3, cropped, in_place);
            if (channels > 1)
                test(format, channels, {0.0f, 0.5f, 1.0f}, 257, 3, cropped, in_place);
        }
    }
}

void check_invalid(FrameViewRef src1, FrameViewRef src2, Frame& dst, const std::vector<float>& alphas) {
    const auto pixels = dst.view_ref().get_span();
    const std::vector<std::byte> original(pixels.begin(), pixels.end());
    bool rejected = false;
    try {
        if (alphas.size() == 1)
            blend(src1, src2, dst.view_mut(), alphas[0]);
        else
            blend(src1, src2, dst.view_mut(), alphas[0], alphas[1], alphas[2]);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected)
        throw std::runtime_error("invalid blend input was accepted");
    check_frame(dst, original);
}

int invalid_inputs(void) {
    Frame rgb(17, 3, PixelFormat::RGB, FrameLayout::Planar,
        std::vector<std::byte>(17 * 3 * 3, std::byte{100}));
    Frame dst(17, 3, PixelFormat::RGB, FrameLayout::Planar,
        std::vector<std::byte>(17 * 3 * 3, std::byte{79}));
    for (float alpha : {-0.1f, 1.1f, std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        check_invalid(rgb.view_ref(), rgb.view_ref(), dst, {alpha});
        check_invalid(rgb.view_ref(), rgb.view_ref(), dst, {alpha, 0.5f, 0.75f});
        check_invalid(rgb.view_ref(), rgb.view_ref(), dst, {0.25f, alpha, 0.75f});
        check_invalid(rgb.view_ref(), rgb.view_ref(), dst, {0.25f, 0.5f, alpha});
    }

    Frame wide(18, 3, PixelFormat::RGB, FrameLayout::Planar,
        std::vector<std::byte>(18 * 3 * 3, std::byte{100}));
    Frame tall(17, 4, PixelFormat::RGB, FrameLayout::Planar,
        std::vector<std::byte>(17 * 4 * 3, std::byte{100}));
    Frame rgba(17, 3, PixelFormat::RGBA, FrameLayout::Planar,
        std::vector<std::byte>(17 * 3 * 4, std::byte{100}));
    Frame interleaved(17, 3, PixelFormat::RGB, FrameLayout::Interleaved,
        std::vector<std::byte>(17 * 3 * 3, std::byte{100}));
    for (Frame* invalid : {&wide, &tall, &rgba, &interleaved}) {
        for (const std::vector<float>& alphas : {std::vector<float>{0.5f}, std::vector<float>{0.25f, 0.5f, 0.75f}}) {
            check_invalid(invalid->view_ref(), rgb.view_ref(), dst, alphas);
            check_invalid(rgb.view_ref(), invalid->view_ref(), dst, alphas);
            check_invalid(rgb.view_ref(), rgb.view_ref(), *invalid, alphas);
        }
    }

    Frame gray(17, 3, PixelFormat::Gray, FrameLayout::Planar,
        std::vector<std::byte>(17 * 3, std::byte{100}));
    check_invalid(gray.view_ref(), gray.view_ref(), gray, {0.25f, 0.5f, 0.75f});
    return 0;
}

int gray8(void) {
    test_sizes(PixelFormat::Gray, 1);
    return 0;
}

int rgb8(void) {
    test_sizes(PixelFormat::RGB, 3);
    return 0;
}

int rgba8(void) {
    test_sizes(PixelFormat::RGBA, 4);
    return 0;
}

int main(void) {
    int status = 0;
    status |= gray8();
    status |= rgb8();
    status |= rgba8();
    status |= invalid_inputs();
    return status;
}
