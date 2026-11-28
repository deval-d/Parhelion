#include "parhelion-1.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

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
    bool cropped
) {
    const std::size_t frame_width = cropped ? width + 6 : width;
    const std::size_t frame_height = cropped ? height + 4 : height;
    const Coordinate xy(cropped ? 3 : 0, cropped ? 2 : 0);
    const std::size_t n = frame_width * frame_height;

    std::vector<std::byte> expected(n * channels);
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = static_cast<std::byte>((i * 7 + i / 251 + 13) % 256);
    Frame frame(frame_width, frame_height, format, FrameLayout::Planar, expected);
    const auto view = cropped
        ? frame.view_mut(xy, width, height) : frame.view_mut();

    if (alphas.size() == 1)
        scale(view, alphas[0]);
    else
        scale(view, alphas[0], alphas[1], alphas[2]);

    // alpha and pixels outside crop should stay same
    const std::size_t color_channels = channels == 4 ? 3 : channels;
    for (std::size_t c = 0; c < color_channels; ++c) {
        const float alpha = alphas.size() == 1 ? alphas[0] : alphas[c];
        for (std::size_t y = xy.y; y < xy.y + height; ++y) {
            for (std::size_t x = xy.x; x < xy.x + width; ++x) {
                const std::size_t index = c * n + y * frame_width + x;
                const float value = std::to_integer<int>(expected[index]) * alpha;
                expected[index] = static_cast<std::byte>(std::min(value, 255.0f));
            }
        }
    }

    try {
        check_frame(frame, expected);
    } catch (const std::runtime_error&) {
        std::cerr << "scale: " << width << " x " << height
                  << ", channels " << channels << ", cropped " << cropped
                  << ", alphas";
        for (float alpha : alphas)
            std::cerr << ' ' << alpha;
        std::cerr << '\n';
        throw;
    }
}

void test_sizes(PixelFormat format, std::size_t channels) {
    for (Coordinate size : {
        Coordinate(1, 1), Coordinate(3, 2), Coordinate(63, 1),
        Coordinate(64, 1), Coordinate(65, 1), Coordinate(255, 1),
        Coordinate(256, 1), Coordinate(257, 1), Coordinate(1025, 3)
    }) {
        for (bool cropped : {false, true}) {
            for (float alpha : {0.0f, 0.5f, 1.0f, 1.5f, 3.25f, std::numeric_limits<float>::max()})
                test(format, channels, {alpha}, size.x, size.y, cropped);
            if (channels > 1) {
                test(format, channels, {0.5f, 1.5f, 2.0f}, size.x, size.y, cropped);
                test(format, channels, {0.0f, 1.0f, 3.25f}, size.x, size.y, cropped);
            }
        }
    }
}

template <typename Action>
void check_invalid(Frame& frame, Action action) {
    const auto pixels = frame.view_ref().get_span();
    const std::vector<std::byte> original(pixels.begin(), pixels.end());
    bool rejected = false;
    try {
        action(frame.view_mut());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected)
        throw std::runtime_error("invalid scale input was accepted");
    check_frame(frame, original);
}

int invalid_inputs(void) {
    Frame rgb(17, 3, PixelFormat::RGB, FrameLayout::Planar,
        std::vector<std::byte>(17 * 3 * 3, std::byte{100}));
    for (float alpha : {-1.0f, std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        check_invalid(rgb, [&](FrameViewMut view) { scale(view, alpha); });
        check_invalid(rgb, [&](FrameViewMut view) { scale(view, alpha, 1.0f, 2.0f); });
        check_invalid(rgb, [&](FrameViewMut view) { scale(view, 1.0f, alpha, 2.0f); });
        check_invalid(rgb, [&](FrameViewMut view) { scale(view, 1.0f, 2.0f, alpha); });
    }

    Frame interleaved(17, 3, PixelFormat::RGB, FrameLayout::Interleaved,
        std::vector<std::byte>(17 * 3 * 3, std::byte{100}));
    check_invalid(interleaved, [](FrameViewMut view) { scale(view, 1.5f); });
    check_invalid(interleaved, [](FrameViewMut view) { scale(view, 0.5f, 1.0f, 2.0f); });

    Frame gray(17, 3, PixelFormat::Gray, FrameLayout::Planar,
        std::vector<std::byte>(17 * 3, std::byte{100}));
    check_invalid(gray, [](FrameViewMut view) { scale(view, 0.5f, 1.0f, 2.0f); });
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
