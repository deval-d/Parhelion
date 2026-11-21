#include "image_types.hpp"
#include "color_types.hpp"
#include <iostream>
#include <stdexcept>

void fill(FrameViewMut frame, Color::Gray color);
void fill(FrameViewMut frame, Color::RGB color);
void fill(FrameViewMut frame, Color::RGBA color);

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

template <typename ColorType>
void test(
    PixelFormat format,
    FrameLayout layout,
    ColorType color,
    const std::vector<std::byte>& values,
    std::size_t width,
    std::size_t height,
    bool cropped
) {
    const std::size_t frame_width = cropped ? width + 6 : width;
    const std::size_t frame_height = cropped ? height + 4 : height;
    const Coordinate xy(cropped ? 3 : 0, cropped ? 2 : 0);
    const std::size_t n = frame_width * frame_height;
    const std::size_t channels = values.size();

    std::vector<std::byte> expected(n * channels, std::byte{171});
    Frame frame(frame_width, frame_height, format, layout, expected);

    if (cropped)
        fill(frame.view_mut(xy, width, height), color);
    else
        fill(frame.view_mut(), color);

    // outside the crop should stay same 
    for (std::size_t y = xy.y; y < xy.y + height; ++y) {
        for (std::size_t x = xy.x; x < xy.x + width; ++x) {
            const std::size_t pixel = y * frame_width + x;
            for (std::size_t c = 0; c < channels; ++c) {
                const std::size_t index = layout == FrameLayout::Planar
                    ? c * n + pixel
                    : pixel * channels + c;
                expected[index] = values[c];
            }
        }
    }

    try {
        check_frame(frame, expected);
    } catch (const std::runtime_error&) {
        std::cerr << "fill: " << width << " x " << height
                  << ", channels " << channels
                  << ", cropped " << cropped << '\n';
        throw;
    }
}

template <typename ColorType>
void test_sizes(
    PixelFormat format,
    FrameLayout layout,
    ColorType color,
    const std::vector<std::byte>& values
) {
    for (Coordinate size : {
        Coordinate(3, 2), Coordinate(2047, 1), Coordinate(2048, 1),
        Coordinate(2049, 1), Coordinate(127, 19), Coordinate(255, 9),
        Coordinate(256, 9), Coordinate(257, 9), Coordinate(1025, 3)
    }) {
        test(format, layout, color, values, size.x, size.y, false);
        test(format, layout, color, values, size.x, size.y, true);
    }
}

int gray8(void) {
    for (auto layout : {FrameLayout::Interleaved, FrameLayout::Planar}) {
        for (int value : {0, 53, 255}) {
            const auto byte = static_cast<std::byte>(value);
            test_sizes(PixelFormat::Gray, layout, Color::Gray{byte}, {byte});
        }
    }
    return 0;
}

int rgb8(void) {
    test_sizes(PixelFormat::RGB, FrameLayout::Planar,
        Color::RGB{std::byte{0}, std::byte{53}, std::byte{255}},
        {std::byte{0}, std::byte{53}, std::byte{255}});
    test_sizes(PixelFormat::RGB, FrameLayout::Planar,
        Color::RGB{}, {std::byte{0}, std::byte{0}, std::byte{0}});
    return 0;
}

int rgba8(void) {
    test_sizes(PixelFormat::RGBA, FrameLayout::Planar,
        Color::RGBA{std::byte{0}, std::byte{53}, std::byte{211}, std::byte{255}},
        {std::byte{0}, std::byte{53}, std::byte{211}, std::byte{255}});
    test_sizes(PixelFormat::RGBA, FrameLayout::Planar,
        Color::RGBA{}, {std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0}});
    return 0;
}

int main(void) {
    int status = 0;
    status |= gray8();
    status |= rgb8();
    status |= rgba8();
    return status;
}
