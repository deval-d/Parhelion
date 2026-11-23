#include "image_types.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

std::pair<uint8_t, uint8_t> minmax_global(FrameViewRef frame);
std::pair<std::array<uint8_t, 3>, std::array<uint8_t, 3>> minmax_per_channel(FrameViewRef frame);

void check_value(const char* label, uint8_t received, uint8_t expected) {
    if (received != expected) {
        std::cerr << label << ": received " << static_cast<int>(received)
                  << ", expected " << static_cast<int>(expected) << '\n';
        throw std::runtime_error("test failed");
    }
}

void check_frame(const Frame& frame, const std::vector<std::byte>& expected) {
    const auto pixels = frame.view_ref().get_span();
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
    std::size_t width,
    std::size_t height,
    bool cropped,
    int value
) {
    const std::size_t frame_width = cropped ? width + 6 : width;
    const std::size_t frame_height = cropped ? height + 4 : height;
    const Coordinate xy(cropped ? 3 : 0, cropped ? 2 : 0);
    const std::size_t n = frame_width * frame_height;
    const std::size_t color_channels = channels == 4 ? 3 : channels;

    std::vector<std::byte> original(n * channels);
    for (std::size_t i = 0; i < original.size(); ++i)
        original[i] = i % 2 == 0 ? std::byte{0} : std::byte{255};
    for (std::size_t c = 0; c < color_channels; ++c) {
        for (std::size_t y = xy.y; y < xy.y + height; ++y) {
            for (std::size_t x = xy.x; x < xy.x + width; ++x) {
                const std::size_t index = c * n + y * frame_width + x;
                const int pixel = value < 0 ? 10 + c * 60 + (x * 7 + y * 13) % 40 : value;
                original[index] = static_cast<std::byte>(pixel);
            }
        }
        if (value == -2) {
            original[c * n + xy.y * frame_width + xy.x] = static_cast<std::byte>(200 + c);
            original[c * n + (xy.y + height - 1) * frame_width + xy.x + width - 1] = static_cast<std::byte>(1 + c);
        }
    }
    Frame frame(frame_width, frame_height, format, FrameLayout::Planar, original);
    const auto view = cropped ? frame.view_ref(xy, width, height) : frame.view_ref();

    std::array<uint8_t, 3> minima{255, 255, 255};
    std::array<uint8_t, 3> maxima{0, 0, 0};
    for (std::size_t c = 0; c < color_channels; ++c) {
        for (std::size_t y = xy.y; y < xy.y + height; ++y) {
            for (std::size_t x = xy.x; x < xy.x + width; ++x) {
                const std::size_t index = c * n + y * frame_width + x;
                const auto pixel = std::to_integer<uint8_t>(original[index]);
                minima[c] = std::min(minima[c], pixel);
                maxima[c] = std::max(maxima[c], pixel);
            }
        }
    }

    try {
        const auto result = minmax_global(view);
        check_value("global minimum", result.first, *std::min_element(minima.begin(), minima.end()));
        check_value("global maximum", result.second, *std::max_element(maxima.begin(), maxima.end()));
        if (channels > 1) {
            const auto result = minmax_per_channel(view);
            for (std::size_t c = 0; c < 3; ++c) {
                check_value("channel minimum", result.first[c], minima[c]);
                check_value("channel maximum", result.second[c], maxima[c]);
            }
        }
        check_frame(frame, original);
    } catch (const std::runtime_error&) {
        std::cerr << "minmax: " << width << " x " << height
                  << ", channels " << channels << ", cropped " << cropped
                  << ", value " << value << '\n';
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
        Coordinate(256, 1), Coordinate(257, 1), Coordinate(1023, 3),
        Coordinate(1024, 3), Coordinate(1025, 3)
    }) {
        for (bool cropped : {false, true}) {
            for (int value : {-2, -1, 0, 73, 255})
                test(format, channels, size.x, size.y, cropped, value);
        }
    }
}

template <typename Action>
void check_invalid(Frame& frame, Action action) {
    const auto pixels = frame.view_ref().get_span();
    const std::vector<std::byte> original(pixels.begin(), pixels.end());
    bool rejected = false;
    try {
        action(frame.view_ref());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected)
        throw std::runtime_error("invalid minmax input was accepted");
    check_frame(frame, original);
}

int invalid_inputs(void) {
    for (PixelFormat format : {PixelFormat::Gray, PixelFormat::RGB, PixelFormat::RGBA}) {
        const std::size_t channels = format == PixelFormat::Gray ? 1 : format == PixelFormat::RGB ? 3 : 4;
        Frame interleaved(17, 3, format, FrameLayout::Interleaved,
            std::vector<std::byte>(17 * 3 * channels, std::byte{100}));
        check_invalid(interleaved, [](FrameViewRef view) { minmax_global(view); });
        check_invalid(interleaved, [](FrameViewRef view) { minmax_per_channel(view); });
    }

    Frame gray(17, 3, PixelFormat::Gray, FrameLayout::Planar,
        std::vector<std::byte>(17 * 3, std::byte{100}));
    check_invalid(gray, [](FrameViewRef view) { minmax_per_channel(view); });

    for (Coordinate size : {Coordinate(0, 3), Coordinate(17, 0)}) {
        Frame empty(size.x, size.y, PixelFormat::RGB, FrameLayout::Planar, {});
        check_invalid(empty, [](FrameViewRef view) { minmax_global(view); });
        check_invalid(empty, [](FrameViewRef view) { minmax_per_channel(view); });
    }
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
