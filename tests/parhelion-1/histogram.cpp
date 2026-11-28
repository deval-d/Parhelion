#include "parhelion-1.hpp"
#include <array>
#include <iostream>
#include <stdexcept>

void check_histogram(
    const std::array<std::uint64_t, 256>& received,
    const std::array<std::uint64_t, 256>& expected,
    std::size_t channel
) {
    for (std::size_t i = 0; i < 256; ++i) {
        if (received[i] != expected[i]) {
            std::cerr << "channel " << channel << ", bin " << i
                      << ": received " << received[i]
                      << ", expected " << expected[i] << '\n';
            throw std::runtime_error("test failed");
        }
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
    std::vector<std::byte> original(n * channels, std::byte{251});
    for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t y = xy.y; y < xy.y + height; ++y) {
            for (std::size_t x = xy.x; x < xy.x + width; ++x) {
                const std::size_t index = c * n + y * frame_width + x;
                const int pixel = value < 0 ? (x * 37 + y * 13 + c * 53) % 256 : value;
                original[index] = static_cast<std::byte>(pixel);
            }
        }
    }
    Frame frame(frame_width, frame_height, format, FrameLayout::Planar, original);
    const auto view = cropped ? frame.view_ref(xy, width, height) : frame.view_ref();
    std::array<std::array<std::uint64_t, 256>, 4> expected{};
    for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t y = xy.y; y < xy.y + height; ++y) {
            for (std::size_t x = xy.x; x < xy.x + width; ++x) {
                const std::size_t index = c * n + y * frame_width + x;
                ++expected[c][std::to_integer<unsigned>(original[index])];
            }
        }
    }

    try {
        switch (format) {
            case PixelFormat::Gray:
                check_histogram(histogram_gray(view), expected[0], 0);
                break;

            case PixelFormat::RGB: {
                const auto result = histogram_rgb(view);
                for (std::size_t c = 0; c < 3; ++c)
                    check_histogram(result[c], expected[c], c);
                break;
            }

            case PixelFormat::RGBA: {
                const auto result = histogram_rgba(view);
                for (std::size_t c = 0; c < 4; ++c)
                    check_histogram(result[c], expected[c], c);
                break;
            }
        }
        check_frame(frame, original);
    } catch (const std::runtime_error&) {
        std::cerr << "histogram: " << width << " x " << height
                  << ", channels " << channels << ", cropped " << cropped
                  << ", value " << value << '\n';
        throw;
    }
}

void test_sizes(PixelFormat format, std::size_t channels) {
    for (Coordinate size : {
        Coordinate(1, 1), Coordinate(2, 1), Coordinate(3, 1),
        Coordinate(4, 1), Coordinate(5, 1), Coordinate(63, 1),
        Coordinate(64, 1), Coordinate(65, 1), Coordinate(127, 1),
        Coordinate(128, 1), Coordinate(129, 1), Coordinate(255, 1),
        Coordinate(256, 1), Coordinate(257, 1), Coordinate(17, 5),
        Coordinate(1023, 3), Coordinate(1024, 3), Coordinate(1025, 3)
    }) {
        for (bool cropped : {false, true}) {
            for (int value : {-1, 0, 73, 255})
                test(format, channels, size.x, size.y, cropped, value);
        }
    }
}

template <typename Action>
void check_invalid(FrameViewRef view, Action action) {
    const auto pixels = view.get_span();
    const std::vector<std::byte> original(pixels.begin(), pixels.end());
    bool rejected = false;
    try {
        action(view);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected)
        throw std::runtime_error("invalid histogram input was accepted");
    for (std::size_t i = 0; i < original.size(); ++i) {
        if (pixels[i] != original[i])
            throw std::runtime_error("invalid histogram input was modified");
    }
}

int invalid_inputs(void) {
    for (PixelFormat format : {PixelFormat::Gray, PixelFormat::RGB, PixelFormat::RGBA}) {
        const std::size_t channels = format == PixelFormat::Gray ? 1 : format == PixelFormat::RGB ? 3 : 4;
        for (FrameLayout layout : {FrameLayout::Planar, FrameLayout::Interleaved}) {
            Frame frame(17, 3, format, layout, std::vector<std::byte>(17 * 3 * channels, std::byte{100}));
            const auto view = frame.view_ref();
            if (layout != FrameLayout::Planar || format != PixelFormat::Gray)
                check_invalid(view, [](FrameViewRef view) { histogram_gray(view); });
            if (layout != FrameLayout::Planar || format != PixelFormat::RGB)
                check_invalid(view, [](FrameViewRef view) { histogram_rgb(view); });
            if (layout != FrameLayout::Planar || format != PixelFormat::RGBA)
                check_invalid(view, [](FrameViewRef view) { histogram_rgba(view); });

            if (layout == FrameLayout::Planar) {
                for (Coordinate size : {Coordinate(0, 3), Coordinate(17, 0)}) {
                    bool rejected = false;
                    try {
                        FrameViewRef empty(size.x, size.y, view.get_row_stride(), view.get_channel_stride(),
                            format, layout, view.get_span());
                    } catch (const std::invalid_argument&) {
                        rejected = true;
                    }
                    if (!rejected)
                        throw std::runtime_error("empty view was accepted");
                }
            }
        }
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
