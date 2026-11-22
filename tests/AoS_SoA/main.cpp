#include "AoS_SoA.hpp"
#include <iostream>
#include <stdexcept>

void check_frame(
    const Frame& frame,
    PixelFormat format,
    FrameLayout layout,
    std::size_t width,
    std::size_t height,
    const std::vector<std::byte>& expected
) {
    bool passed = true;

    auto check = [&](const char* name, auto received, auto wanted) {
        if (received != wanted) {
            std::cerr << name
                      << ": received " << received
                      << ", expected " << wanted << '\n';
            passed = false;
        }
    };

    check("format", static_cast<int>(frame.get_pixel_format()), static_cast<int>(format));
    check("layout", static_cast<int>(frame.get_frame_layout()), static_cast<int>(layout));
    check("width", frame.get_width(), width);
    check("height", frame.get_height(), height);

    const auto pixels = frame.view_ref().get_span();
    if (pixels.size() != expected.size()) {
        std::cerr << "incorrect buffer size\n";
        passed = false;
    }

    if (pixels.size() == expected.size()) {
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (pixels[i] != expected[i]) {
                std::cerr << "byte " << i
                          << ": received " << std::to_integer<int>(pixels[i])
                          << ", expected " << std::to_integer<int>(expected[i]) << '\n';
                passed = false;
                break;
            }
        }
    }

    if (!passed)
        throw std::runtime_error("test failed");
}

void test(PixelFormat format, std::size_t channels, std::size_t width, std::size_t height) {
    const std::size_t n = width * height;
    std::vector<std::byte> interleaved(n * channels);
    std::vector<std::byte> planar(n * channels);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < channels; ++c) {
            const auto value = static_cast<std::byte>((i * 7 + c * 53) % 256);
            interleaved[i * channels + c] = value;
            planar[c * n + i] = value;
        }
    }

    Frame frame(width, height, format, FrameLayout::Interleaved, interleaved);
    convert_to_planar(frame);
    check_frame(frame, format, FrameLayout::Planar, width, height, planar);

    convert_to_planar(frame);
    check_frame(frame, format, FrameLayout::Planar, width, height, planar);

    convert_to_interleaved(frame);
    check_frame(frame, format, FrameLayout::Interleaved, width, height, interleaved);

    convert_to_interleaved(frame);
    check_frame(frame, format, FrameLayout::Interleaved, width, height, interleaved);

    Frame other(width, height, format, FrameLayout::Planar, planar);
    convert_to_interleaved(other);
    check_frame(other, format, FrameLayout::Interleaved, width, height, interleaved);
}

int gray8(void) {
    test(PixelFormat::Gray, 1, 3, 2);
    test(PixelFormat::Gray, 1, 13, 5);
    return 0;
}

int rgb8(void) {
    test(PixelFormat::RGB, 3, 3, 2);
    test(PixelFormat::RGB, 3, 13, 5);
    for (std::size_t width : {63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 2049})
        test(PixelFormat::RGB, 3, width, 1);
    test(PixelFormat::RGB, 3, 17, 65);
    return 0;
}

int rgba8(void) {
    test(PixelFormat::RGBA, 4, 3, 2);
    test(PixelFormat::RGBA, 4, 13, 5);
    for (std::size_t width : {63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 2049})
        test(PixelFormat::RGBA, 4, width, 1);
    test(PixelFormat::RGBA, 4, 17, 65);
    return 0;
}

int main(void) {
    int status = 0;
    status |= gray8();
    status |= rgb8();
    status |= rgba8();
    return status;
}
