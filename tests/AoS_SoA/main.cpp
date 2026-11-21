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

    check("format", frame.get_pixel_format(), format);
    check("layout", frame.get_frame_layout(), layout);
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

    Frame frame(width, height, format, Interleaved, interleaved);
    convert_to_planar(frame);
    check_frame(frame, format, Planar, width, height, planar);

    convert_to_planar(frame);
    check_frame(frame, format, Planar, width, height, planar);

    convert_to_interleaved(frame);
    check_frame(frame, format, Interleaved, width, height, interleaved);

    convert_to_interleaved(frame);
    check_frame(frame, format, Interleaved, width, height, interleaved);

    Frame other(width, height, format, Planar, planar);
    convert_to_interleaved(other);
    check_frame(other, format, Interleaved, width, height, interleaved);
}

int gray8(void) {
    test(Gray, 1, 3, 2);
    test(Gray, 1, 13, 5);
    return 0;
}

int rgb8(void) {
    test(RGB, 3, 3, 2);
    test(RGB, 3, 13, 5);
    return 0;
}

int rgba8(void) {
    test(RGBA, 4, 3, 2);
    test(RGBA, 4, 13, 5);
    return 0;
}

int main(void) {
    int status = 0;
    status |= gray8();
    status |= rgb8();
    status |= rgba8();
    return status;
}
