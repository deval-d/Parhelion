#include "image_types.hpp"
#include <iostream>
#include <stdexcept>

void copy(FrameViewRef src, FrameViewMut dst);

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
    FrameLayout layout,
    std::size_t channels,
    std::size_t width,
    std::size_t height,
    bool crop_src,
    bool crop_dst
) {
    const std::size_t src_width = crop_src ? width + 6 : width;
    const std::size_t src_height = crop_src ? height + 4 : height;
    const std::size_t dst_width = crop_dst ? width + 10 : width;
    const std::size_t dst_height = crop_dst ? height + 6 : height;
    const Coordinate src_xy(crop_src ? 3 : 0, crop_src ? 2 : 0);
    const Coordinate dst_xy(crop_dst ? 5 : 0, crop_dst ? 3 : 0);
    const std::size_t src_n = src_width * src_height;
    const std::size_t dst_n = dst_width * dst_height;

    std::vector<std::byte> original(src_n * channels);
    for (std::size_t i = 0; i < original.size(); ++i)
        original[i] = static_cast<std::byte>((i * 7 + i / 251 + 13) % 256);

    std::vector<std::byte> expected(dst_n * channels, std::byte{171});
    Frame src(src_width, src_height, format, layout, original);
    Frame dst(dst_width, dst_height, format, layout, expected);

    const auto src_view = crop_src
        ? src.view_ref(src_xy, width, height) : src.view_ref();
    const auto dst_view = crop_dst
        ? dst.view_mut(dst_xy, width, height) : dst.view_mut();
    copy(src_view, dst_view);

    // only destination crop should change
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t src_pixel = (src_xy.y + y) * src_width + src_xy.x + x;
            const std::size_t dst_pixel = (dst_xy.y + y) * dst_width + dst_xy.x + x;
            for (std::size_t c = 0; c < channels; ++c) {
                const std::size_t src_index = layout == FrameLayout::Planar
                    ? c * src_n + src_pixel : src_pixel * channels + c;
                const std::size_t dst_index = layout == FrameLayout::Planar
                    ? c * dst_n + dst_pixel : dst_pixel * channels + c;
                expected[dst_index] = original[src_index];
            }
        }
    }

    try {
        check_frame(dst, expected);
        check_frame(src, original);
    } catch (const std::runtime_error&) {
        std::cerr << "copy: " << width << " x " << height
                  << ", channels " << channels
                  << ", layout " << static_cast<int>(layout)
                  << ", cropped source " << crop_src
                  << ", cropped destination " << crop_dst << '\n';
        throw;
    }
}

void test_sizes(PixelFormat format, std::size_t channels) {
    for (auto layout : {FrameLayout::Interleaved, FrameLayout::Planar}) {
        for (Coordinate size : {
            Coordinate(3, 2), Coordinate(4095, 1), Coordinate(4096, 1),
            Coordinate(4097, 1), Coordinate(127, 19), Coordinate(255, 9),
            Coordinate(256, 9), Coordinate(257, 9), Coordinate(1025, 3)
        }) {
            for (bool crop_src : {false, true}) {
                for (bool crop_dst : {false, true})
                    test(format, layout, channels, size.x, size.y, crop_src, crop_dst);
            }
        }
    }
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
    return status;
}
