#include "image_types.hpp"
#include <arm_neon.h>

inline uint8x16_t premultiply_alpha_vector(uint8x16_t color, uint8x16_t alpha) {
    uint16x8_t p0 = vmull_u8(vget_low_u8(color), vget_low_u8(alpha));
    uint16x8_t p1 = vmull_high_u8(color, alpha);
    uint16x8_t bias = vdupq_n_u16(128);
    p0 = vaddq_u16(p0, bias);
    p1 = vaddq_u16(p1, bias);
    p0 = vsraq_n_u16(p0, p0, 8);
    p1 = vsraq_n_u16(p1, p1, 8);
    uint8x8_t result = vshrn_n_u16(p0, 8);
    return vshrn_high_n_u16(result, p1, 8);
}

inline void premultiply_alpha_buffer(
    std::uint8_t *r,
    std::uint8_t *g,
    std::uint8_t *b,
    const std::uint8_t *a,
    std::size_t n
) {
    std::size_t n_full = n & ~std::size_t(15);
    std::size_t i = 0;
    for (; i < n_full; i += 16) {
        uint8x16_t alpha = vld1q_u8(a + i);
        uint8x16_t rv = premultiply_alpha_vector(vld1q_u8(r + i), alpha);
        uint8x16_t gv = premultiply_alpha_vector(vld1q_u8(g + i), alpha);
        uint8x16_t bv = premultiply_alpha_vector(vld1q_u8(b + i), alpha);
        vst1q_u8(r + i, rv);
        vst1q_u8(g + i, gv);
        vst1q_u8(b + i, bv);
    }

    for (; i < n; ++i) {
        std::uint16_t alpha = a[i];
        r[i] = (r[i] * alpha + 127) / 255;
        g[i] = (g[i] * alpha + 127) / 255;
        b[i] = (b[i] * alpha + 127) / 255;
    }
}

void premultiply_alpha_channels(
    FrameViewMut r_frame,
    FrameViewMut g_frame,
    FrameViewMut b_frame,
    FrameViewRef a_frame
) {
    std::size_t width = r_frame.get_width();
    std::size_t height = r_frame.get_height();
    std::size_t stride = r_frame.get_row_stride();

    std::uint8_t *r = reinterpret_cast<std::uint8_t *>(r_frame.get_span().data());
    std::uint8_t *g = reinterpret_cast<std::uint8_t *>(g_frame.get_span().data());
    std::uint8_t *b = reinterpret_cast<std::uint8_t *>(b_frame.get_span().data());
    const std::uint8_t *a = reinterpret_cast<const std::uint8_t *>(a_frame.get_span().data());

    // fast path
    if (width == stride) {
        premultiply_alpha_buffer(r, g, b, a, width * height);
        return;
    }

    for (std::size_t y = 0; y < height; ++y) {
        premultiply_alpha_buffer(
            r + y * stride,
            g + y * stride,
            b + y * stride,
            a + y * stride,
            width
        );
    }
}

void premultiply_alpha(FrameViewMut frame) {
    if (frame.get_pixel_format() != PixelFormat::RGBA)
        throw std::invalid_argument("expected RGBA frame");
    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");

    FrameViewMut r_frame = frame.channel_mut(0);
    FrameViewMut g_frame = frame.channel_mut(1);
    FrameViewMut b_frame = frame.channel_mut(2);
    FrameViewRef a_frame = frame.channel_ref(3);
    premultiply_alpha_channels(r_frame, g_frame, b_frame, a_frame);
}
