#include "parhelion-2.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include <functional>

namespace {

inline uint8x8_t sobel_vector(
    uint8x8_t lo0,
    uint8x8_t hi0,
    uint8x8_t lo1,
    uint8x8_t hi1,
    uint8x8_t lo2,
    uint8x8_t hi2
) {
    int16x8_t r0 = vreinterpretq_s16_u16(vsubl_u8(hi0, lo0));
    int16x8_t r1 = vreinterpretq_s16_u16(vsubl_u8(hi1, lo1));
    int16x8_t r2 = vreinterpretq_s16_u16(vsubl_u8(hi2, lo2));
    int16x8_t sum = vaddq_s16(vaddq_s16(r0, r2), vshlq_n_s16(r1, 1));
    return vqmovun_s16(sum);
}

template <bool horizontal, bool packed>
void sobel_buffer(
    const std::uint8_t *src,
    std::uint8_t *dst,
    std::size_t width,
    std::size_t height,
    std::size_t src_stride,
    std::size_t dst_stride
) {
    if constexpr (packed) {
        src_stride = width;
        dst_stride = width;
    }
    for (std::size_t y = 0; y < height; ++y) {
        const std::uint8_t *top = src + (y == 0 ? 0 : y - 1) * src_stride;
        const std::uint8_t *row = src + y * src_stride;
        const std::uint8_t *bottom = src + (y + 1 == height ? y : y + 1) * src_stride;
        std::uint8_t *output = dst + y * dst_stride;
        auto scalar = [&](std::size_t x) {
            const std::size_t left = x == 0 ? 0 : x - 1;
            const std::size_t right = x + 1 == width ? x : x + 1;
            int sum;
            if constexpr (horizontal)
                sum = int(top[right]) - top[left] + 2 * (int(row[right]) - row[left])
                    + int(bottom[right]) - bottom[left];
            else
                sum = int(bottom[left]) - top[left] + 2 * (int(bottom[x]) - top[x])
                    + int(bottom[right]) - top[right];
            output[x] = static_cast<std::uint8_t>(std::clamp(sum, 0, 255));
        };
        scalar(0);
        std::size_t i = 1;
        for (; i + 16 < width; i += 16) {
            uint8x16_t lo0, hi0, lo1, hi1, lo2, hi2;
            if constexpr (horizontal) {
                lo0 = vld1q_u8(top + i - 1);
                hi0 = vld1q_u8(top + i + 1);
                lo1 = vld1q_u8(row + i - 1);
                hi1 = vld1q_u8(row + i + 1);
                lo2 = vld1q_u8(bottom + i - 1);
                hi2 = vld1q_u8(bottom + i + 1);
            } else {
                lo0 = vld1q_u8(top + i - 1);
                hi0 = vld1q_u8(bottom + i - 1);
                lo1 = vld1q_u8(top + i);
                hi1 = vld1q_u8(bottom + i);
                lo2 = vld1q_u8(top + i + 1);
                hi2 = vld1q_u8(bottom + i + 1);
            }
            uint8x8_t r0 = sobel_vector(vget_low_u8(lo0), vget_low_u8(hi0), vget_low_u8(lo1), vget_low_u8(hi1), vget_low_u8(lo2), vget_low_u8(hi2));
            uint8x8_t r1 = sobel_vector(vget_high_u8(lo0), vget_high_u8(hi0), vget_high_u8(lo1), vget_high_u8(hi1), vget_high_u8(lo2), vget_high_u8(hi2));
            vst1q_u8(output + i, vcombine_u8(r0, r1));
        }
        for (; i < width; ++i)
            scalar(i);
    }
}

template <bool horizontal>
void sobel_channels(FrameViewRef src, FrameViewMut dst) {
    if (src.get_frame_layout() != FrameLayout::Planar ||
        dst.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
    if (src.get_pixel_format() != dst.get_pixel_format())
        throw std::invalid_argument("mismatched pixel formats");
    if (src.get_width() != dst.get_width() || src.get_height() != dst.get_height())
        throw std::invalid_argument("mismatched dimensions");

    const std::size_t width = src.get_width();
    const std::size_t height = src.get_height();
    if (width == 0 || height == 0)
        return;
    const auto src_buf = src.get_span();
    const auto dst_buf = dst.get_span();
    std::less<const std::byte *> before;
    if (before(src_buf.data(), dst_buf.data() + dst_buf.size()) &&
        before(dst_buf.data(), src_buf.data() + src_buf.size()))
        throw std::invalid_argument("source and destination must not overlap");

    const std::size_t src_stride = src.get_row_stride();
    const std::size_t dst_stride = dst.get_row_stride();
    const bool packed = src_stride == width && dst_stride == width;
    for (std::size_t c = 0; c < src.get_n_channels(); ++c) {
        const auto src_channel = src.channel_ref(c);
        const auto dst_channel = dst.channel_mut(c);
        const std::uint8_t *src_ptr = reinterpret_cast<const std::uint8_t *>(src_channel.get_span().data());
        std::uint8_t *dst_ptr = reinterpret_cast<std::uint8_t *>(dst_channel.get_span().data());
        if (packed) {
            sobel_buffer<horizontal, true>(src_ptr, dst_ptr, width, height, width, width);
        } else {
            sobel_buffer<horizontal, false>(src_ptr, dst_ptr, width, height, src_stride, dst_stride);
        }
    }
}

} // namespace

void sobel_x(FrameViewRef src, FrameViewMut dst) {
    sobel_channels<true>(src, dst);
}

void sobel_y(FrameViewRef src, FrameViewMut dst) {
    sobel_channels<false>(src, dst);
}
