#include "parhelion-2.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <functional>
#include <stdexcept>
#include <vector>

namespace {

template <std::size_t radius>
inline uint16x8_t box_blur_sum(uint16x8_t r0, uint16x8_t r1, uint16x8_t r2) {
    uint16x8_t sum = vaddq_u16(r0, vextq_u16(r0, r1, 1));
    sum = vaddq_u16(sum, vextq_u16(r0, r1, 2));
    if constexpr (radius >= 2) {
        sum = vaddq_u16(sum, vextq_u16(r0, r1, 3));
        sum = vaddq_u16(sum, vextq_u16(r0, r1, 4));
    }
    if constexpr (radius >= 3) {
        sum = vaddq_u16(sum, vextq_u16(r0, r1, 5));
        sum = vaddq_u16(sum, vextq_u16(r0, r1, 6));
    }
    if constexpr (radius >= 4) {
        sum = vaddq_u16(sum, vextq_u16(r0, r1, 7));
        sum = vaddq_u16(sum, r1);
    }
    if constexpr (radius >= 5) {
        sum = vaddq_u16(sum, vextq_u16(r1, r2, 1));
        sum = vaddq_u16(sum, vextq_u16(r1, r2, 2));
    }
    if constexpr (radius >= 6) {
        sum = vaddq_u16(sum, vextq_u16(r1, r2, 3));
        sum = vaddq_u16(sum, vextq_u16(r1, r2, 4));
    }
    if constexpr (radius >= 7) {
        sum = vaddq_u16(sum, vextq_u16(r1, r2, 5));
        sum = vaddq_u16(sum, vextq_u16(r1, r2, 6));
    }
    return sum;
}

template <std::size_t radius, bool packed>
void box_blur_buffer(
    std::uint8_t *dst,
    const std::uint8_t *src,
    std::uint16_t *rows,
    std::size_t width,
    std::size_t height,
    std::size_t src_stride,
    std::size_t dst_stride
) {
    constexpr std::uint16_t reciprocals[] = {0, 7282, 5243, 2675, 25891, 4333, 49637, 37283};
    constexpr int shifts[] = {0, 0, 1, 1, 5, 3, 7, 7};
    constexpr std::uint16_t reciprocal = reciprocals[radius];
    constexpr int shift = shifts[radius];
    constexpr std::size_t n_vector = 8;
    if constexpr (packed) {
        src_stride = width;
        dst_stride = width;
    }
    std::uint16_t *row = rows + radius;
    std::size_t bottom = std::min(radius, height - 1);
    std::size_t i = 0;
    for (; i + n_vector <= width; i += n_vector) {
        uint16x8_t sum = vmulq_n_u16(vmovl_u8(vld1_u8(src + i)), radius + 1);
        for (std::size_t y = 1; y <= bottom; ++y)
            sum = vaddw_u8(sum, vld1_u8(src + y * src_stride + i));
        if (radius > bottom) {
            uint16x8_t r0 = vmovl_u8(vld1_u8(src + (height - 1) * src_stride + i));
            sum = vmlaq_n_u16(sum, r0, radius - bottom);
        }
        vst1q_u16(row + i, sum);
    }
    for (; i < width; ++i) {
        std::uint16_t sum = (radius + 1) * src[i];
        for (std::size_t y = 1; y <= bottom; ++y)
            sum += src[y * src_stride + i];
        sum += (radius - bottom) * src[(height - 1) * src_stride + i];
        row[i] = sum;
    }

    for (std::size_t y = 0; y < height; ++y) {
        std::fill_n(rows, radius, row[0]);
        std::fill_n(row + width, radius, row[width - 1]);
        uint16x8_t r0 = vld1q_u16(rows);
        for (i = 0; i < width; i += n_vector) {
            uint16x8_t r1 = vld1q_u16(rows + i + n_vector);
            uint16x8_t r2 = vdupq_n_u16(0);
            if constexpr (radius >= 5)
                r2 = vld1q_u16(rows + i + 2 * n_vector);
            uint16x8_t sum = box_blur_sum<radius>(r0, r1, r2);
            uint32x4_t lo = vmull_n_u16(vget_low_u16(sum), reciprocal);
            uint32x4_t hi = vmull_n_u16(vget_high_u16(sum), reciprocal);
            uint16x8_t result = vcombine_u16(vshrn_n_u32(lo, 16), vshrn_n_u32(hi, 16));
            if constexpr (shift > 0)
                result = vshrq_n_u16(result, shift);
            uint8x8_t pixels = vmovn_u16(result);
            if (width - i >= n_vector) {
                vst1_u8(dst + y * dst_stride + i, pixels);
            } else {
                std::uint8_t tail[8];
                vst1_u8(tail, pixels);
                std::memcpy(dst + y * dst_stride + i, tail, width - i);
            }
            r0 = r1;
        }
        if (y + 1 == height)
            break;

        std::size_t leaving = y > radius ? y - radius : 0;
        std::size_t entering = radius >= height - 1 - y ? height - 1 : y + radius + 1;
        i = 0;
        for (; i + n_vector <= width; i += n_vector) {
            uint16x8_t sum = vld1q_u16(row + i);
            sum = vsubw_u8(sum, vld1_u8(src + leaving * src_stride + i));
            sum = vaddw_u8(sum, vld1_u8(src + entering * src_stride + i));
            vst1q_u16(row + i, sum);
        }
        for (; i < width; ++i) {
            row[i] -= src[leaving * src_stride + i];
            row[i] += src[entering * src_stride + i];
        }
    }
}

template <std::size_t radius>
void box_blur_channels(FrameViewRef src, FrameViewMut dst) {
    std::size_t width = src.get_width();
    std::size_t height = src.get_height();
    std::size_t src_stride = src.get_row_stride();
    std::size_t dst_stride = dst.get_row_stride();
    bool packed = src_stride == width && dst_stride == width;
    std::vector<std::uint16_t> rows(width + 2 * radius + 16);

    for (std::size_t c = 0; c < src.get_n_channels(); ++c) {
        FrameViewRef src_channel = src.channel_ref(c);
        FrameViewMut dst_channel = dst.channel_mut(c);
        const std::uint8_t *src_ptr = reinterpret_cast<const std::uint8_t *>(src_channel.get_span().data());
        std::uint8_t *dst_ptr = reinterpret_cast<std::uint8_t *>(dst_channel.get_span().data());

        // fast path
        if (packed) {
            box_blur_buffer<radius, true>(
                dst_ptr, src_ptr, rows.data(), width, height, width, width
            );
        } else {
            box_blur_buffer<radius, false>(
                dst_ptr, src_ptr, rows.data(), width, height, src_stride, dst_stride
            );
        }
    }
}

} // namespace

void box_blur(FrameViewRef src, FrameViewMut dst, std::size_t radius) {
    if (src.get_frame_layout() != FrameLayout::Planar ||
        dst.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
    if (src.get_pixel_format() != dst.get_pixel_format())
        throw std::invalid_argument("mismatched pixel formats");
    if (src.get_width() != dst.get_width() || src.get_height() != dst.get_height())
        throw std::invalid_argument("mismatched dimensions");
    if (radius < 1 || radius > 7)
        throw std::invalid_argument("radius must be between 1 and 7");
    if (src.get_width() == 0 || src.get_height() == 0)
        return;
    if (src.get_width() > std::numeric_limits<std::size_t>::max() - 2 * radius - 16)
        throw std::overflow_error("row size overflows");

    std::span<const std::byte> src_buf = src.get_span();
    std::span<std::byte> dst_buf = dst.get_span();
    std::less<const std::byte *> before;
    if (before(src_buf.data(), dst_buf.data() + dst_buf.size()) &&
        before(dst_buf.data(), src_buf.data() + src_buf.size()))
        throw std::invalid_argument("source and destination must not overlap");

    switch (radius) {
        case 1:
            box_blur_channels<1>(src, dst);
            break;
        case 2:
            box_blur_channels<2>(src, dst);
            break;
        case 3:
            box_blur_channels<3>(src, dst);
            break;
        case 4:
            box_blur_channels<4>(src, dst);
            break;
        case 5:
            box_blur_channels<5>(src, dst);
            break;
        case 6:
            box_blur_channels<6>(src, dst);
            break;
        case 7:
            box_blur_channels<7>(src, dst);
            break;
    }
}
