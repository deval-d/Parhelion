#include "parhelion-2.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include <functional>

namespace {

inline uint8x8_t laplacian_vector(
    uint8x8_t center,
    uint8x8_t top,
    uint8x8_t bottom,
    uint8x8_t left,
    uint8x8_t right
) {
    int16x8_t neighbors = vreinterpretq_s16_u16(vaddq_u16(vaddl_u8(top, bottom), vaddl_u8(left, right)));
    int16x8_t center_sum = vreinterpretq_s16_u16(vmulq_n_u16(vmovl_u8(center), 4));
    int16x8_t sum = vsubq_s16(neighbors, center_sum);
    return vqmovun_s16(sum);
}

template <bool packed>
void laplacian_buffer(
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
            const int sum = int(top[x]) + bottom[x] + row[left] + row[right] - 4 * int(row[x]);
            output[x] = static_cast<std::uint8_t>(std::clamp(sum, 0, 255));
        };
        scalar(0);
        std::size_t i = 1;
        for (; i + 16 < width; i += 16) {
            uint8x16_t center = vld1q_u8(row + i);
            uint8x16_t t = vld1q_u8(top + i);
            uint8x16_t b = vld1q_u8(bottom + i);
            uint8x16_t left = vld1q_u8(row + i - 1);
            uint8x16_t right = vld1q_u8(row + i + 1);
            uint8x8_t r0 = laplacian_vector(vget_low_u8(center), vget_low_u8(t), vget_low_u8(b), vget_low_u8(left), vget_low_u8(right));
            uint8x8_t r1 = laplacian_vector(vget_high_u8(center), vget_high_u8(t), vget_high_u8(b), vget_high_u8(left), vget_high_u8(right));
            vst1q_u8(output + i, vcombine_u8(r0, r1));
        }
        for (; i < width; ++i)
            scalar(i);
    }
}

} // namespace

void laplacian(FrameViewRef src, FrameViewMut dst) {
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
            laplacian_buffer<true>(src_ptr, dst_ptr, width, height, width, width);
        } else {
            laplacian_buffer<false>(src_ptr, dst_ptr, width, height, src_stride, dst_stride);
        }
    }
}
