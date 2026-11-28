#include "parhelion-2.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>

namespace {

inline float32x4x2_t gaussian_widen(uint16x8_t values) {
    return {
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(values))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(values)))
    };
}

template <bool packed>
void gaussian_blur_buffer(
    const std::uint8_t *src,
    std::uint8_t *dst,
    float *rows,
    const float *weights,
    std::size_t width,
    std::size_t height,
    std::size_t radius,
    std::size_t src_stride,
    std::size_t dst_stride
) {
    constexpr std::size_t n_vector = 8;
    const std::size_t n_full = width & ~(n_vector - 1);
    if constexpr (packed) {
        src_stride = width;
        dst_stride = width;
    }
    float *row = rows + radius;
    for (std::size_t y = 0; y < height; ++y) {
        std::size_t i = 0;
        for (; i < n_full; i += n_vector) {
            float32x4x2_t r0 = gaussian_widen(vmovl_u8(vld1_u8(src + y * src_stride + i)));
            float32x4_t sum0 = vmulq_n_f32(r0.val[0], weights[0]);
            float32x4_t sum1 = vmulq_n_f32(r0.val[1], weights[0]);
            for (std::size_t k = 1; k <= radius; ++k) {
                const std::size_t top = k > y ? 0 : y - k;
                const std::size_t bottom = k >= height - y ? height - 1 : y + k;
                uint16x8_t pair = vaddl_u8(
                    vld1_u8(src + top * src_stride + i),
                    vld1_u8(src + bottom * src_stride + i)
                );
                float32x4x2_t r1 = gaussian_widen(pair);
                sum0 = vfmaq_n_f32(sum0, r1.val[0], weights[k]);
                sum1 = vfmaq_n_f32(sum1, r1.val[1], weights[k]);
            }
            vst1q_f32(row + i, sum0);
            vst1q_f32(row + i + 4, sum1);
        }
        for (; i < width; ++i) {
            float sum = src[y * src_stride + i] * weights[0];
            for (std::size_t k = 1; k <= radius; ++k) {
                const std::size_t top = k > y ? 0 : y - k;
                const std::size_t bottom = k >= height - y ? height - 1 : y + k;
                const unsigned pair = src[top * src_stride + i] + src[bottom * src_stride + i];
                sum = std::fma(float(pair), weights[k], sum);
            }
            row[i] = sum;
        }

        std::fill_n(rows, radius, row[0]);
        std::fill_n(row + width, radius, row[width - 1]);
        i = 0;
        for (; i < n_full; i += n_vector) {
            float32x4_t sum0 = vmulq_n_f32(vld1q_f32(row + i), weights[0]);
            float32x4_t sum1 = vmulq_n_f32(vld1q_f32(row + i + 4), weights[0]);
            for (std::size_t k = 1; k <= radius; ++k) {
                float32x4_t r0 = vaddq_f32(vld1q_f32(row + i - k), vld1q_f32(row + i + k));
                float32x4_t r1 = vaddq_f32(vld1q_f32(row + i + 4 - k), vld1q_f32(row + i + 4 + k));
                sum0 = vfmaq_n_f32(sum0, r0, weights[k]);
                sum1 = vfmaq_n_f32(sum1, r1, weights[k]);
            }
            uint16x8_t result = vcombine_u16(
                vqmovn_u32(vcvtnq_u32_f32(sum0)),
                vqmovn_u32(vcvtnq_u32_f32(sum1))
            );
            vst1_u8(dst + y * dst_stride + i, vqmovn_u16(result));
        }
        for (; i < width; ++i) {
            float sum = row[i] * weights[0];
            for (std::size_t k = 1; k <= radius; ++k)
                sum = std::fma(row[std::ptrdiff_t(i) - std::ptrdiff_t(k)] + row[i + k], weights[k], sum);
            dst[y * dst_stride + i] = static_cast<std::uint8_t>(std::clamp(std::nearbyint(sum), 0.0f, 255.0f));
        }
    }
}

} // namespace

void gaussian_blur(FrameViewRef src, FrameViewMut dst, float sigma, std::size_t radius) {
    if (src.get_frame_layout() != FrameLayout::Planar ||
        dst.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
    if (src.get_pixel_format() != dst.get_pixel_format())
        throw std::invalid_argument("mismatched pixel formats");
    if (src.get_width() != dst.get_width() || src.get_height() != dst.get_height())
        throw std::invalid_argument("mismatched dimensions");
    if (!std::isfinite(sigma) || sigma <= 0.0f)
        throw std::invalid_argument("sigma must be finite and positive");

    const std::size_t width = src.get_width();
    const std::size_t height = src.get_height();
    if (width == 0 || height == 0)
        return;
    if (width > std::numeric_limits<std::size_t>::max() - 8 ||
        radius > (std::numeric_limits<std::size_t>::max() - width - 8) / 2)
        throw std::overflow_error("row size overflows");

    const auto src_buf = src.get_span();
    const auto dst_buf = dst.get_span();
    std::less<const std::byte *> before;
    if (before(src_buf.data(), dst_buf.data() + dst_buf.size()) &&
        before(dst_buf.data(), src_buf.data() + src_buf.size()))
        throw std::invalid_argument("source and destination must not overlap");

    std::vector<float> weights(radius + 1);
    weights[0] = 1.0f;
    double total = 1.0;
    for (std::size_t k = 1; k <= radius; ++k) {
        const double distance = double(k) / double(sigma);
        const double weight = std::exp(-0.5 * distance * distance);
        weights[k] = static_cast<float>(weight);
        total += 2.0 * weight;
    }
    for (float& weight : weights)
        weight = static_cast<float>(double(weight) / total);

    std::vector<float> rows(width + 2 * radius + 8);
    const std::size_t src_stride = src.get_row_stride();
    const std::size_t dst_stride = dst.get_row_stride();
    const bool packed = src_stride == width && dst_stride == width;
    for (std::size_t c = 0; c < src.get_n_channels(); ++c) {
        const auto src_channel = src.channel_ref(c);
        const auto dst_channel = dst.channel_mut(c);
        const std::uint8_t *src_ptr = reinterpret_cast<const std::uint8_t *>(src_channel.get_span().data());
        std::uint8_t *dst_ptr = reinterpret_cast<std::uint8_t *>(dst_channel.get_span().data());
        if (radius == 0) {
            for (std::size_t y = 0; y < height; ++y)
                std::memcpy(dst_ptr + y * dst_stride, src_ptr + y * src_stride, width);
        } else if (packed) {
            gaussian_blur_buffer<true>(src_ptr, dst_ptr, rows.data(), weights.data(),
                width, height, radius, width, width);
        } else {
            gaussian_blur_buffer<false>(src_ptr, dst_ptr, rows.data(), weights.data(),
                width, height, radius, src_stride, dst_stride);
        }
    }
}
