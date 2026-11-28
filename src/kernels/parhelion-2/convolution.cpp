#include "parhelion-2.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>

namespace {

inline float32x4x2_t convolution_widen(uint8x8_t values) {
    uint16x8_t v = vmovl_u8(values);
    return {
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(v))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(v)))
    };
}

template <bool packed>
void convolution_buffer(
    const std::uint8_t *src,
    std::uint8_t *dst,
    std::uint8_t *rows,
    std::span<const float> kernel,
    std::size_t kernel_width,
    std::size_t kernel_height,
    std::size_t width,
    std::size_t height,
    std::size_t src_stride,
    std::size_t dst_stride
) {
    constexpr std::size_t n_vector = 8;
    const std::size_t n_full = width & ~(n_vector - 1);
    const std::size_t rx = kernel_width / 2;
    const std::size_t ry = kernel_height / 2;
    const std::size_t row_stride = width + 2 * rx;
    if constexpr (packed) {
        src_stride = width;
        dst_stride = width;
    }
    for (std::size_t y = 0; y < height; ++y) {
        const std::uint8_t *input = src + y * src_stride;
        std::uint8_t *row = rows + y * row_stride;
        std::fill_n(row, rx, input[0]);
        std::memcpy(row + rx, input, width);
        std::fill_n(row + rx + width, rx, input[width - 1]);
    }

    for (std::size_t y = 0; y < height; ++y) {
        std::size_t i = 0;
        for (; i < n_full; i += n_vector) {
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            for (std::size_t ky = 0; ky < kernel_height; ++ky) {
                const std::size_t sy = ky < ry ? (ry - ky > y ? 0 : y - (ry - ky))
                    : (ky - ry >= height - y ? height - 1 : y + (ky - ry));
                const std::uint8_t *row = rows + sy * row_stride + i;
                for (std::size_t kx = 0; kx < kernel_width; ++kx) {
                    const float weight = kernel[kernel.size() - 1 - (ky * kernel_width + kx)];
                    float32x4x2_t r0 = convolution_widen(vld1_u8(row + kx));
                    sum0 = vfmaq_n_f32(sum0, r0.val[0], weight);
                    sum1 = vfmaq_n_f32(sum1, r0.val[1], weight);
                }
            }
            sum0 = vminnmq_f32(vmaxnmq_f32(sum0, vdupq_n_f32(0.0f)), vdupq_n_f32(255.0f));
            sum1 = vminnmq_f32(vmaxnmq_f32(sum1, vdupq_n_f32(0.0f)), vdupq_n_f32(255.0f));
            uint16x8_t result = vcombine_u16(
                vqmovn_u32(vcvtnq_u32_f32(sum0)),
                vqmovn_u32(vcvtnq_u32_f32(sum1))
            );
            vst1_u8(dst + y * dst_stride + i, vqmovn_u16(result));
        }
        for (; i < width; ++i) {
            float sum = 0.0f;
            for (std::size_t ky = 0; ky < kernel_height; ++ky) {
                const std::size_t sy = ky < ry ? (ry - ky > y ? 0 : y - (ry - ky))
                    : (ky - ry >= height - y ? height - 1 : y + (ky - ry));
                const std::uint8_t *row = rows + sy * row_stride + i;
                for (std::size_t kx = 0; kx < kernel_width; ++kx) {
                    const float weight = kernel[kernel.size() - 1 - (ky * kernel_width + kx)];
                    sum = std::fma(float(row[kx]), weight, sum);
                }
            }
            sum = std::fmin(std::fmax(sum, 0.0f), 255.0f);
            dst[y * dst_stride + i] = static_cast<std::uint8_t>(std::nearbyint(sum));
        }
    }
}

} // namespace

void convolution(
    FrameViewRef src,
    FrameViewMut dst,
    std::span<const float> kernel,
    std::size_t kernel_width,
    std::size_t kernel_height
) {
    if (src.get_frame_layout() != FrameLayout::Planar ||
        dst.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
    if (src.get_pixel_format() != dst.get_pixel_format())
        throw std::invalid_argument("mismatched pixel formats");
    if (src.get_width() != dst.get_width() || src.get_height() != dst.get_height())
        throw std::invalid_argument("mismatched dimensions");
    if (kernel_width == 0 || kernel_height == 0 ||
        kernel_width % 2 == 0 || kernel_height % 2 == 0)
        throw std::invalid_argument("kernel dimensions must be odd and nonzero");
    if (kernel_width > std::numeric_limits<std::size_t>::max() / kernel_height)
        throw std::overflow_error("kernel size overflows");
    if (kernel.size() != kernel_width * kernel_height)
        throw std::invalid_argument("kernel dimensions do not match weights");
    for (float weight : kernel)
        if (!std::isfinite(weight))
            throw std::invalid_argument("kernel weights must be finite");

    const std::size_t width = src.get_width();
    const std::size_t height = src.get_height();
    if (width == 0 || height == 0)
        return;
    if (width > std::numeric_limits<std::size_t>::max() - (kernel_width - 1))
        throw std::overflow_error("row size overflows");
    const std::size_t row_stride = width + kernel_width - 1;
    if (row_stride > std::numeric_limits<std::size_t>::max() / height)
        throw std::overflow_error("scratch size overflows");

    const auto src_buf = src.get_span();
    const auto dst_buf = dst.get_span();
    std::less<const std::byte *> before;
    if (before(src_buf.data(), dst_buf.data() + dst_buf.size()) &&
        before(dst_buf.data(), src_buf.data() + src_buf.size()))
        throw std::invalid_argument("source and destination must not overlap");

    std::vector<std::uint8_t> rows(row_stride * height);
    const std::size_t src_stride = src.get_row_stride();
    const std::size_t dst_stride = dst.get_row_stride();
    const bool packed = src_stride == width && dst_stride == width;
    for (std::size_t c = 0; c < src.get_n_channels(); ++c) {
        const auto src_channel = src.channel_ref(c);
        const auto dst_channel = dst.channel_mut(c);
        const std::uint8_t *src_ptr = reinterpret_cast<const std::uint8_t *>(src_channel.get_span().data());
        std::uint8_t *dst_ptr = reinterpret_cast<std::uint8_t *>(dst_channel.get_span().data());
        if (packed) {
            convolution_buffer<true>(src_ptr, dst_ptr, rows.data(), kernel, kernel_width, kernel_height,
                width, height, width, width);
        } else {
            convolution_buffer<false>(src_ptr, dst_ptr, rows.data(), kernel, kernel_width, kernel_height,
                width, height, src_stride, dst_stride);
        }
    }
}
