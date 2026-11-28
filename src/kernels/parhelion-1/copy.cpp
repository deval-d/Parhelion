#include "parhelion-1.hpp"
#include <arm_sve.h>
#include <cstring>

namespace { 
    inline void copy_sve_buffer(
        const std::uint8_t *src,
        std::uint8_t *dst, 
        std::size_t n
    ) __arm_streaming { 
        std::size_t svl_b = svcntb(); 
        std::size_t n_VGx4 = 4 * svl_b;
        std::size_t n_4xVGx4 = 4 * n_VGx4;
        std::size_t n_full_4xVGx4 = n & ~(n_4xVGx4 - 1); 

        std::size_t i = 0;
        svcount_t pg = svptrue_c8(); 
        for (; i < n_full_4xVGx4; i += n_4xVGx4) { 
            svuint8x4_t sv1 = svld1_u8_x4(pg, src + i); 
            svuint8x4_t sv2 = svld1_u8_x4(pg, src + i + n_VGx4);
            svuint8x4_t sv3 = svld1_u8_x4(pg, src + i + 2 * n_VGx4);
            svuint8x4_t sv4 = svld1_u8_x4(pg, src + i + 3 * n_VGx4); 
            svst1_u8_x4(pg, dst + i, sv1);
            svst1_u8_x4(pg, dst + i + n_VGx4, sv2);
            svst1_u8_x4(pg, dst + i + 2 * n_VGx4, sv3);
            svst1_u8_x4(pg, dst + i + 3 * n_VGx4, sv4);
        }

        if (i == n) 
            return; 

        svcount_t p1 = svwhilelt_c8_u64(i, n, 4); 
        svcount_t p2 = svwhilelt_c8_u64(i + n_VGx4, n, 4);
        svcount_t p3 = svwhilelt_c8_u64(i + 2 * n_VGx4, n, 4);
        svcount_t p4 = svwhilelt_c8_u64(i + 3 * n_VGx4, n, 4);
        svuint8x4_t sv1 = svld1_u8_x4(p1, src + i); 
        svuint8x4_t sv2 = svld1_u8_x4(p2, src + i + n_VGx4);
        svuint8x4_t sv3 = svld1_u8_x4(p3, src + i + 2 * n_VGx4);
        svuint8x4_t sv4 = svld1_u8_x4(p4, src + i + 3 * n_VGx4); 
        svst1_u8_x4(p1, dst + i, sv1);
        svst1_u8_x4(p2, dst + i + n_VGx4, sv2);
        svst1_u8_x4(p3, dst + i + 2 * n_VGx4, sv3);
        svst1_u8_x4(p4, dst + i + 3 * n_VGx4, sv4);
    }

    __arm_locally_streaming 
    void copy_sve(FrameViewRef src, FrameViewMut dst, bool packed) { 
        std::size_t src_width = src.get_width();
        std::size_t src_height = src.get_height(); 
        std::size_t n_channels = src.get_n_channels(); 

        const std::uint8_t *src_ptr = reinterpret_cast<const std::uint8_t *>(src.get_span().data()); 
        std::uint8_t *dst_ptr = reinterpret_cast<std::uint8_t *>(dst.get_span().data()); 
        if (packed) { 
            copy_sve_buffer(
                src_ptr,
                dst_ptr,
                src_width * src_height * n_channels 
            );
            return; 
        }

        switch (src.get_frame_layout()) { 
            case FrameLayout::Planar:
                for (std::size_t c = 0; c < n_channels; ++c) { 
                    for (std::size_t y = 0; y < src_height; ++y) { 
                        copy_sve_buffer(
                            src_ptr + y * src.get_row_stride() + c * src.get_channel_stride().value(),
                            dst_ptr + y * dst.get_row_stride() + c * dst.get_channel_stride().value(),
                            src_width 
                        );
                    }
                }
                return;
            case FrameLayout::Interleaved:
                for (std::size_t y = 0; y < src_height; ++y) { 
                    copy_sve_buffer(
                        src_ptr + y * src.get_row_stride(),
                        dst_ptr + y * dst.get_row_stride(),
                        src_width * n_channels 
                    );
                }
                return;
        }
    }

    void copy_memcpy(FrameViewRef src, FrameViewMut dst, bool packed) {
        std::size_t width = src.get_width();
        std::size_t height = src.get_height();
        std::size_t n_channels = src.get_n_channels();
        const std::byte *src_ptr = src.get_span().data();
        std::byte *dst_ptr = dst.get_span().data();

        if (packed) {
            std::memcpy(dst_ptr, src_ptr, width * height * n_channels);
            return;
        }

        switch (src.get_frame_layout()) {
            case FrameLayout::Planar:
                for (std::size_t c = 0; c < n_channels; ++c) {
                    for (std::size_t y = 0; y < height; ++y) {
                        std::memcpy(
                            dst_ptr + c * dst.get_channel_stride().value() + y * dst.get_row_stride(),
                            src_ptr + c * src.get_channel_stride().value() + y * src.get_row_stride(),
                            width
                        );
                    }
                }
                return;

            case FrameLayout::Interleaved:
                for (std::size_t y = 0; y < height; ++y) {
                    std::memcpy(
                        dst_ptr + y * dst.get_row_stride(),
                        src_ptr + y * src.get_row_stride(),
                        width * n_channels
                    );
                }
                return; 
        }
    }
}

void copy(FrameViewRef src, FrameViewMut dst) { 
    std::size_t width = src.get_width();
    std::size_t height = src.get_height();

    if (src.get_frame_layout() != dst.get_frame_layout())
        throw std::invalid_argument("mismatched frame layouts");
    if (src.get_pixel_format() != dst.get_pixel_format())
        throw std::invalid_argument("mismatched pixel formats"); 
    if (width != dst.get_width() || height != dst.get_height())
        throw std::invalid_argument("mismatched dimensions");

    if (width == 0 || height == 0)
        return; 

    std::size_t n_channels = src.get_n_channels();
    bool planar = src.get_frame_layout() == FrameLayout::Planar;
    std::size_t row_bytes = planar ? width : width * n_channels;
    bool packed = src.get_row_stride() == row_bytes &&
                  dst.get_row_stride() == row_bytes;

    if (planar) {
        packed = packed &&
            src.get_channel_stride() == width * height &&
            dst.get_channel_stride() == width * height;
    }

    if (packed ? width * height * n_channels < 4 * 1024 : row_bytes < 256) {
        copy_memcpy(src, dst, packed);
        return;
    }

    copy_sve(src, dst, packed);
}
