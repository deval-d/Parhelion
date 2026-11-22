#include "image_types.hpp"
#include "color_types.hpp"
#include <arm_sve.h>
#include <cstring>

namespace {
    inline void fill_sve_buffer(
        std::uint8_t *data,
        std::size_t n,
        std::byte value
    ) __arm_streaming {
        std::size_t svl_b = svcntb(); 
        std::size_t n_VGx4 = 4 * svl_b; 
        std::size_t n_4xVGx4 = 4 * n_VGx4; 
        std::size_t n_full_4xVGx4 = n & ~(n_4xVGx4 - 1); 

        svuint8_t v = svdup_n_u8(std::to_integer<std::uint8_t>(value));
        svuint8x4_t vx4 = svcreate4_u8(v, v, v, v); 
        
        std::size_t i = 0; 
        svcount_t pg = svptrue_c8(); 
        for (; i < n_full_4xVGx4; i += n_4xVGx4) { 
            svst1_u8_x4(pg, data + i, vx4);
            svst1_u8_x4(pg, data + i + n_VGx4, vx4);
            svst1_u8_x4(pg, data + i + 2 * n_VGx4, vx4);
            svst1_u8_x4(pg, data + i + 3 * n_VGx4, vx4); 
        }

        if (i == n) 
            return; 

        svcount_t pg1 = svwhilelt_c8_u64(i, n, 4); 
        svcount_t pg2 = svwhilelt_c8_u64(i + n_VGx4, n, 4);
        svcount_t pg3 = svwhilelt_c8_u64(i + 2 * n_VGx4, n, 4);
        svcount_t pg4 = svwhilelt_c8_u64(i + 3 * n_VGx4, n, 4); 
        svst1_u8_x4(pg1, data + i, vx4);
        svst1_u8_x4(pg2, data + i + n_VGx4, vx4);
        svst1_u8_x4(pg3, data + i + 2 * n_VGx4, vx4);
        svst1_u8_x4(pg4, data + i + 3 * n_VGx4, vx4); 
    }

    __arm_locally_streaming
    void fill_sve(FrameViewMut frame, Color::Gray color) {
        std::size_t width = frame.get_width();
        std::size_t height = frame.get_height();
        std::size_t row_stride = frame.get_row_stride();
        std::uint8_t *data = reinterpret_cast<std::uint8_t *>(frame.get_span().data());

        if (row_stride == width) {
            fill_sve_buffer(data, width * height, color.v);
            return;
        }

        for (std::size_t y = 0; y < height; ++y)
            fill_sve_buffer(data + y * row_stride, width, color.v);
    }

    void fill_memset(FrameViewMut frame, Color::Gray color) {
        std::size_t width = frame.get_width();
        std::size_t height = frame.get_height();
        std::size_t row_stride = frame.get_row_stride();
        std::byte *data = frame.get_span().data();
        int value = std::to_integer<int>(color.v);

        if (row_stride == width) {
            std::memset(data, value, width * height);
            return;
        }

        for (std::size_t y = 0; y < height; ++y)
            std::memset(data + y * row_stride, value, width);
    }
}

void fill(FrameViewMut frame, Color::Gray color) {
    std::size_t width = frame.get_width();
    std::size_t height = frame.get_height();
    bool packed = frame.get_row_stride() == width;

    if (width == 0 || height == 0)
        return;

    if (color.v == std::byte(0) ||
        (packed ? width * height < 2 * 1024 : width < 256)) {
        fill_memset(frame, color);
        return;
    }

    fill_sve(frame, color);
}

void fill(FrameViewMut frame, Color::RGB color) {
    fill(frame.channel_mut(0), Color::Gray(color.r));
    fill(frame.channel_mut(1), Color::Gray(color.g));
    fill(frame.channel_mut(2), Color::Gray(color.b));
}

void fill(FrameViewMut frame, Color::RGBA color) {
    fill(frame.channel_mut(0), Color::Gray(color.r));
    fill(frame.channel_mut(1), Color::Gray(color.g));
    fill(frame.channel_mut(2), Color::Gray(color.b));
    fill(frame.channel_mut(3), Color::Gray(color.a));
}
