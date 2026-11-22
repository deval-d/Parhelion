#include "image_types.hpp"
#include <stdexcept> 
#include <arm_sve.h>

inline void offset_p(
    std::uint8_t *data,
    std::size_t n, 
    std::uint8_t offset
) __arm_streaming { 
    std::size_t svl_b = svcntb();
    std::size_t n_vgx4 = 4 * svl_b;
    std::size_t n_full_vgx4 = n & ~(n_vgx4 - 1); 

    std::size_t i = 0;
    svcount_t pg = svptrue_c8(); 
    for (; i < n_full_vgx4; i += n_vgx4) { 
        svuint8x4_t v = svld1_u8_x4(pg, data + i); 
        svuint8_t v0 = svqadd_n_u8(svget4_u8(v, 0), offset); 
        svuint8_t v1 = svqadd_n_u8(svget4_u8(v, 1), offset);
        svuint8_t v2 = svqadd_n_u8(svget4_u8(v, 2), offset);
        svuint8_t v3 = svqadd_n_u8(svget4_u8(v, 3), offset); 
        v = svcreate4_u8(v0, v1, v2, v3); 
        svst1_u8_x4(pg, data + i, v);  
    }

    if (i == n) 
        return; 

    svcount_t p = svwhilelt_c8_u64(i, n, 4); 
    svuint8x4_t v = svld1_u8_x4(p, data + i); 
    svuint8_t v0 = svqadd_n_u8(svget4_u8(v, 0), offset); 
    svuint8_t v1 = svqadd_n_u8(svget4_u8(v, 1), offset);
    svuint8_t v2 = svqadd_n_u8(svget4_u8(v, 2), offset);
    svuint8_t v3 = svqadd_n_u8(svget4_u8(v, 3), offset); 
    v = svcreate4_u8(v0, v1, v2, v3); 
    svst1_u8_x4(p, data + i, v);  
}

inline void offset_m(
    std::uint8_t *data,
    std::size_t n, 
    std::uint8_t offset
) __arm_streaming { 
    std::size_t svl_b = svcntb();
    std::size_t n_vgx4 = 4 * svl_b;
    std::size_t n_full_vgx4 = n & ~(n_vgx4 - 1); 

    std::size_t i = 0;
    svcount_t pg = svptrue_c8(); 
    for (; i < n_full_vgx4; i += n_vgx4) { 
        svuint8x4_t v = svld1_u8_x4(pg, data + i); 
        svuint8_t v0 = svqsub_n_u8(svget4_u8(v, 0), offset); 
        svuint8_t v1 = svqsub_n_u8(svget4_u8(v, 1), offset);
        svuint8_t v2 = svqsub_n_u8(svget4_u8(v, 2), offset);
        svuint8_t v3 = svqsub_n_u8(svget4_u8(v, 3), offset); 
        v = svcreate4_u8(v0, v1, v2, v3); 
        svst1_u8_x4(pg, data + i, v);  
    }

    if (i == n) 
        return; 

    svcount_t p = svwhilelt_c8_u64(i, n, 4); 
    svuint8x4_t v = svld1_u8_x4(p, data + i); 
    svuint8_t v0 = svqsub_n_u8(svget4_u8(v, 0), offset); 
    svuint8_t v1 = svqsub_n_u8(svget4_u8(v, 1), offset);
    svuint8_t v2 = svqsub_n_u8(svget4_u8(v, 2), offset);
    svuint8_t v3 = svqsub_n_u8(svget4_u8(v, 3), offset); 
    v = svcreate4_u8(v0, v1, v2, v3); 
    svst1_u8_x4(p, data + i, v);  
}


__arm_locally_streaming 
void offset(FrameViewMut frame, int amount) {
    bool positive = amount >= 0; 
    
    std::size_t width = frame.get_width();
    std::size_t height = frame.get_height(); 
    std::size_t stride = frame.get_row_stride();

    std::uint8_t *data = reinterpret_cast<std::uint8_t *>(frame.get_span().data());

    // fast path 
    if (width == stride) { 
        if (positive) 
            offset_p(data, width * height, amount);
        else 
            offset_m(data, width * height, -amount);
        return; 
    } 

    for (std::size_t y = 0; y < height; ++y) { 
        if (positive) 
            offset_p(data + y * stride, width, amount); 
        else 
            offset_m(data + y * stride, width, -amount); 
    }
}


void brighten(FrameViewMut frame, int amount) { 
    if (amount < -255 || amount > 255)
        throw std::invalid_argument("brightness offset must be in [-255, 255]");

    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout"); 

    switch (frame.get_pixel_format()) { 
        case PixelFormat::Gray:
            offset(frame, amount);
            break;

        case PixelFormat::RGB:
        case PixelFormat::RGBA: 
            FrameViewMut r_frame = frame.channel_mut(0); 
            FrameViewMut g_frame = frame.channel_mut(1);
            FrameViewMut b_frame = frame.channel_mut(2); 
            offset(r_frame, amount);
            offset(g_frame, amount);
            offset(b_frame, amount); 
            break;
    }
}


void brighten(FrameViewMut frame, int r_amount, int g_amount, int b_amount) { 
    if (frame.get_pixel_format() == PixelFormat::Gray)
        throw std::invalid_argument("expected RGB or RGBA frame for per-channel brightness"); 
    if (frame.get_frame_layout() != FrameLayout::Planar) 
        throw std::invalid_argument("expected Planar layout"); 
    if ((r_amount < -255 || r_amount > 255) || 
        (g_amount < -255 || g_amount > 255) || 
        (b_amount < -255 || b_amount > 255))
        throw std::invalid_argument("brightness offsets must be in [-255, 255]");

    switch (frame.get_pixel_format()) { 
        case PixelFormat::RGB:
        case PixelFormat::RGBA: { 
            FrameViewMut r_frame = frame.channel_mut(0); 
            FrameViewMut g_frame = frame.channel_mut(1);
            FrameViewMut b_frame = frame.channel_mut(2); 
            offset(r_frame, r_amount);
            offset(g_frame, g_amount);
            offset(b_frame, b_amount); 
            break;
        }

        case PixelFormat::Gray:
            throw std::logic_error("unreachable"); 
    }
}
