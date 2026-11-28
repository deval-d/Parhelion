#include "parhelion-1.hpp"
#include <arm_sve.h> 

namespace {

void threshold_buffer(
    std::uint8_t *data,
    std::size_t n,
    std::uint8_t t 
) __arm_streaming {
    std::size_t svl_b = svcntb();
    std::size_t n_vgx4 = 4 * svl_b;
    std::size_t n_4xvgx4 = 4 * n_vgx4;
    std::size_t n_full_4xvgx4 = n & ~(n_4xvgx4 - 1);

    std::size_t i = 0;
    svbool_t pb = svptrue_b8();
    svcount_t pg = svptrue_c8();
    for (; i < n_full_4xvgx4; i += n_4xvgx4) {
        svuint8x4_t v0 = svld1_u8_x4(pg, data + i);
        svuint8_t v00 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v0, 0), t), 255);
        svuint8_t v01 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v0, 1), t), 255);
        svuint8_t v02 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v0, 2), t), 255);
        svuint8_t v03 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v0, 3), t), 255);
        v0 = svcreate4_u8(v00, v01, v02, v03);
        svst1_u8_x4(pg, data + i, v0);

        svuint8x4_t v1 = svld1_u8_x4(pg, data + i + n_vgx4);
        svuint8_t v10 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v1, 0), t), 255);
        svuint8_t v11 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v1, 1), t), 255);
        svuint8_t v12 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v1, 2), t), 255);
        svuint8_t v13 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v1, 3), t), 255);
        v1 = svcreate4_u8(v10, v11, v12, v13);
        svst1_u8_x4(pg, data + i + n_vgx4, v1);

        svuint8x4_t v2 = svld1_u8_x4(pg, data + i + 2 * n_vgx4);
        svuint8_t v20 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v2, 0), t), 255);
        svuint8_t v21 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v2, 1), t), 255);
        svuint8_t v22 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v2, 2), t), 255);
        svuint8_t v23 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v2, 3), t), 255);
        v2 = svcreate4_u8(v20, v21, v22, v23);
        svst1_u8_x4(pg, data + i + 2 * n_vgx4, v2);

        svuint8x4_t v3 = svld1_u8_x4(pg, data + i + 3 * n_vgx4);
        svuint8_t v30 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v3, 0), t), 255);
        svuint8_t v31 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v3, 1), t), 255);
        svuint8_t v32 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v3, 2), t), 255);
        svuint8_t v33 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v3, 3), t), 255);
        v3 = svcreate4_u8(v30, v31, v32, v33);
        svst1_u8_x4(pg, data + i + 3 * n_vgx4, v3);
    }

    if (i == n)
        return;

    svcount_t p0 = svwhilelt_c8_u64(i, n, 4);
    svcount_t p1 = svwhilelt_c8_u64(i + n_vgx4, n, 4);
    svcount_t p2 = svwhilelt_c8_u64(i + 2 * n_vgx4, n, 4);
    svcount_t p3 = svwhilelt_c8_u64(i + 3 * n_vgx4, n, 4);
    svuint8x4_t v0 = svld1_u8_x4(p0, data + i);
    svuint8_t v00 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v0, 0), t), 255);
    svuint8_t v01 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v0, 1), t), 255);
    svuint8_t v02 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v0, 2), t), 255);
    svuint8_t v03 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v0, 3), t), 255);
    v0 = svcreate4_u8(v00, v01, v02, v03);
    svst1_u8_x4(p0, data + i, v0);

    svuint8x4_t v1 = svld1_u8_x4(p1, data + i + n_vgx4);
    svuint8_t v10 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v1, 0), t), 255);
    svuint8_t v11 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v1, 1), t), 255);
    svuint8_t v12 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v1, 2), t), 255);
    svuint8_t v13 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v1, 3), t), 255);
    v1 = svcreate4_u8(v10, v11, v12, v13);
    svst1_u8_x4(p1, data + i + n_vgx4, v1);

    svuint8x4_t v2 = svld1_u8_x4(p2, data + i + 2 * n_vgx4);
    svuint8_t v20 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v2, 0), t), 255);
    svuint8_t v21 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v2, 1), t), 255);
    svuint8_t v22 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v2, 2), t), 255);
    svuint8_t v23 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v2, 3), t), 255);
    v2 = svcreate4_u8(v20, v21, v22, v23);
    svst1_u8_x4(p2, data + i + 2 * n_vgx4, v2);

    svuint8x4_t v3 = svld1_u8_x4(p3, data + i + 3 * n_vgx4);
    svuint8_t v30 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v3, 0), t), 255);
    svuint8_t v31 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v3, 1), t), 255);
    svuint8_t v32 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v3, 2), t), 255);
    svuint8_t v33 = svdup_n_u8_z(svcmpgt_n_u8(pb, svget4_u8(v3, 3), t), 255);
    v3 = svcreate4_u8(v30, v31, v32, v33);
    svst1_u8_x4(p3, data + i + 3 * n_vgx4, v3);
}

__arm_locally_streaming
void threshold_channel(FrameViewMut frame, std::uint8_t t) { 
    std::size_t width = frame.get_width();
    std::size_t height = frame.get_height();
    std::size_t stride = frame.get_row_stride();

    std::uint8_t *data = reinterpret_cast<std::uint8_t *>(frame.get_span().data()); 

    if (width == stride)
        threshold_buffer(data, width * height, t); 
    else { 
        for (std::size_t y = 0; y < height; ++y) { 
            threshold_buffer(data + y * stride, width, t); 
        }
    }
}

} // namespace

void threshold(FrameViewMut frame, std::uint8_t rt, std::uint8_t gt, std::uint8_t bt) { 
    if (frame.get_pixel_format() == PixelFormat::Gray)
        throw std::invalid_argument("expected RGB or RGBA frame for per-channel threshold"); 
    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");

    switch (frame.get_pixel_format()) { 
        case PixelFormat::RGB:
        case PixelFormat::RGBA: { 
            FrameViewMut r_frame = frame.channel_mut(0);
            FrameViewMut g_frame = frame.channel_mut(1);
            FrameViewMut b_frame = frame.channel_mut(2);
            threshold_channel(r_frame, rt); 
            threshold_channel(g_frame, gt);
            threshold_channel(b_frame, bt);
            break; 
        }

        case PixelFormat::Gray:
            throw std::logic_error("unreachable"); 
    }
}

void threshold(FrameViewMut frame, std::uint8_t t) { 
    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout"); 

    switch (frame.get_pixel_format()) { 
        case PixelFormat::Gray:
            threshold_channel(frame, t);
            break;

        case PixelFormat::RGB:
        case PixelFormat::RGBA: { 
            FrameViewMut r_frame = frame.channel_mut(0);
            FrameViewMut g_frame = frame.channel_mut(1);
            FrameViewMut b_frame = frame.channel_mut(2);
            threshold_channel(r_frame, t); 
            threshold_channel(g_frame, t);
            threshold_channel(b_frame, t);
            break;
        }
    }
}

