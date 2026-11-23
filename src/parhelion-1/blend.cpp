#include "image_types.hpp"
#include <arm_sme.h>
#include <arm_sve.h>
#include <cmath>

inline void blend_buffer(
    float alpha,
    const std::uint8_t *src1,
    const std::uint8_t *src2,
    std::uint8_t *dst,
    std::size_t n
) __arm_streaming __arm_inout("za") {
    svfloat32_t av = svdup_f32(alpha);
    svfloat32_t bv = svdup_f32(1.0f - alpha);

    std::size_t svl = svcntw();
    std::size_t n_vgx4 = 4 * svl;
    std::size_t n_4xvgx4 = 4 * n_vgx4;
    std::size_t n_full_4xvgx4 = n & ~(n_4xvgx4 - 1);

    std::size_t i = 0;
    svbool_t pu = svptrue_b32();
    for (; i < n_full_4xvgx4;) {
        svzero_za();

        std::size_t j = i;
        std::size_t k = i;
        svuint32x4_t u0 = svcreate4_u32(
            svld1ub_u32(pu, src1 + i),
            svld1ub_u32(pu, src1 + i + svl),
            svld1ub_u32(pu, src1 + i + 2 * svl),
            svld1ub_u32(pu, src1 + i + 3 * svl)
        );
        i += n_vgx4;
        svuint32x4_t u1 = svcreate4_u32(
            svld1ub_u32(pu, src1 + i),
            svld1ub_u32(pu, src1 + i + svl),
            svld1ub_u32(pu, src1 + i + 2 * svl),
            svld1ub_u32(pu, src1 + i + 3 * svl)
        );
        i += n_vgx4;
        svuint32x4_t u2 = svcreate4_u32(
            svld1ub_u32(pu, src1 + i),
            svld1ub_u32(pu, src1 + i + svl),
            svld1ub_u32(pu, src1 + i + 2 * svl),
            svld1ub_u32(pu, src1 + i + 3 * svl)
        );
        i += n_vgx4;
        svuint32x4_t u3 = svcreate4_u32(
            svld1ub_u32(pu, src1 + i),
            svld1ub_u32(pu, src1 + i + svl),
            svld1ub_u32(pu, src1 + i + 2 * svl),
            svld1ub_u32(pu, src1 + i + 3 * svl)
        );
        i += n_vgx4;

        svfloat32x4_t s0 = svcvt_f32_u32_x4(u0);
        svfloat32x4_t s1 = svcvt_f32_u32_x4(u1);
        svfloat32x4_t s2 = svcvt_f32_u32_x4(u2);
        svfloat32x4_t s3 = svcvt_f32_u32_x4(u3);
        svmla_single_za32_f32_vg1x4(0, s0, av);
        svmla_single_za32_f32_vg1x4(1, s1, av);
        svmla_single_za32_f32_vg1x4(2, s2, av);
        svmla_single_za32_f32_vg1x4(3, s3, av);

        u0 = svcreate4_u32(
            svld1ub_u32(pu, src2 + j),
            svld1ub_u32(pu, src2 + j + svl),
            svld1ub_u32(pu, src2 + j + 2 * svl),
            svld1ub_u32(pu, src2 + j + 3 * svl)
        );
        j += n_vgx4;
        u1 = svcreate4_u32(
            svld1ub_u32(pu, src2 + j),
            svld1ub_u32(pu, src2 + j + svl),
            svld1ub_u32(pu, src2 + j + 2 * svl),
            svld1ub_u32(pu, src2 + j + 3 * svl)
        );
        j += n_vgx4;
        u2 = svcreate4_u32(
            svld1ub_u32(pu, src2 + j),
            svld1ub_u32(pu, src2 + j + svl),
            svld1ub_u32(pu, src2 + j + 2 * svl),
            svld1ub_u32(pu, src2 + j + 3 * svl)
        );
        j += n_vgx4;
        u3 = svcreate4_u32(
            svld1ub_u32(pu, src2 + j),
            svld1ub_u32(pu, src2 + j + svl),
            svld1ub_u32(pu, src2 + j + 2 * svl),
            svld1ub_u32(pu, src2 + j + 3 * svl)
        );
        s0 = svcvt_f32_u32_x4(u0);
        s1 = svcvt_f32_u32_x4(u1);
        s2 = svcvt_f32_u32_x4(u2);
        s3 = svcvt_f32_u32_x4(u3);
        svmla_single_za32_f32_vg1x4(0, s0, bv);
        svmla_single_za32_f32_vg1x4(1, s1, bv);
        svmla_single_za32_f32_vg1x4(2, s2, bv);
        svmla_single_za32_f32_vg1x4(3, s3, bv);
        s0 = svread_za32_f32_vg1x4(0);
        s1 = svread_za32_f32_vg1x4(1);
        s2 = svread_za32_f32_vg1x4(2);
        s3 = svread_za32_f32_vg1x4(3);
        u0 = svcvt_u32_f32_x4(s0);
        u1 = svcvt_u32_f32_x4(s1);
        u2 = svcvt_u32_f32_x4(s2);
        u3 = svcvt_u32_f32_x4(s3);

        svst1b_u32(pu, dst + k, svget4_u32(u0, 0));
        svst1b_u32(pu, dst + k + svl, svget4_u32(u0, 1));
        svst1b_u32(pu, dst + k + 2 * svl, svget4_u32(u0, 2));
        svst1b_u32(pu, dst + k + 3 * svl, svget4_u32(u0, 3));
        k += n_vgx4;
        svst1b_u32(pu, dst + k, svget4_u32(u1, 0));
        svst1b_u32(pu, dst + k + svl, svget4_u32(u1, 1));
        svst1b_u32(pu, dst + k + 2 * svl, svget4_u32(u1, 2));
        svst1b_u32(pu, dst + k + 3 * svl, svget4_u32(u1, 3));
        k += n_vgx4;
        svst1b_u32(pu, dst + k, svget4_u32(u2, 0));
        svst1b_u32(pu, dst + k + svl, svget4_u32(u2, 1));
        svst1b_u32(pu, dst + k + 2 * svl, svget4_u32(u2, 2));
        svst1b_u32(pu, dst + k + 3 * svl, svget4_u32(u2, 3));
        k += n_vgx4;
        svst1b_u32(pu, dst + k, svget4_u32(u3, 0));
        svst1b_u32(pu, dst + k + svl, svget4_u32(u3, 1));
        svst1b_u32(pu, dst + k + 2 * svl, svget4_u32(u3, 2));
        svst1b_u32(pu, dst + k + 3 * svl, svget4_u32(u3, 3));
    }

    if (i == n)
        return;

    svzero_za();
    svcount_t pg = svwhilelt_c32_u64(i, n, 4);
    svbool_t p0 = svpext_lane_c32(pg, 0);
    svbool_t p1 = svpext_lane_c32(pg, 1);
    svbool_t p2 = svpext_lane_c32(pg, 2);
    svbool_t p3 = svpext_lane_c32(pg, 3);
    std::size_t j = i;
    std::size_t k = i;
    svuint32x4_t u0 = svcreate4_u32(
        svld1ub_u32(p0, src1 + i),
        svld1ub_u32(p1, src1 + i + svl),
        svld1ub_u32(p2, src1 + i + 2 * svl),
        svld1ub_u32(p3, src1 + i + 3 * svl)
    );
    i += n_vgx4;
    pg = svwhilelt_c32_u64(i, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svuint32x4_t u1 = svcreate4_u32(
        svld1ub_u32(p0, src1 + i),
        svld1ub_u32(p1, src1 + i + svl),
        svld1ub_u32(p2, src1 + i + 2 * svl),
        svld1ub_u32(p3, src1 + i + 3 * svl)
    );
    i += n_vgx4;
    pg = svwhilelt_c32_u64(i, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svuint32x4_t u2 = svcreate4_u32(
        svld1ub_u32(p0, src1 + i),
        svld1ub_u32(p1, src1 + i + svl),
        svld1ub_u32(p2, src1 + i + 2 * svl),
        svld1ub_u32(p3, src1 + i + 3 * svl)
    );
    i += n_vgx4;
    pg = svwhilelt_c32_u64(i, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svuint32x4_t u3 = svcreate4_u32(
        svld1ub_u32(p0, src1 + i),
        svld1ub_u32(p1, src1 + i + svl),
        svld1ub_u32(p2, src1 + i + 2 * svl),
        svld1ub_u32(p3, src1 + i + 3 * svl)
    );
    i += n_vgx4;

    svfloat32x4_t s0 = svcvt_f32_u32_x4(u0);
    svfloat32x4_t s1 = svcvt_f32_u32_x4(u1);
    svfloat32x4_t s2 = svcvt_f32_u32_x4(u2);
    svfloat32x4_t s3 = svcvt_f32_u32_x4(u3);
    svmla_single_za32_f32_vg1x4(0, s0, av);
    svmla_single_za32_f32_vg1x4(1, s1, av);
    svmla_single_za32_f32_vg1x4(2, s2, av);
    svmla_single_za32_f32_vg1x4(3, s3, av);

    pg = svwhilelt_c32_u64(j, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    u0 = svcreate4_u32(
        svld1ub_u32(p0, src2 + j),
        svld1ub_u32(p1, src2 + j + svl),
        svld1ub_u32(p2, src2 + j + 2 * svl),
        svld1ub_u32(p3, src2 + j + 3 * svl)
    );
    j += n_vgx4;
    pg = svwhilelt_c32_u64(j, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    u1 = svcreate4_u32(
        svld1ub_u32(p0, src2 + j),
        svld1ub_u32(p1, src2 + j + svl),
        svld1ub_u32(p2, src2 + j + 2 * svl),
        svld1ub_u32(p3, src2 + j + 3 * svl)
    );
    j += n_vgx4;
    pg = svwhilelt_c32_u64(j, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    u2 = svcreate4_u32(
        svld1ub_u32(p0, src2 + j),
        svld1ub_u32(p1, src2 + j + svl),
        svld1ub_u32(p2, src2 + j + 2 * svl),
        svld1ub_u32(p3, src2 + j + 3 * svl)
    );
    j += n_vgx4;
    pg = svwhilelt_c32_u64(j, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    u3 = svcreate4_u32(
        svld1ub_u32(p0, src2 + j),
        svld1ub_u32(p1, src2 + j + svl),
        svld1ub_u32(p2, src2 + j + 2 * svl),
        svld1ub_u32(p3, src2 + j + 3 * svl)
    );
    s0 = svcvt_f32_u32_x4(u0);
    s1 = svcvt_f32_u32_x4(u1);
    s2 = svcvt_f32_u32_x4(u2);
    s3 = svcvt_f32_u32_x4(u3);
    svmla_single_za32_f32_vg1x4(0, s0, bv);
    svmla_single_za32_f32_vg1x4(1, s1, bv);
    svmla_single_za32_f32_vg1x4(2, s2, bv);
    svmla_single_za32_f32_vg1x4(3, s3, bv);
    s0 = svread_za32_f32_vg1x4(0);
    s1 = svread_za32_f32_vg1x4(1);
    s2 = svread_za32_f32_vg1x4(2);
    s3 = svread_za32_f32_vg1x4(3);
    u0 = svcvt_u32_f32_x4(s0);
    u1 = svcvt_u32_f32_x4(s1);
    u2 = svcvt_u32_f32_x4(s2);
    u3 = svcvt_u32_f32_x4(s3);

    pg = svwhilelt_c32_u64(k, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svst1b_u32(p0, dst + k, svget4_u32(u0, 0));
    svst1b_u32(p1, dst + k + svl, svget4_u32(u0, 1));
    svst1b_u32(p2, dst + k + 2 * svl, svget4_u32(u0, 2));
    svst1b_u32(p3, dst + k + 3 * svl, svget4_u32(u0, 3));
    k += n_vgx4;
    pg = svwhilelt_c32_u64(k, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svst1b_u32(p0, dst + k, svget4_u32(u1, 0));
    svst1b_u32(p1, dst + k + svl, svget4_u32(u1, 1));
    svst1b_u32(p2, dst + k + 2 * svl, svget4_u32(u1, 2));
    svst1b_u32(p3, dst + k + 3 * svl, svget4_u32(u1, 3));
    k += n_vgx4;
    pg = svwhilelt_c32_u64(k, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svst1b_u32(p0, dst + k, svget4_u32(u2, 0));
    svst1b_u32(p1, dst + k + svl, svget4_u32(u2, 1));
    svst1b_u32(p2, dst + k + 2 * svl, svget4_u32(u2, 2));
    svst1b_u32(p3, dst + k + 3 * svl, svget4_u32(u2, 3));
    k += n_vgx4;
    pg = svwhilelt_c32_u64(k, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svst1b_u32(p0, dst + k, svget4_u32(u3, 0));
    svst1b_u32(p1, dst + k + svl, svget4_u32(u3, 1));
    svst1b_u32(p2, dst + k + 2 * svl, svget4_u32(u3, 2));
    svst1b_u32(p3, dst + k + 3 * svl, svget4_u32(u3, 3));
}


__arm_locally_streaming __arm_new("za")
void blend_channel(
    FrameViewRef src1,
    FrameViewRef src2,
    FrameViewMut dst,
    float alpha,
    std::size_t channel
) {
    std::size_t width = dst.get_width();
    std::size_t height = dst.get_height();
    std::size_t src1_stride = src1.get_row_stride();
    std::size_t src2_stride = src2.get_row_stride();
    std::size_t dst_stride = dst.get_row_stride();

    const std::uint8_t *src1_ptr = reinterpret_cast<const std::uint8_t *>(src1.get_span().data());
    const std::uint8_t *src2_ptr = reinterpret_cast<const std::uint8_t *>(src2.get_span().data());
    std::uint8_t *dst_ptr = reinterpret_cast<std::uint8_t *>(dst.get_span().data());
    if (channel != 0) {
        src1_ptr += channel * src1.get_channel_stride().value();
        src2_ptr += channel * src2.get_channel_stride().value();
        dst_ptr += channel * dst.get_channel_stride().value();
    }

    // fast path
    if (src1_stride == width && src2_stride == width && dst_stride == width) {
        blend_buffer(alpha, src1_ptr, src2_ptr, dst_ptr, width * height);
        return;
    }

    for (std::size_t y = 0; y < height; ++y) {
        blend_buffer(
            alpha,
            src1_ptr + y * src1_stride,
            src2_ptr + y * src2_stride,
            dst_ptr + y * dst_stride,
            width
        );
    }
}

void blend(
    FrameViewRef src1,
    FrameViewRef src2,
    FrameViewMut dst,
    float r_alpha, float g_alpha, float b_alpha
) {
    if (!std::isfinite(r_alpha) || !std::isfinite(g_alpha) || !std::isfinite(b_alpha) ||
        r_alpha < 0 || r_alpha > 1 || g_alpha < 0 || g_alpha > 1 || b_alpha < 0 || b_alpha > 1)
        throw std::invalid_argument("blend weights must be finite and in [0, 1]");
    if (src1.get_width() != dst.get_width() || src2.get_width() != dst.get_width() ||
        src1.get_height() != dst.get_height() || src2.get_height() != dst.get_height())
        throw std::invalid_argument("frame dimensions must match");
    if (src1.get_pixel_format() != dst.get_pixel_format() ||
        src2.get_pixel_format() != dst.get_pixel_format())
        throw std::invalid_argument("pixel formats must match");
    if (src1.get_frame_layout() != FrameLayout::Planar ||
        src2.get_frame_layout() != FrameLayout::Planar || dst.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
    if (dst.get_pixel_format() == PixelFormat::Gray)
        throw std::invalid_argument("expected RGB or RGBA frame for per-channel blending");

    switch (dst.get_pixel_format()) {
        case PixelFormat::RGB:
        case PixelFormat::RGBA:
            blend_channel(src1, src2, dst, r_alpha, 0);
            blend_channel(src1, src2, dst, g_alpha, 1);
            blend_channel(src1, src2, dst, b_alpha, 2);
            break;

        case PixelFormat::Gray:
            throw std::logic_error("unreachable");
    }
}

void blend(FrameViewRef src1, FrameViewRef src2, FrameViewMut dst, float alpha) {
    if (!std::isfinite(alpha) || alpha < 0 || alpha > 1)
        throw std::invalid_argument("blend weight must be finite and in [0, 1]");
    if (src1.get_width() != dst.get_width() || src2.get_width() != dst.get_width() ||
        src1.get_height() != dst.get_height() || src2.get_height() != dst.get_height())
        throw std::invalid_argument("frame dimensions must match");
    if (src1.get_pixel_format() != dst.get_pixel_format() ||
        src2.get_pixel_format() != dst.get_pixel_format())
        throw std::invalid_argument("pixel formats must match");
    if (src1.get_frame_layout() != FrameLayout::Planar ||
        src2.get_frame_layout() != FrameLayout::Planar || dst.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");

    switch (dst.get_pixel_format()) {
        case PixelFormat::Gray:
            blend_channel(src1, src2, dst, alpha, 0);
            break;

        case PixelFormat::RGB:
        case PixelFormat::RGBA:
            blend_channel(src1, src2, dst, alpha, 0);
            blend_channel(src1, src2, dst, alpha, 1);
            blend_channel(src1, src2, dst, alpha, 2);
            break;
    }
}
