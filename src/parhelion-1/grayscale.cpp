#include "parhelion-1.hpp"
#include <arm_sme.h>
#include <arm_sve.h>

namespace {

inline svfloat32x4_t grayscale_load(const std::uint8_t *data, svcount_t pg) __arm_streaming {
    std::size_t svl = svcntw();
    svbool_t p0 = svpext_lane_c32(pg, 0);
    svbool_t p1 = svpext_lane_c32(pg, 1);
    svbool_t p2 = svpext_lane_c32(pg, 2);
    svbool_t p3 = svpext_lane_c32(pg, 3);
    svuint32x4_t v = svcreate4_u32(
        svld1ub_u32(p0, data),
        svld1ub_u32(p1, data + svl),
        svld1ub_u32(p2, data + 2 * svl),
        svld1ub_u32(p3, data + 3 * svl)
    );
    return svcvt_f32_u32_x4(v);
}

inline void grayscale_buffer(
    const std::uint8_t *r,
    const std::uint8_t *g,
    const std::uint8_t *b,
    std::uint8_t *dst,
    std::size_t n
) __arm_streaming __arm_inout("za") {
    std::size_t svl = svcntw();
    std::size_t n_vgx4 = 4 * svl;
    std::size_t n_4xvgx4 = 4 * n_vgx4;
    std::size_t n_full_4xvgx4 = n & ~(n_4xvgx4 - 1);

    std::size_t i = 0;
    svbool_t pu = svptrue_b32();
    svcount_t pg = svptrue_c32();
    svfloat32_t rv = svdup_f32(0.2126f);
    svfloat32_t gv = svdup_f32(0.7152f);
    svfloat32_t bv = svdup_f32(0.0722f);
    for (; i < n_full_4xvgx4; i += n_4xvgx4) {
        svzero_za();
        svfloat32x4_t s0 = grayscale_load(r + i, pg);
        svfloat32x4_t s1 = grayscale_load(r + i + n_vgx4, pg);
        svfloat32x4_t s2 = grayscale_load(r + i + 2 * n_vgx4, pg);
        svfloat32x4_t s3 = grayscale_load(r + i + 3 * n_vgx4, pg);
        svmla_single_za32_f32_vg1x4(0, s0, rv);
        svmla_single_za32_f32_vg1x4(1, s1, rv);
        svmla_single_za32_f32_vg1x4(2, s2, rv);
        svmla_single_za32_f32_vg1x4(3, s3, rv);

        s0 = grayscale_load(g + i, pg);
        s1 = grayscale_load(g + i + n_vgx4, pg);
        s2 = grayscale_load(g + i + 2 * n_vgx4, pg);
        s3 = grayscale_load(g + i + 3 * n_vgx4, pg);
        svmla_single_za32_f32_vg1x4(0, s0, gv);
        svmla_single_za32_f32_vg1x4(1, s1, gv);
        svmla_single_za32_f32_vg1x4(2, s2, gv);
        svmla_single_za32_f32_vg1x4(3, s3, gv);

        s0 = grayscale_load(b + i, pg);
        s1 = grayscale_load(b + i + n_vgx4, pg);
        s2 = grayscale_load(b + i + 2 * n_vgx4, pg);
        s3 = grayscale_load(b + i + 3 * n_vgx4, pg);
        svmla_single_za32_f32_vg1x4(0, s0, bv);
        svmla_single_za32_f32_vg1x4(1, s1, bv);
        svmla_single_za32_f32_vg1x4(2, s2, bv);
        svmla_single_za32_f32_vg1x4(3, s3, bv);

        s0 = svread_za32_f32_vg1x4(0);
        s1 = svread_za32_f32_vg1x4(1);
        s2 = svread_za32_f32_vg1x4(2);
        s3 = svread_za32_f32_vg1x4(3);
        svuint32x4_t u0 = svcvt_u32_f32_x4(s0);
        svuint32x4_t u1 = svcvt_u32_f32_x4(s1);
        svuint32x4_t u2 = svcvt_u32_f32_x4(s2);
        svuint32x4_t u3 = svcvt_u32_f32_x4(s3);

        svst1b_u32(pu, dst + i, svget4_u32(u0, 0));
        svst1b_u32(pu, dst + i + svl, svget4_u32(u0, 1));
        svst1b_u32(pu, dst + i + 2 * svl, svget4_u32(u0, 2));
        svst1b_u32(pu, dst + i + 3 * svl, svget4_u32(u0, 3));
        svst1b_u32(pu, dst + i + n_vgx4, svget4_u32(u1, 0));
        svst1b_u32(pu, dst + i + n_vgx4 + svl, svget4_u32(u1, 1));
        svst1b_u32(pu, dst + i + n_vgx4 + 2 * svl, svget4_u32(u1, 2));
        svst1b_u32(pu, dst + i + n_vgx4 + 3 * svl, svget4_u32(u1, 3));
        svst1b_u32(pu, dst + i + 2 * n_vgx4, svget4_u32(u2, 0));
        svst1b_u32(pu, dst + i + 2 * n_vgx4 + svl, svget4_u32(u2, 1));
        svst1b_u32(pu, dst + i + 2 * n_vgx4 + 2 * svl, svget4_u32(u2, 2));
        svst1b_u32(pu, dst + i + 2 * n_vgx4 + 3 * svl, svget4_u32(u2, 3));
        svst1b_u32(pu, dst + i + 3 * n_vgx4, svget4_u32(u3, 0));
        svst1b_u32(pu, dst + i + 3 * n_vgx4 + svl, svget4_u32(u3, 1));
        svst1b_u32(pu, dst + i + 3 * n_vgx4 + 2 * svl, svget4_u32(u3, 2));
        svst1b_u32(pu, dst + i + 3 * n_vgx4 + 3 * svl, svget4_u32(u3, 3));
    }

    if (i == n)
        return;

    svzero_za();
    pg = svwhilelt_c32_u64(i, n, 4);
    svfloat32x4_t s0 = grayscale_load(r + i, pg);
    pg = svwhilelt_c32_u64(i + n_vgx4, n, 4);
    svfloat32x4_t s1 = grayscale_load(r + i + n_vgx4, pg);
    pg = svwhilelt_c32_u64(i + 2 * n_vgx4, n, 4);
    svfloat32x4_t s2 = grayscale_load(r + i + 2 * n_vgx4, pg);
    pg = svwhilelt_c32_u64(i + 3 * n_vgx4, n, 4);
    svfloat32x4_t s3 = grayscale_load(r + i + 3 * n_vgx4, pg);
    svmla_single_za32_f32_vg1x4(0, s0, rv);
    svmla_single_za32_f32_vg1x4(1, s1, rv);
    svmla_single_za32_f32_vg1x4(2, s2, rv);
    svmla_single_za32_f32_vg1x4(3, s3, rv);

    pg = svwhilelt_c32_u64(i, n, 4);
    s0 = grayscale_load(g + i, pg);
    pg = svwhilelt_c32_u64(i + n_vgx4, n, 4);
    s1 = grayscale_load(g + i + n_vgx4, pg);
    pg = svwhilelt_c32_u64(i + 2 * n_vgx4, n, 4);
    s2 = grayscale_load(g + i + 2 * n_vgx4, pg);
    pg = svwhilelt_c32_u64(i + 3 * n_vgx4, n, 4);
    s3 = grayscale_load(g + i + 3 * n_vgx4, pg);
    svmla_single_za32_f32_vg1x4(0, s0, gv);
    svmla_single_za32_f32_vg1x4(1, s1, gv);
    svmla_single_za32_f32_vg1x4(2, s2, gv);
    svmla_single_za32_f32_vg1x4(3, s3, gv);

    pg = svwhilelt_c32_u64(i, n, 4);
    s0 = grayscale_load(b + i, pg);
    pg = svwhilelt_c32_u64(i + n_vgx4, n, 4);
    s1 = grayscale_load(b + i + n_vgx4, pg);
    pg = svwhilelt_c32_u64(i + 2 * n_vgx4, n, 4);
    s2 = grayscale_load(b + i + 2 * n_vgx4, pg);
    pg = svwhilelt_c32_u64(i + 3 * n_vgx4, n, 4);
    s3 = grayscale_load(b + i + 3 * n_vgx4, pg);
    svmla_single_za32_f32_vg1x4(0, s0, bv);
    svmla_single_za32_f32_vg1x4(1, s1, bv);
    svmla_single_za32_f32_vg1x4(2, s2, bv);
    svmla_single_za32_f32_vg1x4(3, s3, bv);

    s0 = svread_za32_f32_vg1x4(0);
    s1 = svread_za32_f32_vg1x4(1);
    s2 = svread_za32_f32_vg1x4(2);
    s3 = svread_za32_f32_vg1x4(3);
    svuint32x4_t u0 = svcvt_u32_f32_x4(s0);
    svuint32x4_t u1 = svcvt_u32_f32_x4(s1);
    svuint32x4_t u2 = svcvt_u32_f32_x4(s2);
    svuint32x4_t u3 = svcvt_u32_f32_x4(s3);

    pg = svwhilelt_c32_u64(i, n, 4);
    svbool_t p0 = svpext_lane_c32(pg, 0);
    svbool_t p1 = svpext_lane_c32(pg, 1);
    svbool_t p2 = svpext_lane_c32(pg, 2);
    svbool_t p3 = svpext_lane_c32(pg, 3);
    svst1b_u32(p0, dst + i, svget4_u32(u0, 0));
    svst1b_u32(p1, dst + i + svl, svget4_u32(u0, 1));
    svst1b_u32(p2, dst + i + 2 * svl, svget4_u32(u0, 2));
    svst1b_u32(p3, dst + i + 3 * svl, svget4_u32(u0, 3));

    pg = svwhilelt_c32_u64(i + n_vgx4, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svst1b_u32(p0, dst + i + n_vgx4, svget4_u32(u1, 0));
    svst1b_u32(p1, dst + i + n_vgx4 + svl, svget4_u32(u1, 1));
    svst1b_u32(p2, dst + i + n_vgx4 + 2 * svl, svget4_u32(u1, 2));
    svst1b_u32(p3, dst + i + n_vgx4 + 3 * svl, svget4_u32(u1, 3));

    pg = svwhilelt_c32_u64(i + 2 * n_vgx4, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svst1b_u32(p0, dst + i + 2 * n_vgx4, svget4_u32(u2, 0));
    svst1b_u32(p1, dst + i + 2 * n_vgx4 + svl, svget4_u32(u2, 1));
    svst1b_u32(p2, dst + i + 2 * n_vgx4 + 2 * svl, svget4_u32(u2, 2));
    svst1b_u32(p3, dst + i + 2 * n_vgx4 + 3 * svl, svget4_u32(u2, 3));

    pg = svwhilelt_c32_u64(i + 3 * n_vgx4, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svst1b_u32(p0, dst + i + 3 * n_vgx4, svget4_u32(u3, 0));
    svst1b_u32(p1, dst + i + 3 * n_vgx4 + svl, svget4_u32(u3, 1));
    svst1b_u32(p2, dst + i + 3 * n_vgx4 + 2 * svl, svget4_u32(u3, 2));
    svst1b_u32(p3, dst + i + 3 * n_vgx4 + 3 * svl, svget4_u32(u3, 3));
}

__arm_locally_streaming __arm_new("za")
void grayscale_channels(
    FrameViewRef r_frame,
    FrameViewRef g_frame, 
    FrameViewRef b_frame, 
    FrameViewMut dst
) {
    std::size_t width = dst.get_width();
    std::size_t height = dst.get_height();
    std::size_t r_stride = r_frame.get_row_stride();
    std::size_t g_stride = g_frame.get_row_stride();
    std::size_t b_stride = b_frame.get_row_stride();
    std::size_t dst_stride = dst.get_row_stride();

    const std::uint8_t *r = reinterpret_cast<const std::uint8_t *>(r_frame.get_span().data());
    const std::uint8_t *g = reinterpret_cast<const std::uint8_t *>(g_frame.get_span().data());
    const std::uint8_t *b = reinterpret_cast<const std::uint8_t *>(b_frame.get_span().data());
    std::uint8_t *data = reinterpret_cast<std::uint8_t *>(dst.get_span().data());

    // fast path
    if (r_stride == width && g_stride == width && b_stride == width && dst_stride == width) {
        grayscale_buffer(r, g, b, data, width * height);
        return;
    }

    for (std::size_t y = 0; y < height; ++y) {
        grayscale_buffer(
            r + y * r_stride,
            g + y * g_stride,
            b + y * b_stride,
            data + y * dst_stride,
            width
        );
    }
}

} // namespace

void grayscale(FrameViewRef src, FrameViewMut dst) {
    if (src.get_pixel_format() == PixelFormat::Gray)
        throw std::invalid_argument("expected RGB or RGBA source");
    if (dst.get_pixel_format() != PixelFormat::Gray)
        throw std::invalid_argument("expected Gray destination");
    if (src.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
    if (src.get_width() != dst.get_width() || src.get_height() != dst.get_height())
        throw std::invalid_argument("frame dimensions must match");

    FrameViewRef r_frame = src.channel_ref(0);
    FrameViewRef g_frame = src.channel_ref(1);
    FrameViewRef b_frame = src.channel_ref(2);
    grayscale_channels(r_frame, g_frame, b_frame, dst);
}
