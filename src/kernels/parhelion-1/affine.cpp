#include "parhelion-1.hpp"
#include <arm_sme.h>
#include <arm_sve.h>
#include <cmath>

namespace {

inline void affine_strided(
    std::uint8_t *data,
    std::size_t width,
    std::size_t height,
    std::size_t row_stride,
    float alpha,
    float beta
) __arm_streaming {
    const std::size_t lanes = svcntw();
    const svfloat32_t bias = svdup_f32(beta);
    for (std::size_t y = 0; y < height; ++y) {
        std::uint8_t *row = data + y * row_stride;
        for (std::size_t x = 0; x < width; x += lanes) {
            const svbool_t active = svwhilelt_b32_u64(x, width);
            const svfloat32_t values = svcvt_f32_u32_x(active, svld1ub_u32(active, row + x));
            const svfloat32_t transformed = svmla_n_f32_x(active, bias, values, alpha);
            const svuint32_t result = svmin_n_u32_x(active, svcvt_u32_f32_x(active, transformed), 255);
            svst1b_u32(active, row + x, result);
        }
    }
}

inline void affine_buffer(
    std::uint8_t *data,
    std::size_t n,
    float alpha,
    float beta
) __arm_streaming __arm_inout("za") {
    std::size_t svl = svcntw();
    std::size_t n_vgx4 = 4 * svl;
    std::size_t n_4xvgx4 = 4 * n_vgx4;
    std::size_t n_full_4xvgx4 = n & ~(n_4xvgx4 - 1);

    std::size_t i = 0;
    svbool_t pb = svptrue_b32();
    svcount_t pg = svptrue_c32();

    svfloat32_t av = svdup_f32(alpha);
    svfloat32_t bv = svdup_f32(beta);
    svfloat32x4_t bias = svcreate4_f32(bv, bv, bv, bv);
    svuint32_t max = svdup_u32(255);
    for (; i < n_full_4xvgx4;) {
        svwrite_za32_f32_vg1x4(0, bias);
        svwrite_za32_f32_vg1x4(1, bias);
        svwrite_za32_f32_vg1x4(2, bias);
        svwrite_za32_f32_vg1x4(3, bias);

        std::size_t j = i;
        svuint32x4_t v0 = svcreate4_u32(
            svld1ub_u32(pb, data + i),
            svld1ub_u32(pb, data + i + svl),
            svld1ub_u32(pb, data + i + 2 * svl),
            svld1ub_u32(pb, data + i + 3 * svl)
        );
        i += n_vgx4;
        svuint32x4_t v1 = svcreate4_u32(
            svld1ub_u32(pb, data + i),
            svld1ub_u32(pb, data + i + svl),
            svld1ub_u32(pb, data + i + 2 * svl),
            svld1ub_u32(pb, data + i + 3 * svl)
        );
        i += n_vgx4;
        svuint32x4_t v2 = svcreate4_u32(
            svld1ub_u32(pb, data + i),
            svld1ub_u32(pb, data + i + svl),
            svld1ub_u32(pb, data + i + 2 * svl),
            svld1ub_u32(pb, data + i + 3 * svl)
        );
        i += n_vgx4;
        svuint32x4_t v3 = svcreate4_u32(
            svld1ub_u32(pb, data + i),
            svld1ub_u32(pb, data + i + svl),
            svld1ub_u32(pb, data + i + 2 * svl),
            svld1ub_u32(pb, data + i + 3 * svl)
        );
        i += n_vgx4;

        svfloat32x4_t s0 = svcvt_f32_u32_x4(v0);
        svfloat32x4_t s1 = svcvt_f32_u32_x4(v1);
        svfloat32x4_t s2 = svcvt_f32_u32_x4(v2);
        svfloat32x4_t s3 = svcvt_f32_u32_x4(v3);
        svmla_single_za32_f32_vg1x4(0, s0, av);
        svmla_single_za32_f32_vg1x4(1, s1, av);
        svmla_single_za32_f32_vg1x4(2, s2, av);
        svmla_single_za32_f32_vg1x4(3, s3, av);
        s0 = svread_za32_f32_vg1x4(0);
        s1 = svread_za32_f32_vg1x4(1);
        s2 = svread_za32_f32_vg1x4(2);
        s3 = svread_za32_f32_vg1x4(3);
        v0 = svcvt_u32_f32_x4(s0);
        v1 = svcvt_u32_f32_x4(s1);
        v2 = svcvt_u32_f32_x4(s2);
        v3 = svcvt_u32_f32_x4(s3);
        v0 = svmin_single_u32_x4(v0, max);
        v1 = svmin_single_u32_x4(v1, max);
        v2 = svmin_single_u32_x4(v2, max);
        v3 = svmin_single_u32_x4(v3, max);
        svst1b_u32(pb, data + j, svget4_u32(v0, 0));
        svst1b_u32(pb, data + j + svl, svget4_u32(v0, 1));
        svst1b_u32(pb, data + j + 2 * svl, svget4_u32(v0, 2));
        svst1b_u32(pb, data + j + 3 * svl, svget4_u32(v0, 3));
        j += n_vgx4;
        svst1b_u32(pb, data + j, svget4_u32(v1, 0));
        svst1b_u32(pb, data + j + svl, svget4_u32(v1, 1));
        svst1b_u32(pb, data + j + 2 * svl, svget4_u32(v1, 2));
        svst1b_u32(pb, data + j + 3 * svl, svget4_u32(v1, 3));
        j += n_vgx4;
        svst1b_u32(pb, data + j, svget4_u32(v2, 0));
        svst1b_u32(pb, data + j + svl, svget4_u32(v2, 1));
        svst1b_u32(pb, data + j + 2 * svl, svget4_u32(v2, 2));
        svst1b_u32(pb, data + j + 3 * svl, svget4_u32(v2, 3));
        j += n_vgx4;
        svst1b_u32(pb, data + j, svget4_u32(v3, 0));
        svst1b_u32(pb, data + j + svl, svget4_u32(v3, 1));
        svst1b_u32(pb, data + j + 2 * svl, svget4_u32(v3, 2));
        svst1b_u32(pb, data + j + 3 * svl, svget4_u32(v3, 3));
    }

    if (i == n)
        return;

    svwrite_za32_f32_vg1x4(0, bias);
    svwrite_za32_f32_vg1x4(1, bias);
    svwrite_za32_f32_vg1x4(2, bias);
    svwrite_za32_f32_vg1x4(3, bias);
    pg = svwhilelt_c32_u64(i, n, 4);
    svbool_t p0 = svpext_lane_c32(pg, 0);
    svbool_t p1 = svpext_lane_c32(pg, 1);
    svbool_t p2 = svpext_lane_c32(pg, 2);
    svbool_t p3 = svpext_lane_c32(pg, 3);
    std::size_t j = i;
    svuint32x4_t v0 = svcreate4_u32(
        svld1ub_u32(p0, data + i),
        svld1ub_u32(p1, data + i + svl),
        svld1ub_u32(p2, data + i + 2 * svl),
        svld1ub_u32(p3, data + i + 3 * svl)
    );
    i += n_vgx4;
    pg = svwhilelt_c32_u64(i, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svuint32x4_t v1 = svcreate4_u32(
        svld1ub_u32(p0, data + i),
        svld1ub_u32(p1, data + i + svl),
        svld1ub_u32(p2, data + i + 2 * svl),
        svld1ub_u32(p3, data + i + 3 * svl)
    );
    i += n_vgx4;
    pg = svwhilelt_c32_u64(i, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svuint32x4_t v2 = svcreate4_u32(
        svld1ub_u32(p0, data + i),
        svld1ub_u32(p1, data + i + svl),
        svld1ub_u32(p2, data + i + 2 * svl),
        svld1ub_u32(p3, data + i + 3 * svl)
    );
    i += n_vgx4;
    pg = svwhilelt_c32_u64(i, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svuint32x4_t v3 = svcreate4_u32(
        svld1ub_u32(p0, data + i),
        svld1ub_u32(p1, data + i + svl),
        svld1ub_u32(p2, data + i + 2 * svl),
        svld1ub_u32(p3, data + i + 3 * svl)
    );
    i += n_vgx4;

    svfloat32x4_t s0 = svcvt_f32_u32_x4(v0);
    svfloat32x4_t s1 = svcvt_f32_u32_x4(v1);
    svfloat32x4_t s2 = svcvt_f32_u32_x4(v2);
    svfloat32x4_t s3 = svcvt_f32_u32_x4(v3);
    svmla_single_za32_f32_vg1x4(0, s0, av);
    svmla_single_za32_f32_vg1x4(1, s1, av);
    svmla_single_za32_f32_vg1x4(2, s2, av);
    svmla_single_za32_f32_vg1x4(3, s3, av);
    s0 = svread_za32_f32_vg1x4(0);
    s1 = svread_za32_f32_vg1x4(1);
    s2 = svread_za32_f32_vg1x4(2);
    s3 = svread_za32_f32_vg1x4(3);
    v0 = svcvt_u32_f32_x4(s0);
    v1 = svcvt_u32_f32_x4(s1);
    v2 = svcvt_u32_f32_x4(s2);
    v3 = svcvt_u32_f32_x4(s3);
    v0 = svmin_single_u32_x4(v0, max);
    v1 = svmin_single_u32_x4(v1, max);
    v2 = svmin_single_u32_x4(v2, max);
    v3 = svmin_single_u32_x4(v3, max);
    pg = svwhilelt_c32_u64(j, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svst1b_u32(p0, data + j, svget4_u32(v0, 0));
    svst1b_u32(p1, data + j + svl, svget4_u32(v0, 1));
    svst1b_u32(p2, data + j + 2 * svl, svget4_u32(v0, 2));
    svst1b_u32(p3, data + j + 3 * svl, svget4_u32(v0, 3));
    j += n_vgx4;
    pg = svwhilelt_c32_u64(j, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svst1b_u32(p0, data + j, svget4_u32(v1, 0));
    svst1b_u32(p1, data + j + svl, svget4_u32(v1, 1));
    svst1b_u32(p2, data + j + 2 * svl, svget4_u32(v1, 2));
    svst1b_u32(p3, data + j + 3 * svl, svget4_u32(v1, 3));
    j += n_vgx4;
    pg = svwhilelt_c32_u64(j, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svst1b_u32(p0, data + j, svget4_u32(v2, 0));
    svst1b_u32(p1, data + j + svl, svget4_u32(v2, 1));
    svst1b_u32(p2, data + j + 2 * svl, svget4_u32(v2, 2));
    svst1b_u32(p3, data + j + 3 * svl, svget4_u32(v2, 3));
    j += n_vgx4;
    pg = svwhilelt_c32_u64(j, n, 4);
    p0 = svpext_lane_c32(pg, 0);
    p1 = svpext_lane_c32(pg, 1);
    p2 = svpext_lane_c32(pg, 2);
    p3 = svpext_lane_c32(pg, 3);
    svst1b_u32(p0, data + j, svget4_u32(v3, 0));
    svst1b_u32(p1, data + j + svl, svget4_u32(v3, 1));
    svst1b_u32(p2, data + j + 2 * svl, svget4_u32(v3, 2));
    svst1b_u32(p3, data + j + 3 * svl, svget4_u32(v3, 3));
}

__arm_locally_streaming __arm_new("za")
void affine_channel(FrameViewMut frame, float alpha, float beta) {
    std::size_t width = frame.get_width();
    std::size_t height = frame.get_height();
    std::size_t stride = frame.get_row_stride();

    std::uint8_t *data = reinterpret_cast<std::uint8_t *>(frame.get_span().data());

    // fast path
    if (width == stride)
        affine_buffer(data, width * height, alpha, beta);
    else
        affine_strided(data, width, height, stride, alpha, beta);
}

} // namespace

void affine(
    FrameViewMut frame,
    float r_alpha, float g_alpha, float b_alpha,
    float r_beta, float g_beta, float b_beta
) {
    if (!std::isfinite(r_alpha) || !std::isfinite(g_alpha) || !std::isfinite(b_alpha) ||
        r_alpha < 0 || g_alpha < 0 || b_alpha < 0)
        throw std::invalid_argument("scales must be finite and nonnegative");
    if (!std::isfinite(r_beta) || !std::isfinite(g_beta) || !std::isfinite(b_beta))
        throw std::invalid_argument("offsets must be finite");
    if (frame.get_pixel_format() == PixelFormat::Gray)
        throw std::invalid_argument("expected RGB or RGBA frame for per-channel affine");
    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");

    switch (frame.get_pixel_format()) {
        case PixelFormat::RGB:
        case PixelFormat::RGBA: {
            FrameViewMut r_frame = frame.channel_mut(0);
            FrameViewMut g_frame = frame.channel_mut(1);
            FrameViewMut b_frame = frame.channel_mut(2);
            affine_channel(r_frame, r_alpha, r_beta);
            affine_channel(g_frame, g_alpha, g_beta);
            affine_channel(b_frame, b_alpha, b_beta);
            break;
        }

        case PixelFormat::Gray:
            throw std::logic_error("unreachable");
    }
}

void affine(FrameViewMut frame, float alpha, float beta) {
    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
    if (!std::isfinite(alpha) || alpha < 0)
        throw std::invalid_argument("scale must be finite and nonnegative");
    if (!std::isfinite(beta))
        throw std::invalid_argument("offset must be finite");

    switch (frame.get_pixel_format()) {
        case PixelFormat::Gray:
            affine_channel(frame, alpha, beta);
            break;

        case PixelFormat::RGB:
        case PixelFormat::RGBA: {
            FrameViewMut r_frame = frame.channel_mut(0);
            FrameViewMut g_frame = frame.channel_mut(1);
            FrameViewMut b_frame = frame.channel_mut(2);
            affine_channel(r_frame, alpha, beta);
            affine_channel(g_frame, alpha, beta);
            affine_channel(b_frame, alpha, beta);
            break;
        }
    }
}
