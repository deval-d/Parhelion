#include "parhelion-1.hpp"
#include <arm_sme.h>
#include <arm_sve.h>
#include <cmath>

namespace {

inline void scale_strided(
    std::uint8_t *data,
    std::size_t width,
    std::size_t height,
    std::size_t row_stride,
    float alpha 
) __arm_streaming __arm_inout("za") { 
    uint32_t svl = svcntw(); 
    uint32_t n_vgx4 = 4 * svl;

    uint32_t i = 0;
    std::size_t offs = 0; 
    svfloat32_t alphav = svdup_n_f32(alpha);
    svuint32_t maxv = svdup_n_u32(255); 

    svzero_za(); 
    for (; i < 256; i += n_vgx4) { 
        svuint32x4_t idxs = svcreate4_u32(
            svindex_u32(i, 1),
            svindex_u32(i + svl, 1),
            svindex_u32(i + 2 * svl, 1),
            svindex_u32(i + 3 * svl, 1)
        ); 

        svfloat32x4_t vgx4 = svcvt_f32_u32_x4(idxs); 
        svmla_single_za32_f32_vg1x4(offs, vgx4, alphav); 
        offs += 4; 
    }

    svfloat32x4_t v0 = svread_za32_f32_vg1x4(0);
    svfloat32x4_t v1 = svread_za32_f32_vg1x4(4);
    svfloat32x4_t v2 = svread_za32_f32_vg1x4(8);
    svfloat32x4_t v3 = svread_za32_f32_vg1x4(12);
    svuint32x4_t u0 = svcvt_u32_f32_x4(v0);
    svuint32x4_t u1 = svcvt_u32_f32_x4(v1);
    svuint32x4_t u2 = svcvt_u32_f32_x4(v2); 
    svuint32x4_t u3 = svcvt_u32_f32_x4(v3); 
    u0 = svmin_single_u32_x4(u0, maxv);
    svuint16_t h00 = svuzp1_u16(
        svreinterpret_u16_u32(svget4_u32(u0, 0)),
        svreinterpret_u16_u32(svget4_u32(u0, 1))
    );
    svuint16_t h01 = svuzp1_u16(
        svreinterpret_u16_u32(svget4_u32(u0, 2)),
        svreinterpret_u16_u32(svget4_u32(u0, 3))
    );
    svuint8_t t0 = svuzp1_u8(
        svreinterpret_u8_u16(h00), svreinterpret_u8_u16(h01)
    );

    u1 = svmin_single_u32_x4(u1, maxv);
    svuint16_t h10 = svuzp1_u16(
        svreinterpret_u16_u32(svget4_u32(u1, 0)),
        svreinterpret_u16_u32(svget4_u32(u1, 1))
    );
    svuint16_t h11 = svuzp1_u16(
        svreinterpret_u16_u32(svget4_u32(u1, 2)),
        svreinterpret_u16_u32(svget4_u32(u1, 3))
    );
    svuint8_t t1 = svuzp1_u8(
        svreinterpret_u8_u16(h10), svreinterpret_u8_u16(h11)
    );

    u2 = svmin_single_u32_x4(u2, maxv);
    svuint16_t h20 = svuzp1_u16(
        svreinterpret_u16_u32(svget4_u32(u2, 0)),
        svreinterpret_u16_u32(svget4_u32(u2, 1))
    );
    svuint16_t h21 = svuzp1_u16(
        svreinterpret_u16_u32(svget4_u32(u2, 2)),
        svreinterpret_u16_u32(svget4_u32(u2, 3))
    );
    svuint8_t t2 = svuzp1_u8(
        svreinterpret_u8_u16(h20), svreinterpret_u8_u16(h21)
    );

    u3 = svmin_single_u32_x4(u3, maxv);
    svuint16_t h30 = svuzp1_u16(
        svreinterpret_u16_u32(svget4_u32(u3, 0)),
        svreinterpret_u16_u32(svget4_u32(u3, 1))
    );
    svuint16_t h31 = svuzp1_u16(
        svreinterpret_u16_u32(svget4_u32(u3, 2)),
        svreinterpret_u16_u32(svget4_u32(u3, 3))
    );
    svuint8_t t3 = svuzp1_u8(
        svreinterpret_u8_u16(h30), svreinterpret_u8_u16(h31)
    );

    std::size_t svl_b = svcntb();
    std::size_t n_vgx4_b = 4 * svl_b;
    std::size_t n_4xvgx4_b = 4 * n_vgx4_b;
    std::size_t n_full_4xvgx4 = width & ~(n_4xvgx4_b - 1);
    svbool_t pb = svptrue_b8();
    svcount_t pg = svptrue_c8();
    for (std::size_t y = 0; y < height; ++y) {
        std::uint8_t *row = data + y * row_stride;
        std::size_t x = 0;
        for (; x < n_full_4xvgx4;) {
            svuint8x4_t s0 = svld1_u8_x4(pg, row + x + 0 * n_vgx4_b);
            svuint8x4_t s1 = svld1_u8_x4(pg, row + x + 1 * n_vgx4_b);
            svuint8x4_t s2 = svld1_u8_x4(pg, row + x + 2 * n_vgx4_b);
            svuint8x4_t s3 = svld1_u8_x4(pg, row + x + 3 * n_vgx4_b);
            {
                svuint8_t p0 = svget4_u8(s0, 0);
                svuint8_t r00 = svtbl_u8(t0, p0);
                svuint8_t r01 = svtbl_u8(t1, svsub_n_u8_x(pb, p0, 64));
                svuint8_t r02 = svtbl_u8(t2, svsub_n_u8_x(pb, p0, 128));
                svuint8_t r03 = svtbl_u8(t3, svsub_n_u8_x(pb, p0, 192));
                svuint8_t v0 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r00, r01),
                    svorr_u8_x(pb, r02, r03)
                );
                svuint8_t p1 = svget4_u8(s0, 1);
                svuint8_t r10 = svtbl_u8(t0, p1);
                svuint8_t r11 = svtbl_u8(t1, svsub_n_u8_x(pb, p1, 64));
                svuint8_t r12 = svtbl_u8(t2, svsub_n_u8_x(pb, p1, 128));
                svuint8_t r13 = svtbl_u8(t3, svsub_n_u8_x(pb, p1, 192));
                svuint8_t v1 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r10, r11),
                    svorr_u8_x(pb, r12, r13)
                );
                svuint8_t p2 = svget4_u8(s0, 2);
                svuint8_t r20 = svtbl_u8(t0, p2);
                svuint8_t r21 = svtbl_u8(t1, svsub_n_u8_x(pb, p2, 64));
                svuint8_t r22 = svtbl_u8(t2, svsub_n_u8_x(pb, p2, 128));
                svuint8_t r23 = svtbl_u8(t3, svsub_n_u8_x(pb, p2, 192));
                svuint8_t v2 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r20, r21),
                    svorr_u8_x(pb, r22, r23)
                );
                svuint8_t p3 = svget4_u8(s0, 3);
                svuint8_t r30 = svtbl_u8(t0, p3);
                svuint8_t r31 = svtbl_u8(t1, svsub_n_u8_x(pb, p3, 64));
                svuint8_t r32 = svtbl_u8(t2, svsub_n_u8_x(pb, p3, 128));
                svuint8_t r33 = svtbl_u8(t3, svsub_n_u8_x(pb, p3, 192));
                svuint8_t v3 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r30, r31),
                    svorr_u8_x(pb, r32, r33)
                );
                svst1_u8_x4(pg, row + x, svcreate4_u8(v0, v1, v2, v3));
                x += n_vgx4_b;
            }
            {
                svuint8_t p0 = svget4_u8(s1, 0);
                svuint8_t r00 = svtbl_u8(t0, p0);
                svuint8_t r01 = svtbl_u8(t1, svsub_n_u8_x(pb, p0, 64));
                svuint8_t r02 = svtbl_u8(t2, svsub_n_u8_x(pb, p0, 128));
                svuint8_t r03 = svtbl_u8(t3, svsub_n_u8_x(pb, p0, 192));
                svuint8_t v0 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r00, r01),
                    svorr_u8_x(pb, r02, r03)
                );
                svuint8_t p1 = svget4_u8(s1, 1);
                svuint8_t r10 = svtbl_u8(t0, p1);
                svuint8_t r11 = svtbl_u8(t1, svsub_n_u8_x(pb, p1, 64));
                svuint8_t r12 = svtbl_u8(t2, svsub_n_u8_x(pb, p1, 128));
                svuint8_t r13 = svtbl_u8(t3, svsub_n_u8_x(pb, p1, 192));
                svuint8_t v1 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r10, r11),
                    svorr_u8_x(pb, r12, r13)
                );
                svuint8_t p2 = svget4_u8(s1, 2);
                svuint8_t r20 = svtbl_u8(t0, p2);
                svuint8_t r21 = svtbl_u8(t1, svsub_n_u8_x(pb, p2, 64));
                svuint8_t r22 = svtbl_u8(t2, svsub_n_u8_x(pb, p2, 128));
                svuint8_t r23 = svtbl_u8(t3, svsub_n_u8_x(pb, p2, 192));
                svuint8_t v2 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r20, r21),
                    svorr_u8_x(pb, r22, r23)
                );
                svuint8_t p3 = svget4_u8(s1, 3);
                svuint8_t r30 = svtbl_u8(t0, p3);
                svuint8_t r31 = svtbl_u8(t1, svsub_n_u8_x(pb, p3, 64));
                svuint8_t r32 = svtbl_u8(t2, svsub_n_u8_x(pb, p3, 128));
                svuint8_t r33 = svtbl_u8(t3, svsub_n_u8_x(pb, p3, 192));
                svuint8_t v3 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r30, r31),
                    svorr_u8_x(pb, r32, r33)
                );
                svst1_u8_x4(pg, row + x, svcreate4_u8(v0, v1, v2, v3));
                x += n_vgx4_b;
            }
            {
                svuint8_t p0 = svget4_u8(s2, 0);
                svuint8_t r00 = svtbl_u8(t0, p0);
                svuint8_t r01 = svtbl_u8(t1, svsub_n_u8_x(pb, p0, 64));
                svuint8_t r02 = svtbl_u8(t2, svsub_n_u8_x(pb, p0, 128));
                svuint8_t r03 = svtbl_u8(t3, svsub_n_u8_x(pb, p0, 192));
                svuint8_t v0 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r00, r01),
                    svorr_u8_x(pb, r02, r03)
                );
                svuint8_t p1 = svget4_u8(s2, 1);
                svuint8_t r10 = svtbl_u8(t0, p1);
                svuint8_t r11 = svtbl_u8(t1, svsub_n_u8_x(pb, p1, 64));
                svuint8_t r12 = svtbl_u8(t2, svsub_n_u8_x(pb, p1, 128));
                svuint8_t r13 = svtbl_u8(t3, svsub_n_u8_x(pb, p1, 192));
                svuint8_t v1 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r10, r11),
                    svorr_u8_x(pb, r12, r13)
                );
                svuint8_t p2 = svget4_u8(s2, 2);
                svuint8_t r20 = svtbl_u8(t0, p2);
                svuint8_t r21 = svtbl_u8(t1, svsub_n_u8_x(pb, p2, 64));
                svuint8_t r22 = svtbl_u8(t2, svsub_n_u8_x(pb, p2, 128));
                svuint8_t r23 = svtbl_u8(t3, svsub_n_u8_x(pb, p2, 192));
                svuint8_t v2 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r20, r21),
                    svorr_u8_x(pb, r22, r23)
                );
                svuint8_t p3 = svget4_u8(s2, 3);
                svuint8_t r30 = svtbl_u8(t0, p3);
                svuint8_t r31 = svtbl_u8(t1, svsub_n_u8_x(pb, p3, 64));
                svuint8_t r32 = svtbl_u8(t2, svsub_n_u8_x(pb, p3, 128));
                svuint8_t r33 = svtbl_u8(t3, svsub_n_u8_x(pb, p3, 192));
                svuint8_t v3 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r30, r31),
                    svorr_u8_x(pb, r32, r33)
                );
                svst1_u8_x4(pg, row + x, svcreate4_u8(v0, v1, v2, v3));
                x += n_vgx4_b;
            }
            {
                svuint8_t p0 = svget4_u8(s3, 0);
                svuint8_t r00 = svtbl_u8(t0, p0);
                svuint8_t r01 = svtbl_u8(t1, svsub_n_u8_x(pb, p0, 64));
                svuint8_t r02 = svtbl_u8(t2, svsub_n_u8_x(pb, p0, 128));
                svuint8_t r03 = svtbl_u8(t3, svsub_n_u8_x(pb, p0, 192));
                svuint8_t v0 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r00, r01),
                    svorr_u8_x(pb, r02, r03)
                );
                svuint8_t p1 = svget4_u8(s3, 1);
                svuint8_t r10 = svtbl_u8(t0, p1);
                svuint8_t r11 = svtbl_u8(t1, svsub_n_u8_x(pb, p1, 64));
                svuint8_t r12 = svtbl_u8(t2, svsub_n_u8_x(pb, p1, 128));
                svuint8_t r13 = svtbl_u8(t3, svsub_n_u8_x(pb, p1, 192));
                svuint8_t v1 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r10, r11),
                    svorr_u8_x(pb, r12, r13)
                );
                svuint8_t p2 = svget4_u8(s3, 2);
                svuint8_t r20 = svtbl_u8(t0, p2);
                svuint8_t r21 = svtbl_u8(t1, svsub_n_u8_x(pb, p2, 64));
                svuint8_t r22 = svtbl_u8(t2, svsub_n_u8_x(pb, p2, 128));
                svuint8_t r23 = svtbl_u8(t3, svsub_n_u8_x(pb, p2, 192));
                svuint8_t v2 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r20, r21),
                    svorr_u8_x(pb, r22, r23)
                );
                svuint8_t p3 = svget4_u8(s3, 3);
                svuint8_t r30 = svtbl_u8(t0, p3);
                svuint8_t r31 = svtbl_u8(t1, svsub_n_u8_x(pb, p3, 64));
                svuint8_t r32 = svtbl_u8(t2, svsub_n_u8_x(pb, p3, 128));
                svuint8_t r33 = svtbl_u8(t3, svsub_n_u8_x(pb, p3, 192));
                svuint8_t v3 = svorr_u8_x(
                    pb,
                    svorr_u8_x(pb, r30, r31),
                    svorr_u8_x(pb, r32, r33)
                );
                svst1_u8_x4(pg, row + x, svcreate4_u8(v0, v1, v2, v3));
                x += n_vgx4_b;
            }
        }
        for (; x < width; x += n_vgx4_b) {
            svcount_t tail = svwhilelt_c8_u64(x, width, 4);
            svuint8x4_t pixels = svld1_u8_x4(tail, row + x);
            svuint8_t p0 = svget4_u8(pixels, 0);
            svuint8_t r00 = svtbl_u8(t0, p0);
            svuint8_t r01 = svtbl_u8(t1, svsub_n_u8_x(pb, p0, 64));
            svuint8_t r02 = svtbl_u8(t2, svsub_n_u8_x(pb, p0, 128));
            svuint8_t r03 = svtbl_u8(t3, svsub_n_u8_x(pb, p0, 192));
            svuint8_t v0 = svorr_u8_x(
                pb,
                svorr_u8_x(pb, r00, r01),
                svorr_u8_x(pb, r02, r03)
            );
            svuint8_t p1 = svget4_u8(pixels, 1);
            svuint8_t r10 = svtbl_u8(t0, p1);
            svuint8_t r11 = svtbl_u8(t1, svsub_n_u8_x(pb, p1, 64));
            svuint8_t r12 = svtbl_u8(t2, svsub_n_u8_x(pb, p1, 128));
            svuint8_t r13 = svtbl_u8(t3, svsub_n_u8_x(pb, p1, 192));
            svuint8_t v1 = svorr_u8_x(
                pb,
                svorr_u8_x(pb, r10, r11),
                svorr_u8_x(pb, r12, r13)
            );
            svuint8_t p2 = svget4_u8(pixels, 2);
            svuint8_t r20 = svtbl_u8(t0, p2);
            svuint8_t r21 = svtbl_u8(t1, svsub_n_u8_x(pb, p2, 64));
            svuint8_t r22 = svtbl_u8(t2, svsub_n_u8_x(pb, p2, 128));
            svuint8_t r23 = svtbl_u8(t3, svsub_n_u8_x(pb, p2, 192));
            svuint8_t v2 = svorr_u8_x(
                pb,
                svorr_u8_x(pb, r20, r21),
                svorr_u8_x(pb, r22, r23)
            );
            svuint8_t p3 = svget4_u8(pixels, 3);
            svuint8_t r30 = svtbl_u8(t0, p3);
            svuint8_t r31 = svtbl_u8(t1, svsub_n_u8_x(pb, p3, 64));
            svuint8_t r32 = svtbl_u8(t2, svsub_n_u8_x(pb, p3, 128));
            svuint8_t r33 = svtbl_u8(t3, svsub_n_u8_x(pb, p3, 192));
            svuint8_t v3 = svorr_u8_x(
                pb,
                svorr_u8_x(pb, r30, r31),
                svorr_u8_x(pb, r32, r33)
            );
            svst1_u8_x4(tail, row + x, svcreate4_u8(v0, v1, v2, v3));
        }
    }
}

inline void scale_buffer(
    std::uint8_t *data,
    std::size_t n,
    float alpha
) __arm_streaming __arm_inout("za") { 
    std::size_t svl = svcntw(); 
    std::size_t n_vgx4 = 4 * svl;
    std::size_t n_4xvgx4 = 4 * n_vgx4; 
    std::size_t n_full_4xvgx4 = n & ~(n_4xvgx4 - 1); 

    std::size_t i = 0;
    svbool_t pb = svptrue_b32(); 
    svcount_t pg = svptrue_c32();

    svfloat32_t av = svdup_f32(alpha); 
    svuint32_t max = svdup_u32(255); 
    for (; i < n_full_4xvgx4;) { 
        svzero_za(); 

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

    svzero_za(); 
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
void scale_channel(FrameViewMut frame, float alpha) { 
    std::size_t width = frame.get_width();
    std::size_t height = frame.get_height();
    std::size_t stride = frame.get_row_stride(); 

    std::uint8_t *data = reinterpret_cast<std::uint8_t *>(frame.get_span().data()); 

    // fast paht 
    if (width == stride) 
        scale_buffer(data, width * height, alpha); 
    else
        scale_strided(data, width, height, stride, alpha);
} 

} // namespace

void scale(FrameViewMut frame, float r_alpha, float g_alpha, float b_alpha) { 
    if (!std::isfinite(r_alpha) || !std::isfinite(g_alpha) || !std::isfinite(b_alpha) ||
        r_alpha < 0 || g_alpha < 0 || b_alpha < 0) 
        throw std::invalid_argument("scales must be finite and nonnegative"); 
    if (frame.get_pixel_format() == PixelFormat::Gray)
        throw std::invalid_argument("expected RGB or RGBA frame for per-channel scaling"); 
    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");

    switch (frame.get_pixel_format()) { 
        case PixelFormat::RGB:
        case PixelFormat::RGBA: { 
            FrameViewMut r_frame = frame.channel_mut(0);
            FrameViewMut g_frame = frame.channel_mut(1);
            FrameViewMut b_frame = frame.channel_mut(2);
            scale_channel(r_frame, r_alpha); 
            scale_channel(g_frame, g_alpha);
            scale_channel(b_frame, b_alpha);
            break; 
        }

        case PixelFormat::Gray:
            throw std::logic_error("unreachable"); 
    }
}

void scale(FrameViewMut frame, float alpha) { 
    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout"); 
    if (!std::isfinite(alpha) || alpha < 0)
        throw std::invalid_argument("scale must be finite and nonnegative");

    switch (frame.get_pixel_format()) { 
        case PixelFormat::Gray:
            scale_channel(frame, alpha);
            break;

        case PixelFormat::RGB:
        case PixelFormat::RGBA: { 
            FrameViewMut r_frame = frame.channel_mut(0);
            FrameViewMut g_frame = frame.channel_mut(1);
            FrameViewMut b_frame = frame.channel_mut(2);
            scale_channel(r_frame, alpha); 
            scale_channel(g_frame, alpha);
            scale_channel(b_frame, alpha);
            break;
        }
    }
}
