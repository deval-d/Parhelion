#pragma once 
#include "AoS_SoA.hpp"
#include <arm_sve.h> 

namespace { 
    __arm_locally_streaming 
    void rgb_to_planar(Frame& frame) { 
        std::size_t n = frame.get_width() * frame.get_height(); 

        std::span<std::byte> src = frame.get_buf_span_mut(0, n * 3).value();
        std::uint8_t *src_ptr = reinterpret_cast<std::uint8_t *>(src.data()); 

        std::vector<std::byte> out(n * 3); 
        std::uint8_t *dst = reinterpret_cast<std::uint8_t *>(out.data()); 
        std::uint8_t *r_ptr = dst;
        std::uint8_t *g_ptr = dst + n;
        std::uint8_t *b_ptr = dst + 2 * n;

        std::size_t svlb = svcntb();
        std::size_t n_full = (n / svlb) * svlb;

        // full vgx3 blocks 
        std::size_t i = 0; 
        for (; i < n_full ; i += svlb) {
            svbool_t pg = svptrue_b8(); 

            svuint8x3_t rgb = svld3_u8(pg, src_ptr + i * 3);
            svuint8_t rv = svget3_u8(rgb, 0); 
            svuint8_t gv = svget3_u8(rgb, 1);
            svuint8_t bv = svget3_u8(rgb, 2); 

            svst1_u8(pg, r_ptr + i, rv);
            svst1_u8(pg, g_ptr + i, gv);
            svst1_u8(pg, b_ptr + i, bv); 
        }

        // leftover 
        for (; i < n; i += svlb) { 
            svbool_t pg = svwhilelt_b8_u64(i, n); 

            svuint8x3_t rgb = svld3_u8(pg, src_ptr + i * 3); 
            svuint8_t rv = svget3_u8(rgb, 0);
            svuint8_t gv = svget3_u8(rgb, 1);
            svuint8_t bv = svget3_u8(rgb, 2); 

            svst1_u8(pg, r_ptr + i, rv); 
            svst1_u8(pg, g_ptr + i, gv);
            svst1_u8(pg, b_ptr + i, bv); 
        }

        frame.replace_buf(std::move(out)); 
    }

    __arm_locally_streaming 
    void rgb_to_interleaved(Frame& frame) { 
        std::size_t n = frame.get_width() * frame.get_height(); 

        std::span<std::byte> src = frame.get_buf_span_mut(0, n * 3).value(); 
        std::uint8_t *src_ptr = reinterpret_cast<std::uint8_t *>(src.data());
        std::uint8_t *r_ptr = src_ptr;
        std::uint8_t *g_ptr = src_ptr + n;
        std::uint8_t *b_ptr = src_ptr + 2 * n; 

        std::vector<std::byte> out(n * 3); 
        std::uint8_t *dst = reinterpret_cast<std::uint8_t *>(out.data()); 

        std::size_t svlb = svcntb();
        std::size_t n_full = (n / svlb) * svlb; 

        std::size_t i = 0; 
        for (; i < n_full; i += svlb) { 
            svbool_t pg = svptrue_b8(); 

            svuint8_t rv = svld1_u8(pg, r_ptr + i); 
            svuint8_t gv = svld1_u8(pg, g_ptr + i);
            svuint8_t bv = svld1_u8(pg, b_ptr + i); 
            svuint8x3_t rgb = svcreate3_u8(rv, gv, bv); 

            svst3_u8(pg, dst + 3 * i, rgb); 
        }

        for (; i < n; i += svlb) { 
            svbool_t pg = svwhilelt_b8_u64(i, n); 

            svuint8_t rv = svld1_u8(pg, r_ptr + i); 
            svuint8_t gv = svld1_u8(pg, g_ptr + i);
            svuint8_t bv = svld1_u8(pg, b_ptr + i); 
            svuint8x3_t rgb = svcreate3_u8(rv, gv, bv); 

            svst3_u8(pg, dst + 3 * i, rgb); 
        }

        frame.replace_buf(std::move(out)); 
    }

    __arm_locally_streaming 
    void rgba_to_planar(Frame& frame) { 
        std::size_t n = frame.get_width() * frame.get_height(); 

        std::span<std::byte> src = frame.get_buf_span_mut(0, n * 4).value();
        std::uint8_t *src_ptr = reinterpret_cast<std::uint8_t *>(src.data()); 

        std::vector<std::byte> out(n * 4);
        std::uint8_t *dst = reinterpret_cast<std::uint8_t *>(out.data());
        std::uint8_t *r_ptr = dst;
        std::uint8_t *g_ptr = dst + n;
        std::uint8_t *b_ptr = dst + 2 * n;
        std::uint8_t *a_ptr = dst + 3 * n; 

        std::size_t svlb = svcntb();
        std::size_t n_full = (n / svlb) * svlb;

        // full vgx3 blocks 
        std::size_t i = 0; 
        for (; i < n_full ; i += svlb) {
            svbool_t pg = svptrue_b8(); 

            svuint8x4_t rgba = svld4_u8(pg, src_ptr + i * 4);
            svuint8_t rv = svget4_u8(rgba, 0); 
            svuint8_t gv = svget4_u8(rgba, 1);
            svuint8_t bv = svget4_u8(rgba, 2);
            svuint8_t av = svget4_u8(rgba, 3);

            svst1_u8(pg, r_ptr + i, rv);
            svst1_u8(pg, g_ptr + i, gv);
            svst1_u8(pg, b_ptr + i, bv);
            svst1_u8(pg, a_ptr + i, av); 
        }

        // leftover 
        for (; i < n; i += svlb) { 
            svbool_t pg = svwhilelt_b8_u64(i, n); 

            svuint8x4_t rgba = svld4_u8(pg, src_ptr + i * 4); 
            svuint8_t rv = svget4_u8(rgba, 0);
            svuint8_t gv = svget4_u8(rgba, 1);
            svuint8_t bv = svget4_u8(rgba, 2); 
            svuint8_t av = svget4_u8(rgba, 3);

            svst1_u8(pg, r_ptr + i, rv); 
            svst1_u8(pg, g_ptr + i, gv);
            svst1_u8(pg, b_ptr + i, bv); 
            svst1_u8(pg, a_ptr + i, av); 
        }
        
        frame.replace_buf(std::move(out)); 
    }

    __arm_locally_streaming 
    void rgba_to_interleaved(Frame& frame) { 
        std::size_t n = frame.get_width() * frame.get_height(); 

        std::span<std::byte> src = frame.get_buf_span_mut(0, n * 4).value(); 
        std::uint8_t *src_ptr = reinterpret_cast<std::uint8_t *>(src.data());
        std::uint8_t *r_ptr = src_ptr;
        std::uint8_t *g_ptr = src_ptr + n;
        std::uint8_t *b_ptr = src_ptr + 2 * n; 
        std::uint8_t *a_ptr = src_ptr + 3 * n; 

        std::vector<std::byte> out(n * 4); 
        std::uint8_t *dst = reinterpret_cast<std::uint8_t *>(out.data()); 

        std::size_t svlb = svcntb();
        std::size_t n_full = (n / svlb) * svlb; 

        std::size_t i = 0; 
        for (; i < n_full; i += svlb) { 
            svbool_t pg = svptrue_b8(); 

            svuint8_t rv = svld1_u8(pg, r_ptr + i); 
            svuint8_t gv = svld1_u8(pg, g_ptr + i);
            svuint8_t bv = svld1_u8(pg, b_ptr + i); 
            svuint8_t av = svld1_u8(pg, a_ptr + i); 
            svuint8x4_t rgba = svcreate4_u8(rv, gv, bv, av); 

            svst4_u8(pg, dst + 4 * i, rgba); 
        }

        for (; i < n; i += svlb) { 
            svbool_t pg = svwhilelt_b8_u64(i, n); 

            svuint8_t rv = svld1_u8(pg, r_ptr + i); 
            svuint8_t gv = svld1_u8(pg, g_ptr + i);
            svuint8_t bv = svld1_u8(pg, b_ptr + i); 
            svuint8_t av = svld1_u8(pg, a_ptr + i);
            svuint8x4_t rgba = svcreate4_u8(rv, gv, bv, av); 

            svst4_u8(pg, dst + 4 * i, rgba); 
        }

        frame.replace_buf(std::move(out)); 
    }
}


inline void convert_to_planar(Frame& frame) { 
    if (frame.get_frame_layout() == Planar) 
        return;

    switch (frame.get_pixel_format()) { 
        case (Gray): {
            break;
        }
        case (RGB): {
            rgb_to_planar(frame);
            break;
        }
        case (RGBA): {
            rgba_to_planar(frame);
            break; 
        }
    }

    frame.set_frame_layout(Planar);  
}

inline void convert_to_interleaved(Frame& frame) { 
    if (frame.get_frame_layout() == Interleaved) 
        return; 

    switch (frame.get_pixel_format()) { 
        case (Gray): {
            break;
        }
        case (RGB): {
            rgb_to_interleaved(frame); 
            break;
        }
        case (RGBA): {
            rgba_to_interleaved(frame);
            break;
        }
    }

    frame.set_frame_layout(Interleaved); 
}
