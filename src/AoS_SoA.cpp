#include "AoS_SoA.hpp"
#include <arm_sve.h> 
#include <stdexcept>

namespace { 
    __arm_locally_streaming 
    void rgb_to_planar(FrameViewRef view_ref, FrameViewMut view_mut) { 
        std::size_t n = view_ref.get_width() * view_ref.get_height(); 

        std::span<const std::byte> src = view_ref.get_span(); 
        const std::uint8_t *src_ptr = reinterpret_cast<const std::uint8_t *>(src.data()); 

        std::span<std::byte> out = view_mut.get_span(); 
        std::uint8_t *dst = reinterpret_cast<std::uint8_t *>(out.data()); 
        std::uint8_t *r_ptr = dst;
        std::uint8_t *g_ptr = dst + n;
        std::uint8_t *b_ptr = dst + 2 * n;

        std::size_t svlb = svcntb();
        std::size_t n_full = (n / svlb) * svlb;

    // full vgx3 blocks 
    svbool_t p = svptrue_b8(); 
    svboolx4_t pg = svcreate4_b(p, p, p, p);
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
    }

    __arm_locally_streaming 
    void rgb_to_interleaved(FrameViewRef view_ref, FrameViewMut view_mut) { 
        std::size_t n = view_ref.get_width() * view_ref.get_height(); 

        std::span<const std::byte> src = view_ref.get_span(); 
        const std::uint8_t *src_ptr = reinterpret_cast<const std::uint8_t *>(src.data());
        const std::uint8_t *r_ptr = src_ptr;
        const std::uint8_t *g_ptr = src_ptr + n;
        const std::uint8_t *b_ptr = src_ptr + 2 * n; 

        std::span<std::byte> out = view_mut.get_span(); 
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
    }

    __arm_locally_streaming 
    void rgba_to_planar(FrameViewRef view_ref, FrameViewMut view_mut) { 
        std::size_t n = view_ref.get_width() * view_ref.get_height(); 

        std::span<const std::byte> src = view_ref.get_span(); 
        const std::uint8_t *src_ptr = reinterpret_cast<const std::uint8_t *>(src.data()); 

        std::span<std::byte> out = view_mut.get_span(); 
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
    }

    __arm_locally_streaming 
    void rgba_to_interleaved(FrameViewRef view_ref, FrameViewMut view_mut) { 
        std::size_t n = view_ref.get_width() * view_ref.get_height(); 

        std::span<const std::byte> src = view_ref.get_span(); 
        const std::uint8_t *src_ptr = reinterpret_cast<const std::uint8_t *>(src.data());
        const std::uint8_t *r_ptr = src_ptr;
        const std::uint8_t *g_ptr = src_ptr + n;
        const std::uint8_t *b_ptr = src_ptr + 2 * n; 
        const std::uint8_t *a_ptr = src_ptr + 3 * n; 

        std::span<std::byte> out = view_mut.get_span(); 
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
    }
}

void convert_to_planar(Frame& frame) { 
    if (frame.get_frame_layout() == FrameLayout::Planar) 
        return;
    if (frame.get_pixel_format() == PixelFormat::Gray) {
        frame.set_frame_layout(FrameLayout::Planar);
        return; 
    }

    std::size_t width = frame.get_width();
    std::size_t height = frame.get_height();
    std::size_t n_channels = frame.get_n_channels(); 

    FrameViewRef view_ref = frame.view_ref();

    // temporary buf 
    std::vector<std::byte> out(width * height * n_channels);
    FrameViewMut view_mut(
        width, height,
        width, width * height, 
        frame.get_pixel_format(), FrameLayout::Planar,
        std::span<std::byte>(out.begin(), out.end())
    ); 

    switch (frame.get_pixel_format()) { 
        case PixelFormat::RGB: 
            rgb_to_planar(view_ref, view_mut);
            break;

        case PixelFormat::RGBA: 
            rgba_to_planar(view_ref, view_mut);
            break; 

        case PixelFormat::Gray:
            throw std::logic_error("should be unreachable"); 
    }

    frame.replace_buf(std::move(out)); 
    frame.set_frame_layout(FrameLayout::Planar);  
}

void convert_to_interleaved(Frame& frame) { 
    if (frame.get_frame_layout() == FrameLayout::Interleaved) 
        return; 
    if (frame.get_pixel_format() == PixelFormat::Gray) {
        frame.set_frame_layout(FrameLayout::Interleaved);
        return; 
    }

    std::size_t width = frame.get_width();
    std::size_t height = frame.get_height();
    std::size_t n_channels = frame.get_n_channels();

    FrameViewRef view_ref = frame.view_ref(); 

    // temporary buf 
    std::vector<std::byte> out(width * height * n_channels); 
    FrameViewMut view_mut(
        width, height,
        width * n_channels, 
        std::nullopt,
        frame.get_pixel_format(), 
        FrameLayout::Interleaved,
        std::span<std::byte>(out.begin(), out.end())
    ); 

    switch (frame.get_pixel_format()) { 
        case PixelFormat::RGB: 
            rgb_to_interleaved(view_ref, view_mut); 
            break;

        case PixelFormat::RGBA: 
            rgba_to_interleaved(view_ref, view_mut);
            break;

        case PixelFormat::Gray:
            throw std::logic_error("should be unreachable"); 
    }

    frame.replace_buf(std::move(out)); 
    frame.set_frame_layout(FrameLayout::Interleaved); 
}
