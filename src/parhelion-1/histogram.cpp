#include "parhelion-1.hpp"
#include <arm_sve.h>
#include <array>

namespace {

inline svuint64x4_t add4(
    svbool_t pu,
    svuint64x4_t a,
    svuint64x4_t b
) __arm_streaming {
    return svcreate4_u64(
        svadd_u64_x(pu, svget4_u64(a, 0), svget4_u64(b, 0)),
        svadd_u64_x(pu, svget4_u64(a, 1), svget4_u64(b, 1)),
        svadd_u64_x(pu, svget4_u64(a, 2), svget4_u64(b, 2)),
        svadd_u64_x(pu, svget4_u64(a, 3), svget4_u64(b, 3))
    );
}

__arm_locally_streaming
std::array<std::uint64_t, 256> histogram_channel(
    const std::uint8_t *data,
    std::size_t width,
    std::size_t height,
    std::size_t stride
) {
    std::array<std::uint64_t, 256> h0{};
    std::array<std::uint64_t, 256> h1{};  
    std::array<std::uint64_t, 256> h2{};  
    std::array<std::uint64_t, 256> h3{}; 
    for (std::size_t y = 0; y < height; ++y) {
        const std::uint8_t *row = data + y * stride;
        std::size_t i = 0;
        for (; i + 3 < width; i += 4) {
            ++h0[row[i + 0]];
            ++h1[row[i + 1]];
            ++h2[row[i + 2]];
            ++h3[row[i + 3]];
        }

        for (; i < width; ++i)
            ++h0[row[i]];
    }

    // reduction 
    std::size_t svl_u64 = svcntd(); 
    std::size_t n_vgx4 = 4 * svl_u64;
    std::size_t n_4xvgx4 = 4 * n_vgx4;
    std::size_t n_full_4xvgx4 = 256 & ~(n_4xvgx4 - 1);
    svbool_t pu = svptrue_b64(); 
    svcount_t pg = svptrue_c64();
    std::size_t i = 0;

    std::uint64_t *h0_ptr = h0.data(); 
    std::uint64_t *h1_ptr = h1.data();
    std::uint64_t *h2_ptr = h2.data();
    std::uint64_t *h3_ptr = h3.data(); 

    svuint64x4_t h0v0, h1v0;
    svuint64x4_t h0v1, h1v1;
    svuint64x4_t h0v2, h1v2;
    svuint64x4_t h0v3, h1v3; 
    for (; i < n_full_4xvgx4; i += n_4xvgx4) { 
         h0v0 = svld1_u64_x4(pg, h0_ptr + i); 
         h0v1 = svld1_u64_x4(pg, h0_ptr + i + n_vgx4); 
         h0v2 = svld1_u64_x4(pg, h0_ptr + i + 2 * n_vgx4); 
         h0v3 = svld1_u64_x4(pg, h0_ptr + i + 3 * n_vgx4);

         h1v0 = svld1_u64_x4(pg, h1_ptr + i); 
         h1v1 = svld1_u64_x4(pg, h1_ptr + i + n_vgx4); 
         h1v2 = svld1_u64_x4(pg, h1_ptr + i + 2 * n_vgx4); 
         h1v3 = svld1_u64_x4(pg, h1_ptr + i + 3 * n_vgx4);
         h0v0 = add4(pu, h0v0, h1v0); 
         h0v1 = add4(pu, h0v1, h1v1);
         h0v2 = add4(pu, h0v2, h1v2);
         h0v3 = add4(pu, h0v3, h1v3); 

         h1v0 = svld1_u64_x4(pg, h2_ptr + i);
         h1v1 = svld1_u64_x4(pg, h2_ptr + i + n_vgx4);
         h1v2 = svld1_u64_x4(pg, h2_ptr + i + 2 * n_vgx4);
         h1v3 = svld1_u64_x4(pg, h2_ptr + i + 3 * n_vgx4);
         h0v0 = add4(pu, h0v0, h1v0); 
         h0v1 = add4(pu, h0v1, h1v1);
         h0v2 = add4(pu, h0v2, h1v2);
         h0v3 = add4(pu, h0v3, h1v3); 

         h1v0 = svld1_u64_x4(pg, h3_ptr + i);
         h1v1 = svld1_u64_x4(pg, h3_ptr + i + n_vgx4);
         h1v2 = svld1_u64_x4(pg, h3_ptr + i + 2 * n_vgx4);
         h1v3 = svld1_u64_x4(pg, h3_ptr + i + 3 * n_vgx4);
         h0v0 = add4(pu, h0v0, h1v0); 
         h0v1 = add4(pu, h0v1, h1v1);
         h0v2 = add4(pu, h0v2, h1v2);
         h0v3 = add4(pu, h0v3, h1v3); 

         svst1_u64_x4(pg, h0_ptr + i, h0v0);
         svst1_u64_x4(pg, h0_ptr + i + n_vgx4, h0v1);
         svst1_u64_x4(pg, h0_ptr + i + 2 * n_vgx4, h0v2);
         svst1_u64_x4(pg, h0_ptr + i + 3 * n_vgx4, h0v3); 
    }

    for (; i < 256; i += n_vgx4) {
        pg = svwhilelt_c64_u64(i, 256, 4);
        h0v0 = svld1_u64_x4(pg, h0_ptr + i);
        h1v0 = svld1_u64_x4(pg, h1_ptr + i);
        h0v0 = add4(pu, h0v0, h1v0);
        h1v0 = svld1_u64_x4(pg, h2_ptr + i);
        h0v0 = add4(pu, h0v0, h1v0);
        h1v0 = svld1_u64_x4(pg, h3_ptr + i);
        h0v0 = add4(pu, h0v0, h1v0);
        svst1_u64_x4(pg, h0_ptr + i, h0v0);
    }

    return h0; 
}

std::array<std::uint64_t, 256> histogram_view(FrameViewRef frame) {
    std::size_t width = frame.get_width();
    std::size_t height = frame.get_height();
    std::size_t stride = frame.get_row_stride();
    const std::uint8_t *data = reinterpret_cast<const std::uint8_t *>(frame.get_span().data());

    // fast path
    if (width == stride)
        return histogram_channel(data, width * height, 1, width * height);

    return histogram_channel(data, width, height, stride);
}

} // namespace

std::array<std::uint64_t, 256> histogram_gray(FrameViewRef frame) {
    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
    if (frame.get_pixel_format() != PixelFormat::Gray)
        throw std::invalid_argument("expected Gray frame");
    if (frame.get_width() == 0 || frame.get_height() == 0)
        throw std::invalid_argument("expected nonempty frame");

    return histogram_view(frame);
}

std::array<std::array<std::uint64_t, 256>, 3> histogram_rgb(FrameViewRef frame) {
    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
    if (frame.get_pixel_format() != PixelFormat::RGB)
        throw std::invalid_argument("expected RGB frame");
    if (frame.get_width() == 0 || frame.get_height() == 0)
        throw std::invalid_argument("expected nonempty frame");

    FrameViewRef r_frame = frame.channel_ref(0);
    FrameViewRef g_frame = frame.channel_ref(1);
    FrameViewRef b_frame = frame.channel_ref(2);
    return {histogram_view(r_frame), histogram_view(g_frame), histogram_view(b_frame)};
}

std::array<std::array<std::uint64_t, 256>, 4> histogram_rgba(FrameViewRef frame) {
    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
    if (frame.get_pixel_format() != PixelFormat::RGBA)
        throw std::invalid_argument("expected RGBA frame");
    if (frame.get_width() == 0 || frame.get_height() == 0)
        throw std::invalid_argument("expected nonempty frame");

    FrameViewRef r_frame = frame.channel_ref(0);
    FrameViewRef g_frame = frame.channel_ref(1);
    FrameViewRef b_frame = frame.channel_ref(2);
    FrameViewRef a_frame = frame.channel_ref(3);
    return {histogram_view(r_frame), histogram_view(g_frame), histogram_view(b_frame), histogram_view(a_frame)};
}
