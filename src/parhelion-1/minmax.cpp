#include "image_types.hpp" 
#include <arm_sve.h>
#include <utility>
#include <algorithm>
#include <array>

std::pair<uint8_t, uint8_t> global_min_max_buffer(
    const std::uint8_t *data,
    std::size_t n
) __arm_streaming { 

    svuint8_t max = svdup_n_u8(0);
    svuint8x4_t maxv = svcreate4_u8(
        max, max, max, max
    ); 

    svuint8_t min = svdup_n_u8(255);
    svuint8x4_t minv = svcreate4_u8(
        min, min, min, min
    ); 

    std::size_t svl_b = svcntb();
    std::size_t n_vgx4 = 4 * svl_b; 
    std::size_t n_full_vgx4 = n & ~(n_vgx4 - 1); 

    std::size_t i = 0;
    svbool_t pb = svptrue_b8();
    svcount_t pc = svptrue_c8(); 
    for (; i < n_full_vgx4; i += n_vgx4) { 
        svuint8x4_t v = svld1_u8_x4(pc, data + i);
        maxv = svmax_u8_x4(maxv, v); 
        minv = svmin_u8_x4(minv, v); 
    }

    if (i < n) {
        pc = svwhilelt_c8_u64(i, n, 4);
        svbool_t pb0 = svpext_lane_c8(pc, 0);
        svbool_t pb1 = svpext_lane_c8(pc, 1);
        svbool_t pb2 = svpext_lane_c8(pc, 2);
        svbool_t pb3 = svpext_lane_c8(pc, 3);
        svuint8x4_t v = svld1_u8_x4(pc, data + i);
        maxv = svmax_u8_x4(maxv, v);
        svuint8_t min0 = svmin_u8_m(pb0, svget4_u8(minv, 0), svget4_u8(v, 0));
        svuint8_t min1 = svmin_u8_m(pb1, svget4_u8(minv, 1), svget4_u8(v, 1));
        svuint8_t min2 = svmin_u8_m(pb2, svget4_u8(minv, 2), svget4_u8(v, 2));
        svuint8_t min3 = svmin_u8_m(pb3, svget4_u8(minv, 3), svget4_u8(v, 3));
        minv = svcreate4_u8(min0, min1, min2, min3);
    }

    min = svmin_u8_x(pb,
        svmin_u8_x(pb, svget4_u8(minv, 0), svget4_u8(minv, 1)),
        svmin_u8_x(pb, svget4_u8(minv, 2), svget4_u8(minv, 3))
    );
    max = svmax_u8_x(pb,
        svmax_u8_x(pb, svget4_u8(maxv, 0), svget4_u8(maxv, 1)),
        svmax_u8_x(pb, svget4_u8(maxv, 2), svget4_u8(maxv, 3))
    );
    uint8_t minf = svminv_u8(pb, min);
    uint8_t maxf = svmaxv_u8(pb, max);
    return { minf, maxf };
}

__arm_locally_streaming
std::pair<uint8_t, uint8_t> min_max_channel(FrameViewRef frame) {
    std::size_t width = frame.get_width();
    std::size_t height = frame.get_height();
    std::size_t stride = frame.get_row_stride();
    const std::uint8_t *data = reinterpret_cast<const std::uint8_t *>(frame.get_span().data());

    // fast path
    if (width == stride)
        return global_min_max_buffer(data, width * height);

    uint8_t minf = 255;
    uint8_t maxf = 0;
    for (std::size_t y = 0; y < height; ++y) {
        auto result = global_min_max_buffer(data + y * stride, width);
        minf = std::min(minf, result.first);
        maxf = std::max(maxf, result.second);
    }
    return { minf, maxf };
}

std::pair<std::array<uint8_t, 3>, std::array<uint8_t, 3>> minmax_per_channel(FrameViewRef frame) {
    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
    if (frame.get_pixel_format() == PixelFormat::Gray)
        throw std::invalid_argument("expected RGB or RGBA frame");
    if (frame.get_width() == 0 || frame.get_height() == 0)
        throw std::invalid_argument("expected nonempty frame");

    FrameViewRef r_frame = frame.channel_ref(0);
    FrameViewRef g_frame = frame.channel_ref(1);
    FrameViewRef b_frame = frame.channel_ref(2);
    auto r = min_max_channel(r_frame);
    auto g = min_max_channel(g_frame);
    auto b = min_max_channel(b_frame);
    return {{r.first, g.first, b.first}, {r.second, g.second, b.second}};
}

std::pair<uint8_t, uint8_t> minmax_global(FrameViewRef frame) {
    if (frame.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
    if (frame.get_width() == 0 || frame.get_height() == 0)
        throw std::invalid_argument("expected nonempty frame");

    if (frame.get_pixel_format() == PixelFormat::Gray)
        return min_max_channel(frame);

    auto result = minmax_per_channel(frame);
    uint8_t minf = std::min({result.first[0], result.first[1], result.first[2]});
    uint8_t maxf = std::max({result.second[0], result.second[1], result.second[2]});
    return { minf, maxf };
}
