#include <metal_stdlib>
using namespace metal;

struct Layout {
    ulong row_stride;
    ulong channel_stride;
    uint pixel_stride;
};

struct Params {
    Layout dst;
    Layout src1;
    Layout src2;
    uint channels;
    float a[4];
    float b[4];
};

ulong offset(constant Layout& layout, uint2 xy, uint channel) {
    return ulong(xy.y) * layout.row_stride + ulong(xy.x) * layout.pixel_stride
        + ulong(channel) * layout.channel_stride;
}

uchar saturate_byte(float value) {
    return uchar(clamp(value, 0.0f, 255.0f));
}

kernel void fill_pixels(device uchar* dst [[buffer(0)]], constant Params& p [[buffer(1)]],
    uint2 xy [[thread_position_in_grid]]) {
    for (uint c = 0; c < p.channels; ++c)
        dst[offset(p.dst, xy, c)] = uchar(p.a[c]);
}

kernel void copy_pixels(device uchar* dst [[buffer(0)]], constant Params& p [[buffer(1)]],
    const device uchar* src [[buffer(2)]], uint2 xy [[thread_position_in_grid]]) {
    for (uint c = 0; c < p.channels; ++c)
        dst[offset(p.dst, xy, c)] = src[offset(p.src1, xy, c)];
}

kernel void brighten_pixels(device uchar* dst [[buffer(0)]], constant Params& p [[buffer(1)]],
    uint2 xy [[thread_position_in_grid]]) {
    for (uint c = 0; c < p.channels; ++c) {
        const ulong i = offset(p.dst, xy, c);
        dst[i] = uchar(clamp(int(dst[i]) + int(p.a[c]), 0, 255));
    }
}

kernel void scale_pixels(device uchar* dst [[buffer(0)]], constant Params& p [[buffer(1)]],
    uint2 xy [[thread_position_in_grid]]) {
    for (uint c = 0; c < p.channels; ++c) {
        const ulong i = offset(p.dst, xy, c);
        dst[i] = saturate_byte(float(dst[i]) * p.a[c]);
    }
}

kernel void affine_pixels(device uchar* dst [[buffer(0)]], constant Params& p [[buffer(1)]],
    uint2 xy [[thread_position_in_grid]]) {
    for (uint c = 0; c < p.channels; ++c) {
        const ulong i = offset(p.dst, xy, c);
        dst[i] = saturate_byte(fma(float(dst[i]), p.a[c], p.b[c]));
    }
}

kernel void threshold_pixels(device uchar* dst [[buffer(0)]], constant Params& p [[buffer(1)]],
    uint2 xy [[thread_position_in_grid]]) {
    for (uint c = 0; c < p.channels; ++c) {
        const ulong i = offset(p.dst, xy, c);
        dst[i] = dst[i] > p.a[c] ? 255 : 0;
    }
}

kernel void blend_pixels(device uchar* dst [[buffer(0)]], constant Params& p [[buffer(1)]],
    const device uchar* src1 [[buffer(2)]], const device uchar* src2 [[buffer(3)]],
    uint2 xy [[thread_position_in_grid]]) {
    for (uint c = 0; c < p.channels; ++c) {
        const float a = src1[offset(p.src1, xy, c)];
        const float b = src2[offset(p.src2, xy, c)];
        dst[offset(p.dst, xy, c)] = saturate_byte(fma(b, p.b[c], a * p.a[c]));
    }
}

kernel void grayscale_pixels(device uchar* dst [[buffer(0)]], constant Params& p [[buffer(1)]],
    const device uchar* src [[buffer(2)]], uint2 xy [[thread_position_in_grid]]) {
    const float r = src[offset(p.src1, xy, 0)];
    const float g = src[offset(p.src1, xy, 1)];
    const float b = src[offset(p.src1, xy, 2)];
    dst[offset(p.dst, xy, 0)] = saturate_byte(fma(b, 0.0722f, fma(g, 0.7152f, r * 0.2126f)));
}

kernel void premultiply_alpha_pixels(device uchar* dst [[buffer(0)]], constant Params& p [[buffer(1)]],
    uint2 xy [[thread_position_in_grid]]) {
    const uint alpha = dst[offset(p.dst, xy, 3)];
    for (uint c = 0; c < 3; ++c) {
        const ulong i = offset(p.dst, xy, c);
        dst[i] = uchar((uint(dst[i]) * alpha + 127) / 255);
    }
}
