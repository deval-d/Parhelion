#include "metal-parhelion-1.hpp"
#include "parhelion_metal_shader.hpp"
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>

namespace {

enum class Kernel { fill, copy, brighten, scale, affine, threshold, blend, grayscale, premultiply, kernel_count };
constexpr const char* kernel_names[] = {
    "fill_pixels", "copy_pixels", "brighten_pixels", "scale_pixels", "affine_pixels",
    "threshold_pixels", "blend_pixels", "grayscale_pixels", "premultiply_alpha_pixels"
};

using Values = std::array<float, 4>;

struct Layout {
    std::uint64_t row_stride;
    std::uint64_t channel_stride;
    std::uint32_t pixel_stride;
};

struct Params {
    Layout dst, src1, src2;
    std::uint32_t channels;
    Values a, b;
};

static_assert(sizeof(Layout) == 24 && offsetof(Params, a) == 76 && sizeof(Params) == 112);

template <typename View>
Layout layout(View view) {
    if (view.get_frame_layout() == FrameLayout::Planar)
        return {view.get_row_stride(), view.get_channel_stride().value_or(0), 1};
    return {view.get_row_stride(), 1, static_cast<std::uint32_t>(view.get_n_channels())};
}

template <typename View>
void require_planar(View view) {
    if (view.get_frame_layout() != FrameLayout::Planar)
        throw std::invalid_argument("expected Planar layout");
}

void require_color(FrameViewMut frame) {
    if (frame.get_pixel_format() == PixelFormat::Gray)
        throw std::invalid_argument("expected RGB or RGBA frame");
}

void require_match(FrameViewRef src, FrameViewMut dst) {
    if (src.get_pixel_format() != dst.get_pixel_format() ||
        src.get_frame_layout() != dst.get_frame_layout())
        throw std::invalid_argument("frame formats and layouts must match");
}

std::runtime_error metal_error(const char* message, NSError* error) {
    const char* detail = error.localizedDescription.UTF8String;
    return std::runtime_error(detail ? std::string(message) + ": " + detail : message);
}

Values uniform(float value) {
    return {value, value, value, value};
}

std::uint32_t colors(FrameViewMut frame) {
    return static_cast<std::uint32_t>(std::min<std::size_t>(frame.get_n_channels(), 3));
}

} // namespace

struct MetalContext::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipelines[static_cast<std::size_t>(Kernel::kernel_count)];
};

MetalContext::MetalContext() : impl(std::make_unique<Impl>()) {
    @autoreleasepool {
        impl->device = MTLCreateSystemDefaultDevice();
        impl->queue = [impl->device newCommandQueue];
        if (!impl->queue)
            throw std::runtime_error("metal command queue unavailable");

        auto* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        NSError* error = nil;
        auto library = [impl->device newLibraryWithSource:
            [NSString stringWithUTF8String:pointwise_source] options:options error:&error];
        if (!library)
            throw metal_error("metal shader compilation failed", error);

        for (std::size_t i = 0; i < std::size(kernel_names); ++i) {
            auto function = [library newFunctionWithName:[NSString stringWithUTF8String:kernel_names[i]]];
            if (!function)
                throw std::runtime_error("metal kernel missing");
            impl->pipelines[i] = [impl->device newComputePipelineStateWithFunction:function error:&error];
            if (!impl->pipelines[i])
                throw metal_error("metal pipeline creation failed", error);
        }
    }
}

MetalContext::~MetalContext() = default;

struct MetalFrame::Impl {
    id<MTLBuffer> buffer;
    std::size_t width, height, channels;
    PixelFormat format;
    FrameLayout layout;

    template <typename Byte>
    FrameView<Byte> view() const {
        const bool planar = layout == FrameLayout::Planar;
        return {width, height, planar ? width : width * channels,
            planar ? std::optional(width * height) : std::nullopt, format, layout,
            std::span<Byte>(static_cast<Byte*>(buffer.contents), buffer.length)};
    }
};

MetalFrame::MetalFrame(MetalContext& context, std::size_t width, std::size_t height,
    PixelFormat format, FrameLayout layout) {
    if (width == 0 || height == 0)
        throw std::invalid_argument("frame dimensions must be nonzero");
    std::size_t channels;
    switch (format) {
        case PixelFormat::Gray: channels = 1; break;
        case PixelFormat::RGB: channels = 3; break;
        case PixelFormat::RGBA: channels = 4; break;
        default: throw std::invalid_argument("invalid pixel format");
    }
    if (layout != FrameLayout::Planar && layout != FrameLayout::Interleaved)
        throw std::invalid_argument("invalid frame layout");

    auto device = context.impl->device;
    if (width > device.maxBufferLength / channels / height)
        throw std::length_error("frame exceeds metal buffer capacity");
    auto buffer = [device newBufferWithLength:width * height * channels
        options:MTLResourceStorageModeShared];
    if (!buffer)
        throw std::runtime_error("metal buffer allocation failed");
    impl = std::make_unique<Impl>(Impl{buffer, width, height, channels, format, layout});
}

MetalFrame::~MetalFrame() = default;

FrameViewRef MetalFrame::view_ref() const { return impl->view<const std::byte>(); }
FrameViewMut MetalFrame::view_mut() { return impl->view<std::byte>(); }

namespace {

template <typename Byte>
struct MetalBinding : FrameView<Byte> {
    MetalBinding(FrameView<Byte> view, id<MTLBuffer> buffer = nil)
        : FrameView<Byte>(view), buffer(buffer) {}

    id<MTLBuffer> buffer;
};

using MetalInput = MetalBinding<const std::byte>;
using MetalOutput = MetalBinding<std::byte>;

} // namespace

struct MetalPointwise {
    static MetalInput bind(const MetalFrame& frame) {
        return {frame.view_ref(), frame.impl->buffer};
    }

    static MetalOutput bind(MetalFrame& frame) {
        return {frame.view_mut(), frame.impl->buffer};
    }

    static void run(MetalContext& context, Kernel kernel, MetalOutput dst,
        std::uint32_t channels, Values a = {}, Values b = {},
        std::optional<MetalInput> src1 = {}, std::optional<MetalInput> src2 = {}) {
        const auto width = dst.get_width();
        const auto height = dst.get_height();
        for (const auto& src : {src1, src2}) {
            if (src && (src->get_width() != width || src->get_height() != height))
                throw std::invalid_argument("frame dimensions must match");
        }
        if (width > std::numeric_limits<std::uint32_t>::max() ||
            height > std::numeric_limits<std::uint32_t>::max())
            throw std::length_error("frame exceeds metal grid dimensions");

        @autoreleasepool {
            auto& metal = *context.impl;
            auto upload = [&](std::span<const std::byte> bytes) {
                auto buffer = [metal.device newBufferWithBytes:bytes.data() length:bytes.size()
                    options:MTLResourceStorageModeShared];
                if (!buffer)
                    throw std::runtime_error("metal buffer allocation failed");
                return buffer;
            };
            auto buffer_for = [&](const auto& view) -> id<MTLBuffer> {
                if (!view.buffer)
                    return upload(view.get_span());
                if (view.buffer.device != metal.device)
                    throw std::invalid_argument("frame belongs to another metal device");
                return view.buffer;
            };
            auto output = buffer_for(dst);
            id<MTLBuffer> input1 = src1 ? buffer_for(*src1) : nil;
            id<MTLBuffer> input2 = src2 ? buffer_for(*src2) : nil;
            const Params params{layout(dst), src1 ? layout(*src1) : Layout{},
                src2 ? layout(*src2) : Layout{}, channels, a, b};

            auto command = [metal.queue commandBuffer];
            auto encoder = [command computeCommandEncoder];
            if (!encoder)
                throw std::runtime_error("metal compute encoder unavailable");
            auto pipeline = metal.pipelines[static_cast<std::size_t>(kernel)];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:output offset:0 atIndex:0];
            [encoder setBytes:&params length:sizeof(params) atIndex:1];
            if (input1)
                [encoder setBuffer:input1 offset:0 atIndex:2];
            if (input2)
                [encoder setBuffer:input2 offset:0 atIndex:3];
            const auto lanes = pipeline.threadExecutionWidth;
            const auto rows = std::min<NSUInteger>(4, pipeline.maxTotalThreadsPerThreadgroup / lanes);
            [encoder dispatchThreads:MTLSizeMake(width, height, 1)
                threadsPerThreadgroup:MTLSizeMake(lanes, rows, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw metal_error("metal dispatch failed", command.error);
            if (!dst.buffer) {
                const auto bytes = dst.get_span();
                std::memcpy(bytes.data(), output.contents, bytes.size());
            }
        }
    }
};

namespace {

float byte_value(std::byte value) {
    return std::to_integer<unsigned>(value);
}

void transform(MetalContext& context, Kernel kernel, MetalOutput frame, Values a, Values b = {}) {
    require_planar(frame);
    const auto channels = colors(frame);
    for (std::uint32_t c = 0; c < channels; ++c) {
        if (kernel == Kernel::brighten && (a[c] < -255 || a[c] > 255))
            throw std::invalid_argument("brightness offset must be in [-255, 255]");
        if ((kernel == Kernel::scale || kernel == Kernel::affine) &&
            (!std::isfinite(a[c]) || a[c] < 0 || !std::isfinite(b[c])))
            throw std::invalid_argument("expected finite coefficients and nonnegative scales");
    }
    MetalPointwise::run(context, kernel, frame, channels, a, b);
}

void blend_views(MetalContext& context, MetalInput src1, MetalInput src2,
    MetalOutput dst, Values weights) {
    require_planar(dst);
    require_match(src1, dst);
    require_match(src2, dst);
    const auto channels = colors(dst);
    Values complement{};
    for (std::uint32_t c = 0; c < channels; ++c) {
        if (!std::isfinite(weights[c]) || weights[c] < 0 || weights[c] > 1)
            throw std::invalid_argument("blend weights must be finite and in [0, 1]");
        complement[c] = 1.0f - weights[c];
    }
    MetalPointwise::run(context, Kernel::blend, dst, channels, weights, complement, src1, src2);
}

void run_fill(MetalContext& context, MetalOutput frame, Color::Gray color) {
    if (frame.get_pixel_format() != PixelFormat::Gray)
        throw std::invalid_argument("expected a single channel");
    MetalPointwise::run(context, Kernel::fill, frame, 1, {byte_value(color.v)});
}

void run_fill(MetalContext& context, MetalOutput frame, Color::RGB color) {
    require_planar(frame);
    require_color(frame);
    MetalPointwise::run(context, Kernel::fill, frame, 3,
        {byte_value(color.r), byte_value(color.g), byte_value(color.b)});
}

void run_fill(MetalContext& context, MetalOutput frame, Color::RGBA color) {
    require_planar(frame);
    if (frame.get_pixel_format() != PixelFormat::RGBA)
        throw std::invalid_argument("expected RGBA frame");
    MetalPointwise::run(context, Kernel::fill, frame, 4,
        {byte_value(color.r), byte_value(color.g), byte_value(color.b), byte_value(color.a)});
}

void run_copy(MetalContext& context, MetalInput src, MetalOutput dst) {
    require_match(src, dst);
    MetalPointwise::run(context, Kernel::copy, dst, static_cast<std::uint32_t>(dst.get_n_channels()),
        {}, {}, src);
}

void run_brighten(MetalContext& context, MetalOutput frame, int amount) {
    transform(context, Kernel::brighten, frame, uniform(amount));
}

void run_brighten(MetalContext& context, MetalOutput frame, int r, int g, int b) {
    require_color(frame);
    transform(context, Kernel::brighten, frame, {float(r), float(g), float(b)});
}

void run_scale(MetalContext& context, MetalOutput frame, float alpha) {
    transform(context, Kernel::scale, frame, uniform(alpha));
}

void run_scale(MetalContext& context, MetalOutput frame, float r, float g, float b) {
    require_color(frame);
    transform(context, Kernel::scale, frame, {r, g, b});
}

void run_affine(MetalContext& context, MetalOutput frame, float alpha, float beta) {
    transform(context, Kernel::affine, frame, uniform(alpha), uniform(beta));
}

void run_affine(MetalContext& context, MetalOutput frame,
    float r_alpha, float g_alpha, float b_alpha, float r_beta, float g_beta, float b_beta) {
    require_color(frame);
    transform(context, Kernel::affine, frame, {r_alpha, g_alpha, b_alpha}, {r_beta, g_beta, b_beta});
}

void run_threshold(MetalContext& context, MetalOutput frame, std::uint8_t value) {
    transform(context, Kernel::threshold, frame, uniform(value));
}

void run_threshold(MetalContext& context, MetalOutput frame,
    std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    require_color(frame);
    transform(context, Kernel::threshold, frame, {float(r), float(g), float(b)});
}

void run_blend(MetalContext& context, MetalInput src1, MetalInput src2, MetalOutput dst,
    float alpha) {
    blend_views(context, src1, src2, dst, uniform(alpha));
}

void run_blend(MetalContext& context, MetalInput src1, MetalInput src2, MetalOutput dst,
    float r, float g, float b) {
    require_color(dst);
    blend_views(context, src1, src2, dst, {r, g, b});
}

void run_grayscale(MetalContext& context, MetalInput src, MetalOutput dst) {
    require_planar(src);
    if (src.get_pixel_format() == PixelFormat::Gray || dst.get_pixel_format() != PixelFormat::Gray)
        throw std::invalid_argument("expected RGB or RGBA source and Gray destination");
    MetalPointwise::run(context, Kernel::grayscale, dst, 1, {}, {}, src);
}

void run_premultiply_alpha(MetalContext& context, MetalOutput frame) {
    require_planar(frame);
    if (frame.get_pixel_format() != PixelFormat::RGBA)
        throw std::invalid_argument("expected RGBA frame");
    MetalPointwise::run(context, Kernel::premultiply, frame, 3);
}

} // namespace

void metal_fill(MetalContext& context, FrameViewMut frame, Color::Gray color) {
    run_fill(context, frame, color);
}

void metal_fill(MetalContext& context, MetalFrame& frame, Color::Gray color) {
    run_fill(context, MetalPointwise::bind(frame), color);
}

void metal_fill(MetalContext& context, FrameViewMut frame, Color::RGB color) {
    run_fill(context, frame, color);
}

void metal_fill(MetalContext& context, MetalFrame& frame, Color::RGB color) {
    run_fill(context, MetalPointwise::bind(frame), color);
}

void metal_fill(MetalContext& context, FrameViewMut frame, Color::RGBA color) {
    run_fill(context, frame, color);
}

void metal_fill(MetalContext& context, MetalFrame& frame, Color::RGBA color) {
    run_fill(context, MetalPointwise::bind(frame), color);
}

void metal_copy(MetalContext& context, FrameViewRef src, FrameViewMut dst) {
    run_copy(context, src, dst);
}

void metal_copy(MetalContext& context, const MetalFrame& src, MetalFrame& dst) {
    run_copy(context, MetalPointwise::bind(std::as_const(src)), MetalPointwise::bind(dst));
}

void metal_brighten(MetalContext& context, FrameViewMut frame, int amount) {
    run_brighten(context, frame, amount);
}

void metal_brighten(MetalContext& context, MetalFrame& frame, int amount) {
    run_brighten(context, MetalPointwise::bind(frame), amount);
}

void metal_brighten(MetalContext& context, FrameViewMut frame, int r_amount, int g_amount, int b_amount) {
    run_brighten(context, frame, r_amount, g_amount, b_amount);
}

void metal_brighten(MetalContext& context, MetalFrame& frame, int r_amount, int g_amount, int b_amount) {
    run_brighten(context, MetalPointwise::bind(frame), r_amount, g_amount, b_amount);
}

void metal_scale(MetalContext& context, FrameViewMut frame, float alpha) {
    run_scale(context, frame, alpha);
}

void metal_scale(MetalContext& context, MetalFrame& frame, float alpha) {
    run_scale(context, MetalPointwise::bind(frame), alpha);
}

void metal_scale(MetalContext& context, FrameViewMut frame, float r_alpha, float g_alpha, float b_alpha) {
    run_scale(context, frame, r_alpha, g_alpha, b_alpha);
}

void metal_scale(MetalContext& context, MetalFrame& frame, float r_alpha, float g_alpha, float b_alpha) {
    run_scale(context, MetalPointwise::bind(frame), r_alpha, g_alpha, b_alpha);
}

void metal_affine(MetalContext& context, FrameViewMut frame, float alpha, float beta) {
    run_affine(context, frame, alpha, beta);
}

void metal_affine(MetalContext& context, MetalFrame& frame, float alpha, float beta) {
    run_affine(context, MetalPointwise::bind(frame), alpha, beta);
}

void metal_affine(
    MetalContext& context, FrameViewMut frame,
    float r_alpha, float g_alpha, float b_alpha, float r_beta, float g_beta, float b_beta
) {
    run_affine(context, frame, r_alpha, g_alpha, b_alpha, r_beta, g_beta, b_beta);
}

void metal_affine(
    MetalContext& context, MetalFrame& frame,
    float r_alpha, float g_alpha, float b_alpha, float r_beta, float g_beta, float b_beta
) {
    run_affine(context, MetalPointwise::bind(frame), r_alpha, g_alpha, b_alpha, r_beta, g_beta, b_beta);
}

void metal_threshold(MetalContext& context, FrameViewMut frame, std::uint8_t t) {
    run_threshold(context, frame, t);
}

void metal_threshold(MetalContext& context, MetalFrame& frame, std::uint8_t t) {
    run_threshold(context, MetalPointwise::bind(frame), t);
}

void metal_threshold(
    MetalContext& context, FrameViewMut frame,
    std::uint8_t rt, std::uint8_t gt, std::uint8_t bt
) {
    run_threshold(context, frame, rt, gt, bt);
}

void metal_threshold(
    MetalContext& context, MetalFrame& frame,
    std::uint8_t rt, std::uint8_t gt, std::uint8_t bt
) {
    run_threshold(context, MetalPointwise::bind(frame), rt, gt, bt);
}

void metal_blend(
    MetalContext& context, FrameViewRef src1, FrameViewRef src2, FrameViewMut dst,
    float alpha
) {
    run_blend(context, src1, src2, dst, alpha);
}

void metal_blend(
    MetalContext& context, const MetalFrame& src1, const MetalFrame& src2, MetalFrame& dst,
    float alpha
) {
    run_blend(context, MetalPointwise::bind(std::as_const(src1)), MetalPointwise::bind(std::as_const(src2)), MetalPointwise::bind(dst), alpha);
}

void metal_blend(
    MetalContext& context, FrameViewRef src1, FrameViewRef src2, FrameViewMut dst,
    float r_alpha, float g_alpha, float b_alpha
) {
    run_blend(context, src1, src2, dst, r_alpha, g_alpha, b_alpha);
}

void metal_blend(
    MetalContext& context, const MetalFrame& src1, const MetalFrame& src2, MetalFrame& dst,
    float r_alpha, float g_alpha, float b_alpha
) {
    run_blend(context, MetalPointwise::bind(std::as_const(src1)), MetalPointwise::bind(std::as_const(src2)), MetalPointwise::bind(dst), r_alpha, g_alpha, b_alpha);
}

void metal_grayscale(MetalContext& context, FrameViewRef src, FrameViewMut dst) {
    run_grayscale(context, src, dst);
}

void metal_grayscale(MetalContext& context, const MetalFrame& src, MetalFrame& dst) {
    run_grayscale(context, MetalPointwise::bind(std::as_const(src)), MetalPointwise::bind(dst));
}

void metal_premultiply_alpha(MetalContext& context, FrameViewMut frame) {
    run_premultiply_alpha(context, frame);
}

void metal_premultiply_alpha(MetalContext& context, MetalFrame& frame) {
    run_premultiply_alpha(context, MetalPointwise::bind(frame));
}
