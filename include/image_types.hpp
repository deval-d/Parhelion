#pragma once 
#include <span>
#include <vector>
#include <cstddef>
#include <optional>
#include <utility>
#include <stdexcept>

enum PixelFormat { 
    Gray, 
    RGB, 
    RGBA
}; 

enum FrameLayout { 
    Interleaved, 
    Planar
}; 

struct Coordinate { 
    std::size_t x;
    std::size_t y; 

    Coordinate(
        std::size_t x_,
        std::size_t y_
    ) : x(x_), y(y_) {}  
};

template <typename T> 
struct FrameView { 
private:
    std::size_t w;
    std::size_t h; 
    std::size_t row_stride; 
    std::optional<std::size_t> channel_stride; 
    PixelFormat pixel_format;
    FrameLayout frame_layout;
    std::span<T> buf;

public:
    FrameView(
        std::size_t width,
        std::size_t height,
        std::size_t r_stride,
        std::optional<std::size_t> c_stride,
        PixelFormat format,
        FrameLayout layout,
        std::span<T> buffer 
    ) noexcept :
        w(width),
        h(height),
        row_stride(r_stride),
        channel_stride(c_stride),
        pixel_format(format),
        frame_layout(layout),
        buf(buffer)
    {} 

    // getters 
    std::size_t get_width(void)  const { return w; }
    std::size_t get_height(void) const { return h; } 
    std::size_t get_row_stride(void) const { return row_stride; }
    std::optional<std::size_t> get_channel_stride(void) const { return channel_stride; }
    PixelFormat get_pixel_format(void) const { return pixel_format; }
    FrameLayout get_frame_layout(void) const { return frame_layout; }
    std::span<T> get_span(void) const { return buf; }
}; 

using FrameViewRef = FrameView<const std::byte>;
using FrameViewMut = FrameView<std::byte>; 

struct Frame {
private: 
    std::size_t w; 
    std::size_t h;
    PixelFormat pixel_format; 
    FrameLayout frame_layout;
    std::vector<std::byte> buf; 

public:
    /// constructor 
    explicit Frame(
        std::size_t width,
        std::size_t height,
        PixelFormat format,
        FrameLayout layout,
        std::vector<std::byte> buffer
    ) noexcept :
        w(width),
        h(height), 
        pixel_format(format), 
        frame_layout(layout), 
        buf(std::move(buffer))
    {} 

    // getters 
    std::size_t get_width(void)  const { return w; }
    std::size_t get_height(void) const { return h; }
    std::size_t get_n_channels(void) const { 
        switch (pixel_format) { 
            case (Gray) : return 1; 
            case (RGB)  : return 3;
            case (RGBA) : return 4; 
        }
    }
    PixelFormat get_pixel_format(void) const { return pixel_format; }
    FrameLayout get_frame_layout(void) const { return frame_layout; }

    // setters 
    void set_frame_layout(FrameLayout layout) { frame_layout = layout; }
    void replace_buf(std::vector<std::byte>&& new_buf) {
        buf = std::move(new_buf); 
    }

    /// immutable full view 
    FrameViewRef view_ref(void) const {
        std::optional<std::size_t> channel_stride = 
            (frame_layout == Interleaved) 
            ? std::nullopt 
            : std::optional<std::size_t>(w * h);  

        FrameViewRef view(
            w,
            h,
            w * get_n_channels(),
            channel_stride, 
            pixel_format,
            frame_layout,
            std::span<const std::byte>(
                buf.begin(), 
                buf.end()
            )
        ); 
          
        return view; 
    }

    // mutable full view 
    FrameViewMut view_mut(void) { 
        std::optional<std::size_t> channel_stride = 
            (frame_layout == Interleaved) 
            ? std::nullopt 
            : std::optional<std::size_t>(w * h);  

        FrameViewMut view(
            w,
            h,
            w * get_n_channels(),
            channel_stride, 
            pixel_format,
            frame_layout,
            std::span<std::byte>(
                buf.begin(), 
                buf.end()
            )
        ); 
          
        return view; 
    }
            
    /// immutable subview 
    FrameViewRef view_ref(
        Coordinate xy,
        std::size_t width,
        std::size_t height
    ) const { 
        if (width == 0 || height == 0)
            throw std::invalid_argument("view dimensions must be nonzero");

        if (
            (xy.x >= w || xy.y >= h) ||
            (width > w - xy.x || height > h - xy.y)
        )
            throw std::out_of_range("view is outside the frame");

        std::size_t channels = get_n_channels(); 

        std::size_t beg_idx;
        std::size_t end_idx; 
        std::size_t row_stride;
        std::optional<std::size_t> channel_stride; 
        switch (frame_layout) { 
            case (Interleaved): { 
                row_stride = w * channels;
                channel_stride = std::nullopt; 
                beg_idx = xy.x * channels + xy.y * row_stride;
                end_idx = (xy.x + width) * channels + (xy.y + height - 1) * row_stride; 
                break; 
            }
            case (Planar): {
                row_stride = w; 
                channel_stride = w * h; 
                beg_idx = xy.x + xy.y * row_stride;
                end_idx = xy.x + width + (xy.y + height - 1) * row_stride + *channel_stride * (channels - 1); 
                break;
            }
        }
 
        FrameViewRef view(
            width,
            height,
            row_stride,
            channel_stride, 
            pixel_format,
            frame_layout,
            std::span<const std::byte>(
                buf.begin() + beg_idx, 
                buf.begin() + end_idx 
            )
        ); 

        return view; 
    }

    /// mutable subview 
    FrameViewMut view_mut(
        Coordinate xy,
        std::size_t width,
        std::size_t height
    ) { 
        if (width == 0 || height == 0)
            throw std::invalid_argument("view dimensions must be nonzero");

        if (
            (xy.x >= w || xy.y >= h) ||
            (width > w - xy.x || height > h - xy.y)
        )
            throw std::out_of_range("view is outside the frame");

        std::size_t channels = get_n_channels(); 
        std::size_t row_stride = w * channels;

        std::size_t beg_idx;
        std::size_t end_idx; 
        std::optional<std::size_t> channel_stride; 
        switch (frame_layout) { 
            case (Interleaved): { 
                channel_stride = std::nullopt;  
                beg_idx = xy.x * channels + xy.y * row_stride;
                end_idx = (xy.x + width) * channels + (xy.y + height - 1) * row_stride; 
                break; 
            }
            case (Planar): {
                channel_stride = w * h; 
                beg_idx = xy.x + xy.y * row_stride;
                end_idx = xy.x + width + (xy.y + height - 1) * row_stride + *channel_stride * (channels - 1); 
                break;
            }
        }
 
        FrameViewMut view(
            width,
            height,
            row_stride,
            channel_stride, 
            pixel_format,
            frame_layout,
            std::span<std::byte>(
                buf.begin() + beg_idx, 
                buf.begin() + end_idx 
            )
        ); 

        return view; 
    }
};
