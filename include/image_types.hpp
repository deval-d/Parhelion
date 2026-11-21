#pragma once 
#include <span>
#include <vector>
#include <cstddef>
#include <optional>
#include <utility>
#include <type_traits>
#include <stdexcept>

enum class PixelFormat { 
    Gray, 
    RGB, 
    RGBA
}; 

enum class FrameLayout { 
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

    FrameView<std::byte> channel_mut(std::size_t channel) const
        requires std::is_same_v<T, std::byte>
    {
        if (frame_layout != FrameLayout::Planar)
            throw std::invalid_argument("channel views require planar storage");

        std::size_t channels;
        switch (pixel_format) { 
            case PixelFormat::Gray : channels = 1; break; 
            case PixelFormat::RGB  : channels = 3; break; 
            case PixelFormat::RGBA : channels = 4; break; 
        }; 

        if (channel >= channels)
            throw std::out_of_range("channel is outside the frame");

        std::size_t offset = channel * channel_stride.value();
        std::size_t size = (h - 1) * row_stride + w;

        return FrameView<std::byte>(
            w, h,
            row_stride, std::nullopt,
            PixelFormat::Gray, FrameLayout::Planar,
            buf.subspan(offset, size)
        );
    }
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
    ) :
        w(width),
        h(height), 
        pixel_format(format), 
        frame_layout(layout), 
        buf(std::move(buffer))
    {
        if (buf.size() != width * height * get_n_channels())
            throw std::invalid_argument("frame dimensions don't match buffer");  
    } 

    // getters 
    std::size_t get_width(void)  const { return w; }
    std::size_t get_height(void) const { return h; }
    std::size_t get_n_channels(void) const { 
        switch (pixel_format) { 
            case PixelFormat::Gray : return 1; 
            case PixelFormat::RGB  : return 3;
            case PixelFormat::RGBA : return 4; 
        }
    }
    PixelFormat get_pixel_format(void) const { return pixel_format; }
    FrameLayout get_frame_layout(void) const { return frame_layout; }

    // setters 
    void set_frame_layout(FrameLayout layout) { frame_layout = layout; }
    /// WARNING: any previously obtained `FrameView`s would become invalid
    void replace_buf(std::vector<std::byte>&& new_buf) {
        buf = std::move(new_buf); 
    }

    /// immutable full view 
    FrameViewRef view_ref(void) const {
        std::size_t row_stride; 
        std::optional<std::size_t> channel_stride;
        switch (frame_layout) { 
            case (FrameLayout::Interleaved): { 
                channel_stride = std::nullopt;
                row_stride = w * get_n_channels();
                break;
            }
            case (FrameLayout::Planar): { 
                channel_stride = w * h;
                row_stride = w; 
                break;
            }
        }

        FrameViewRef view(
            w,
            h,
            row_stride,
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

    /// mutable full view
    FrameViewMut view_mut(void) {
        std::size_t row_stride; 
        std::optional<std::size_t> channel_stride;
        switch (frame_layout) { 
            case FrameLayout::Interleaved:  
                channel_stride = std::nullopt;
                row_stride = w * get_n_channels();
                break;

            case FrameLayout::Planar: 
                channel_stride = w * h;
                row_stride = w; 
                break;
        }

        FrameViewMut view(
            w,
            h,
            row_stride,
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
            case FrameLayout::Interleaved: 
                row_stride = w * channels;
                channel_stride = std::nullopt; 
                beg_idx = xy.x * channels + xy.y * row_stride;
                end_idx = (xy.x + width) * channels + (xy.y + height - 1) * row_stride; 
                break; 
        
            case FrameLayout::Planar: 
                row_stride = w; 
                channel_stride = w * h; 
                beg_idx = xy.x + xy.y * row_stride;
                end_idx = xy.x + width + (xy.y + height - 1) * row_stride + *channel_stride * (channels - 1); 
                break;
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

        std::size_t beg_idx;
        std::size_t end_idx; 
        std::size_t row_stride;
        std::optional<std::size_t> channel_stride; 
        switch (frame_layout) { 
            case FrameLayout::Interleaved:  
                row_stride = w * channels; 
                channel_stride = std::nullopt;  
                beg_idx = xy.x * channels + xy.y * row_stride;
                end_idx = (xy.x + width) * channels + (xy.y + height - 1) * row_stride; 
                break; 
        
            case FrameLayout::Planar: 
                row_stride = w; 
                channel_stride = w * h; 
                beg_idx = xy.x + xy.y * row_stride;
                end_idx = xy.x + width + (xy.y + height - 1) * row_stride + *channel_stride * (channels - 1); 
                break;
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
