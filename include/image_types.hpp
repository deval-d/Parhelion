#pragma once 
#include <span>
#include <vector>
#include <cstddef>
#include <optional>
#include <utility>

enum PixelFormat { 
    Gray, 
    RGB, 
    RGBA
}; 

enum FrameLayout { 
    Interleaved, 
    Planar
}; 

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
    PixelFormat get_pixel_format(void) const { return pixel_format; }
    FrameLayout get_frame_layout(void) const { return frame_layout; }

    // setters 
    void set_frame_layout(FrameLayout layout) { frame_layout = layout; }

    /// immutable span 
    std::optional<std::span<const std::byte>> get_buf_span_ref(
        std::size_t beg_idx, 
        std::size_t end_idx
    ) const { 
        if (beg_idx > end_idx || end_idx > buf.size()) 
            return std::nullopt;  

        return std::span<const std::byte>(
            buf.begin() + beg_idx, 
            buf.begin() + end_idx
        );  
    }

    /// mutable span 
    std::optional<std::span<std::byte>> get_buf_span_mut(
        std::size_t beg_idx,
        std::size_t end_idx
    ) { 
        if (beg_idx > end_idx || end_idx > buf.size())
            return std::nullopt; 

        return std::span<std::byte>(
            buf.begin() + beg_idx,
            buf.begin() + end_idx
        ); 
    } 

    void replace_buf(std::vector<std::byte>&& new_buf) {
        buf = std::move(new_buf); 
    }
};


