#include "image_load.hpp"
#include "image_write.hpp"
#include <cassert>
#include <iostream>
#include <string> 

void check_frame(
    const Frame& frame,
    PixelFormat format,
    FrameLayout layout,
    std::size_t width,
    std::size_t height
) {
    bool passed = true;

    auto check = [&](const char* name, auto received, auto expected) {
        if (received != expected) {
            std::cerr << name
                      << ": received " << received
                      << ", expected " << expected << '\n';
            passed = false;
        }
    };

    check("format", static_cast<int>(frame.get_pixel_format()), static_cast<int>(format));
    check("layout", static_cast<int>(frame.get_frame_layout()), static_cast<int>(layout));
    check("width", frame.get_width(), width);
    check("height", frame.get_height(), height);

    if (!passed) { 
        throw std::runtime_error("test failed"); 
    }
}
int test(
    std::string file_in,
    std::string file_out,
    PixelFormat format,
    FrameLayout layout,
    std::size_t width,
    std::size_t height
) { 
    Frame frame = image_load(file_in.c_str()); 
    check_frame(frame, format, layout, width, height); 
    image_write(file_out.c_str(), frame); 
    return 0; 
}

int gray8(void) {
    std::string file_in = "./files/gray_in.png";
    std::string file_out = "./files/gray_out.png";

    return test(file_in, file_out, PixelFormat::Gray, FrameLayout::Interleaved, 1024, 1024); 
}

int rgb8(void) { 
    std::string file_in = "./files/rgb_in.png";
    std::string file_out = "./files/rgb_out.png";

    return test(file_in, file_out, PixelFormat::RGB, FrameLayout::Interleaved, 259, 194); 
}

int rgba8(void) { 
    std::string file_in = "./files/rgba_in.png";
    std::string file_out = "./files/rgba_out.png";

    return test(file_in, file_out, PixelFormat::RGBA, FrameLayout::Interleaved, 256, 256); 
}

int main(void) { 
    int status = 0;
    status |= gray8();
    status |= rgb8();
    status |= rgba8(); 
    return status; 
}

