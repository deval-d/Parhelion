#pragma once 
#include <cstddef>

namespace Color {
    struct Gray { 
        std::byte v; 
    };

    struct RGB { 
        std::byte r;
        std::byte g;
        std::byte b;
    }; 

    struct RGBA { 
        std::byte r;
        std::byte g;
        std::byte b;
        std::byte a; 
    };
}
