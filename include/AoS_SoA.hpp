#pragma once 
#include "image_types.hpp"

/// converts a `Frame`'s layout to `FrameLayout::Planar`
void convert_to_planar(Frame& frame); 

/// converts a `Frame`'s layout to `FrameLayout::Interleaved`
void convert_to_interleaved(Frame& frame); 
