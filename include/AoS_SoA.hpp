#pragma once 
#include "image_types.hpp"

/// converts a `Frame`'s layout to `Planar`
void convert_to_planar(Frame& frame); 

/// converts a `Frame`'s layout to `Interleaved`
void convert_to_interleaved(Frame& frame); 
