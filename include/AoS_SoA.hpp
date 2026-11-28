#pragma once 
#include "image_types.hpp"

/// converts to planar storage; invalidates existing views if storage changes.
void convert_to_planar(Frame& frame); 

/// converts to interleaved storage; invalidates existing views if storage changes.
void convert_to_interleaved(Frame& frame); 
