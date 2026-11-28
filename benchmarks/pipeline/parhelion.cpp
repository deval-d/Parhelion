#include "pipeline.hpp"
#include "AoS_SoA.hpp"
#include "parhelion-1.hpp"
#include "parhelion-2.hpp"

namespace pipeline {

Frame parhelion(const Frame& input, const Config& config, Timings* timings) {
    StageTimer timer(timings);
    Frame working = input;
    Frame output(input.get_width(), input.get_height(), input.get_pixel_format(),
        FrameLayout::Planar, std::vector<std::byte>(input.view_ref().get_span().size()));
    timer.mark(setup);

    convert_to_planar(working);
    timer.mark(split);
    brighten(working.view_mut(), config.amount);
    timer.mark(brightness);
    gaussian_blur(working.view_ref(), output.view_mut(), config.sigma, config.radius);
    timer.mark(blur);
    convert_to_interleaved(output);
    timer.mark(merge);
    return output;
}

} // namespace pipeline
