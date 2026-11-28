#include "pipeline.hpp"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace pipeline {
namespace {

void check(vImage_Error error) {
    if (error != kvImageNoError)
        throw std::runtime_error("vimage pipeline failed: " + std::to_string(error));
}

} // namespace

Frame vimage(const Frame& input, const Config& config, Timings* timings) {
    StageTimer timer(timings);
    const auto width = input.get_width();
    const auto height = input.get_height();
    const auto channels = input.get_n_channels();
    const auto count = width * height;
    const auto bytes = input.view_ref().get_span().size();
    std::vector<std::byte> planes(bytes), blurred(bytes), pixels(bytes);
    vImage_Buffer source{const_cast<std::byte*>(input.view_ref().get_span().data()),
        height, width, width * channels};
    vImage_Buffer output{pixels.data(), height, width, width * channels};
    std::array<vImage_Buffer, 4> before{}, after{};
    for (std::size_t c = 0; c < channels; ++c) {
        before[c] = {planes.data() + c * count, height, width, width};
        after[c] = {blurred.data() + c * count, height, width, width};
    }
    std::array<Pixel_8, 256> table;
    for (int value = 0; value < 256; ++value)
        table[value] = static_cast<Pixel_8>(std::clamp(value + config.amount, 0, 255));
    const auto side = 2 * config.radius + 1;
    std::vector<float> weights(side);
    double total = 0;
    for (std::size_t k = 0; k < side; ++k) {
        const double distance = (double(k) - double(config.radius)) / config.sigma;
        weights[k] = static_cast<float>(std::exp(-0.5 * distance * distance));
        total += weights[k];
    }
    for (auto& weight : weights)
        weight = static_cast<float>(weight / total);
    timer.mark(setup);

    if (channels == 1)
        std::memcpy(planes.data(), source.data, bytes);
    else if (channels == 3)
        check(vImageConvert_RGB888toPlanar8(&source, &before[0], &before[1], &before[2], kvImageDoNotTile));
    else
        // these conversions preserve positional channel order; no argb color interpretation is used.
        check(vImageConvert_ARGB8888toPlanar8(&source, &before[0], &before[1], &before[2], &before[3], kvImageDoNotTile));
    timer.mark(split);
    const auto colors = channels == 4 ? 3 : channels;
    for (std::size_t c = 0; c < colors; ++c)
        check(vImageTableLookUp_Planar8(&before[c], &before[c], table.data(), kvImageDoNotTile));
    timer.mark(brightness);
    const auto flags = kvImageEdgeExtend | kvImageDoNotTile;
    const auto size = vImageSepConvolve_Planar8(&before[0], &after[0], nullptr, 0, 0,
        weights.data(), side, weights.data(), side, 0.0f, 0, flags | kvImageGetTempBufferSize);
    if (size < 0)
        throw std::runtime_error("vimage scratch query failed");
    std::vector<std::byte> scratch(std::max<std::size_t>(1, size));
    for (std::size_t c = 0; c < channels; ++c)
        check(vImageSepConvolve_Planar8(&before[c], &after[c], scratch.data(), 0, 0,
            weights.data(), side, weights.data(), side, 0.0f, 0, flags));
    timer.mark(blur);
    if (channels == 1)
        std::memcpy(pixels.data(), blurred.data(), bytes);
    else if (channels == 3)
        check(vImageConvert_Planar8toRGB888(&after[0], &after[1], &after[2], &output, kvImageDoNotTile));
    else
        check(vImageConvert_Planar8toARGB8888(&after[0], &after[1], &after[2], &after[3], &output, kvImageDoNotTile));
    timer.mark(merge);
    return Frame(width, height, input.get_pixel_format(), FrameLayout::Interleaved, std::move(pixels));
}

} // namespace pipeline
