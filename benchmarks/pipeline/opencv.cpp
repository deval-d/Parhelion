#include "pipeline.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace pipeline {

Frame opencv(const Frame& input, const Config& config, Timings* timings) {
    StageTimer timer(timings);
    const auto width = input.get_width();
    const auto height = input.get_height();
    const auto channels = input.get_n_channels();
    const auto bytes = input.view_ref().get_span().size();
    std::vector<std::byte> planes(bytes), blurred(bytes), pixels(bytes);
    // split only reads this header's borrowed input storage.
    cv::Mat source(height, width, CV_MAKETYPE(CV_8U, channels),
        const_cast<std::byte*>(input.view_ref().get_span().data()));
    cv::Mat output(height, width, CV_MAKETYPE(CV_8U, channels), pixels.data());
    std::array<cv::Mat, 4> before, after;
    for (std::size_t c = 0; c < channels; ++c) {
        before[c] = cv::Mat(height, width, CV_8UC1, planes.data() + c * width * height);
        after[c] = cv::Mat(height, width, CV_8UC1, blurred.data() + c * width * height);
    }
    timer.mark(setup);

    cv::split(source, before.data());
    timer.mark(split);
    const auto colors = channels == 4 ? 3 : channels;
    for (std::size_t c = 0; c < colors; ++c)
        cv::add(before[c], cv::Scalar(config.amount), before[c]);
    timer.mark(brightness);
    const auto side = static_cast<int>(2 * config.radius + 1);
    for (std::size_t c = 0; c < channels; ++c)
        cv::GaussianBlur(before[c], after[c], cv::Size(side, side), config.sigma, config.sigma,
            cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    timer.mark(blur);
    cv::merge(after.data(), channels, output);
    timer.mark(merge);
    return Frame(width, height, input.get_pixel_format(), FrameLayout::Interleaved, std::move(pixels));
}

} // namespace pipeline
