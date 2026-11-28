#include "pipeline.hpp"
#ifdef PARHELION_PIPELINE_OPENCV
#include <opencv2/core.hpp>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace pipeline;

struct Backend {
    const char* name;
    Frame (*run)(const Frame&, const Config&, Timings*);
    int tolerance;
};

std::vector<Backend> backends() {
    std::vector<Backend> result{{"parhelion", pipeline::parhelion, 1}};
#ifdef PARHELION_PIPELINE_OPENCV
    // zero disables parallel dispatch; gcd reports its pool size for a request of one.
    cv::setNumThreads(0);
    result.push_back({"opencv", pipeline::opencv, 2});
#endif
#ifdef PARHELION_PIPELINE_VIMAGE
    result.push_back({"vimage", pipeline::vimage, 1});
#endif
    return result;
}

Frame make_input(Coordinate size, PixelFormat format, int pattern = 2) {
    const std::size_t channels = format == PixelFormat::Gray ? 1 : format == PixelFormat::RGB ? 3 : 4;
    std::vector<std::byte> pixels(size.x * size.y * channels);
    std::uint32_t state = 123456789;
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const auto x = (i / channels) % size.x;
        const auto y = (i / channels) / size.x;
        const unsigned value = pattern == 0 ? 0 : pattern == 1 ? 255 : pattern == 2 ? state & 255
            : (x == size.x / 2 && y == size.y / 2 ? 255 : 0);
        pixels[i] = std::byte(value);
    }
    return Frame(size.x, size.y, format, FrameLayout::Interleaved, std::move(pixels));
}

std::vector<std::byte> reference(const Frame& input, const Config& config) {
    const auto width = input.get_width();
    const auto height = input.get_height();
    const auto channels = input.get_n_channels();
    const auto source = input.view_ref().get_span();
    std::vector<double> weights(2 * config.radius + 1);
    const auto radius = static_cast<std::ptrdiff_t>(config.radius);
    double total = 0;
    for (std::ptrdiff_t k = -radius; k <= radius; ++k) {
        const double distance = double(k) / config.sigma;
        weights[k + radius] = std::exp(-0.5 * distance * distance);
        total += weights[k + radius];
    }
    for (auto& weight : weights)
        weight /= total;
    std::vector<double> horizontal(width * height);
    std::vector<std::byte> result(source.size());
    for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                double sum = 0;
                for (std::ptrdiff_t k = -radius; k <= radius; ++k) {
                    const auto xx = std::clamp<std::ptrdiff_t>(std::ptrdiff_t(x) + k, 0, std::ptrdiff_t(width) - 1);
                    int value = std::to_integer<int>(source[(y * width + xx) * channels + c]);
                    if (channels != 4 || c != 3)
                        value = std::clamp(value + config.amount, 0, 255);
                    sum += value * weights[k + radius];
                }
                horizontal[y * width + x] = sum;
            }
        }
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                double sum = 0;
                for (std::ptrdiff_t k = -radius; k <= radius; ++k) {
                    const auto yy = std::clamp<std::ptrdiff_t>(std::ptrdiff_t(y) + k, 0, std::ptrdiff_t(height) - 1);
                    sum += horizontal[yy * width + x] * weights[k + radius];
                }
                result[(y * width + x) * channels + c] =
                    std::byte(static_cast<int>(std::nearbyint(std::clamp(sum, 0.0, 255.0))));
            }
        }
    }
    return result;
}

int verify(const Frame& actual, const Frame& input, const std::vector<std::byte>& expected,
    const Backend& backend, const Config& config) {
    if (actual.get_frame_layout() != FrameLayout::Interleaved ||
        actual.get_pixel_format() != input.get_pixel_format() ||
        actual.get_width() != input.get_width() || actual.get_height() != input.get_height())
        throw std::runtime_error(std::string(backend.name) + " returned incorrect image metadata");
    const auto pixels = actual.view_ref().get_span();
    if (pixels.size() != expected.size())
        throw std::runtime_error("incorrect result size");
    int maximum = 0;
    const int tolerance = config.radius == 0 ? 0 : backend.tolerance;
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        const auto difference = std::abs(std::to_integer<int>(pixels[i]) - std::to_integer<int>(expected[i]));
        maximum = std::max(maximum, difference);
        if (difference > tolerance)
            throw std::runtime_error(std::string(backend.name) + " mismatch at byte " + std::to_string(i)
                + " for " + std::to_string(input.get_width()) + "x" + std::to_string(input.get_height())
                + ": expected " + std::to_string(std::to_integer<int>(expected[i]))
                + ", received " + std::to_string(std::to_integer<int>(pixels[i])));
    }
    return maximum;
}

void check_input(const Frame& input, const std::vector<std::byte>& original) {
    const auto pixels = input.view_ref().get_span();
    if (!std::equal(pixels.begin(), pixels.end(), original.begin(), original.end()))
        throw std::runtime_error("pipeline modified its input");
}

void preflight(const std::vector<Backend>& libraries) {
    std::vector<int> errors(libraries.size(), 0);
    for (Coordinate size : {Coordinate(1, 1), Coordinate(1, 17), Coordinate(17, 1),
            Coordinate(7, 9), Coordinate(8, 9), Coordinate(9, 9), Coordinate(65, 17)}) {
        for (auto format : {PixelFormat::Gray, PixelFormat::RGB, PixelFormat::RGBA}) {
            for (int pattern : {0, 1, 2, 3}) {
                const auto input = make_input(size, format, pattern);
                const auto bytes = input.view_ref().get_span();
                const std::vector<std::byte> original(bytes.begin(), bytes.end());
                for (Config config : {Config{0, 1.0f, 0}, Config{-20, 0.8f, 1}, Config{20, 1.5f, 4}}) {
                    const auto expected = reference(input, config);
                    for (std::size_t b = 0; b < libraries.size(); ++b) {
                        const auto output = libraries[b].run(input, config, nullptr);
                        errors[b] = std::max(errors[b], verify(output, input, expected, libraries[b], config));
                        check_input(input, original);
                    }
                }
            }
        }
    }
    std::cout << "correctness preflight passed; maximum byte errors:";
    for (std::size_t b = 0; b < libraries.size(); ++b)
        std::cout << ' ' << libraries[b].name << '=' << errors[b];
    std::cout << '\n';
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void benchmark(Coordinate size, PixelFormat format, int samples,
    const std::vector<Backend>& libraries, std::uint64_t& checksum) {
    const Config config;
    const auto input = make_input(size, format);
    const auto bytes = input.view_ref().get_span();
    const std::vector<std::byte> original(bytes.begin(), bytes.end());
    const auto expected = reference(input, config);
    for (const auto& backend : libraries)
        verify(backend.run(input, config, nullptr), input, expected, backend, config);
    check_input(input, original);

    struct Measurements {
        std::vector<double> total;
        std::array<std::vector<double>, stage_count + 1> stages;
    };
    std::vector<Measurements> measurements(libraries.size());
    for (int sample = 0; sample < samples + 3; ++sample) {
        // rotate backend order; time whole calls separately from instrumented calls.
        for (std::size_t b = 0; b < libraries.size(); ++b) {
            const auto index = (sample + b) % libraries.size();
            auto& recorded = measurements[index];
            const auto& backend = libraries[index];
            auto run = [&](bool instrument) {
                Timings stages{};
                const auto start = Clock::now();
                const auto output = backend.run(input, config, instrument ? &stages : nullptr);
                const auto end = Clock::now();
                const double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
                const auto pixels = output.view_ref().get_span();
                checksum += std::to_integer<unsigned>(pixels[0])
                    + std::to_integer<unsigned>(pixels[pixels.size() / 2])
                    + std::to_integer<unsigned>(pixels.back());
                if (sample >= 3) {
                    if (!instrument) {
                        recorded.total.push_back(elapsed);
                    } else {
                        for (std::size_t stage = 0; stage < stage_count; ++stage)
                            recorded.stages[stage].push_back(stages[stage]);
                        const auto accounted = std::accumulate(stages.begin(), stages.end(), 0.0);
                        recorded.stages[stage_count].push_back(std::max(0.0, elapsed - accounted));
                    }
                }
                if (sample == samples + 2)
                    verify(output, input, expected, backend, config);
            };
            run(sample % 2 == 0);
            run(sample % 2 != 0);
        }
    }
    check_input(input, original);
    const char* name = format == PixelFormat::Gray ? "gray8" : format == PixelFormat::RGB ? "rgb8" : "rgba8";
    const auto ours = median(measurements[0].total);
    for (std::size_t b = 0; b < libraries.size(); ++b) {
        const auto total = median(measurements[b].total);
        std::cout << std::left << std::setw(7) << name
                  << std::setw(13) << (std::to_string(size.x) + "x" + std::to_string(size.y))
                  << std::setw(11) << libraries[b].name << std::right << std::setw(11) << total;
        for (auto& stage : measurements[b].stages)
            std::cout << std::setw(11) << median(stage);
        std::cout << std::setw(11) << total / ours << std::endl;
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool quick = false;
        for (int i = 1; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            if (argument == "--quick")
                quick = true;
            else if (argument == "--help") {
                std::cout << "usage: parhelion_bench_pipeline [--quick]\n"
                          << "default: 31 samples, 256x256 / 1920x1080 / 3840x2160\n"
                          << "quick: 5 samples, 17x9 / 256x256\n";
                return 0;
            } else {
                throw std::invalid_argument("unknown argument: " + std::string(argument));
            }
        }
        const auto libraries = backends();
#ifdef PARHELION_PIPELINE_OPENCV
        std::cout << "opencv " << CV_VERSION << "; threads " << cv::getNumThreads() << '\n';
#endif
        std::cout << "interleaved -> planar -> brightness +20 -> gaussian sigma 1.5 radius 4 -> interleaved\n"
                  << "single thread; fresh owning output; input unchanged; all backends use planar intermediates\n"
                  << "all per-call allocations, input copies, parameters and scratch included; no disk i/o\n"
                  << "brightness preserves alpha; blur processes alpha; replicated view edges\n"
                  << "median of " << (quick ? 5 : 31) << " samples after 3 warmups; warm repeated input\n"
                  << "total is timed without stage clocks; stage medians are from separate instrumented calls\n"
                  << "setup includes initial storage and parameters; other includes intermediate cleanup and return\n"
                  << "final output destruction and correctness checks excluded; independent medians need not sum\n";
        preflight(libraries);
        std::cout << std::left << std::setw(7) << "format" << std::setw(13) << "size" << std::setw(11) << "backend"
                  << std::right << std::setw(11) << "total ms" << std::setw(11) << "setup"
                  << std::setw(11) << "split" << std::setw(11) << "brighten" << std::setw(11) << "blur"
                  << std::setw(11) << "merge" << std::setw(11) << "other" << std::setw(11) << "time/ours"
                  << '\n' << std::fixed << std::setprecision(4);
        const std::vector<Coordinate> sizes = quick
            ? std::vector<Coordinate>{{17, 9}, {256, 256}}
            : std::vector<Coordinate>{{256, 256}, {1920, 1080}, {3840, 2160}};
        std::uint64_t checksum = 0;
        for (Coordinate size : sizes)
            for (auto format : {PixelFormat::Gray, PixelFormat::RGB, PixelFormat::RGBA})
                benchmark(size, format, quick ? 5 : 31, libraries, checksum);
        std::cout << "checksum: " << checksum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pipeline benchmark: " << error.what() << '\n';
        return 1;
    }
}
