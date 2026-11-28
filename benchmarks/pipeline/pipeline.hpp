#pragma once
#include "image_types.hpp"
#include <array>
#include <chrono>
#include <cstddef>

namespace pipeline {

using Clock = std::chrono::steady_clock;
enum Stage : std::size_t { setup, split, brightness, blur, merge, stage_count };
using Timings = std::array<double, stage_count>;

struct Config {
    int amount = 20;
    float sigma = 1.5f;
    std::size_t radius = 4;
};

class StageTimer {
    Timings* timings;
    Clock::time_point start;

public:
    explicit StageTimer(Timings* output) : timings(output) {
        if (timings)
            start = Clock::now();
    }

    void mark(Stage stage) {
        if (timings) {
            const auto end = Clock::now();
            (*timings)[stage] = std::chrono::duration<double, std::milli>(end - start).count();
            start = end;
        }
    }
};

/// returns an interleaved result; input storage is unchanged; timings are optional.
Frame parhelion(const Frame& input, const Config& config, Timings* timings = nullptr);
#ifdef PARHELION_PIPELINE_OPENCV
Frame opencv(const Frame& input, const Config& config, Timings* timings = nullptr);
#endif
#ifdef PARHELION_PIPELINE_VIMAGE
Frame vimage(const Frame& input, const Config& config, Timings* timings = nullptr);
#endif

} // namespace pipeline
