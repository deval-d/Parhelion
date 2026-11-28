#pragma once

#ifdef PARHELION_METAL_BENCHMARK
#include "metal-parhelion-1.hpp"

inline constexpr int benchmark_threads = 8;
inline constexpr char benchmark_mode[] = "Parhelion Metal: staging copies and GPU wait included. ";
inline MetalContext& benchmark_metal_context() {
    static MetalContext context;
    return context;
}

#define PARHELION_BENCH_CALL(kernel, ...) metal_##kernel(benchmark_metal_context(), __VA_ARGS__)
#elif defined(PARHELION_THREADED_BENCHMARK)
#include "parallel-parhelion-1.hpp"

inline constexpr int benchmark_threads = 8;
inline constexpr char benchmark_mode[] = "Parhelion: pool capacity 8, adaptive dispatch. ";
inline ThreadPool& benchmark_pool() {
    static ThreadPool pool(benchmark_threads);
    return pool;
}

#define PARHELION_BENCH_CALL(kernel, ...) parallel_##kernel(benchmark_pool(), __VA_ARGS__)
#else
inline constexpr int benchmark_threads = 1;
inline constexpr char benchmark_mode[] = "Single thread. ";
#define PARHELION_BENCH_CALL(kernel, ...) kernel(__VA_ARGS__)
#endif
