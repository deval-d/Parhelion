Parhelion is a C++ image-processing engine planned for NEON/SME2 CPU kernels, multithreading, Metal GPU acceleration, and kernel fusion.
Super early stages right now.

## build and test

Requires CMake 3.21+, a C++20 compiler with SME2 ACLE intrinsics and streaming attributes,
and an AArch64 target. Running the kernels requires SME2 hardware and operating-system
support for streaming mode. There is no runtime fallback.

```sh
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release
```

The library is `build/release/libparhelion.a`. Public kernel headers are
`parhelion-1.hpp` and `parhelion-2.hpp`; layout conversion and image i/o have separate
headers in `include/`. 

The default architecture flag is `-march=armv8-a+sme2`.

## benchmarks

Benchmarks are optional and are not run by CTest. OpenCV comparisons require an
existing OpenCV build with `core` and `imgproc`. The bundled build is found automatically;
otherwise pass `-DOpenCV_DIR=/path/to/opencv/config` when configuring.

```sh
sh third_party/build_opencv.sh
cmake --preset benchmarks
cmake --build --preset benchmarks --parallel
./build/benchmarks/benchmarks/parhelion_bench_scale_opencv
```

The [pipeline benchmark](benchmarks/pipeline/readme.md) measures interleaved conversion,
brightness, Gaussian blur, and conversion back, including allocations and stage timings:

```sh
./build/benchmarks/benchmarks/parhelion_bench_pipeline
# shorter correctness and timing run
./build/benchmarks/benchmarks/parhelion_bench_pipeline --quick
```

Apple builds also enable Accelerate/vImage comparisons. Disable either backend with
`-DPARHELION_BENCHMARK_OPENCV=OFF` or `-DPARHELION_BENCHMARK_VIMAGE=OFF`.
