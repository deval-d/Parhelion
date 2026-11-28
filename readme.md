Parhelion is a C++ image-processing engine planned for NEON/SME2 CPU kernels, multithreading, Metal GPU acceleration, and kernel fusion.
Super early stages right now.

## build and test

Requires CMake 3.21+, a C++20 compiler with SME2 ACLE intrinsics and streaming attributes,
and an AArch64 target. Running the kernels requires SME2 hardware and operating-system
support for streaming mode. There is no runtime fallback. Some lookup paths currently
assume 512-bit streaming vectors; other streaming vector lengths are not yet validated.

```sh
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release
```

The library is `build/release/libparhelion.a`. Public kernel headers are
`parhelion-1.hpp` and `parhelion-2.hpp`; layout conversion and image i/o have separate
headers in `include/`. `.clangd` reads the release build's generated compilation database.

Use the `debug` preset for an unoptimized build with debug information. The default
architecture flag is `-march=armv8-a+sme2`, not an Apple-specific CPU target. Optional
CPU tuning can be selected with `cmake --preset release -DPARHELION_CPU=apple-m4`.
`PARHELION_ARCH` overrides the architecture string. Cross-compilation uses the usual
CMake toolchain file; tests must be run on compatible hardware or through an emulator.

If CMake is not on your PATH, the existing local installation can be used as
`third_party/.tools/bin/cmake`; CTest is in the same directory.

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

## use from another cmake project

Clone Parhelion into `external/parhelion`, then add this to the application's
`CMakeLists.txt` after its `project()` declaration:

```cmake
add_subdirectory(external/parhelion)
add_executable(app main.cpp)
target_link_libraries(app PRIVATE parhelion::parhelion)
```

The target supplies the public include directories and C++20 requirement, builds the
kernel implementations, and links them into the application. Tests default to off
when Parhelion is included as a subdirectory. The application can include
`<parhelion-1.hpp>` and `<parhelion-2.hpp>` directly.

Alternatively, install a built library and its headers:

```sh
cmake --install build/release --prefix "$PWD/install"
```

A separate project can configure with `-DCMAKE_PREFIX_PATH=/path/to/parhelion/install`
and use the installed package:

```cmake
find_package(parhelion CONFIG REQUIRED)
target_link_libraries(app PRIVATE parhelion::parhelion)
```
