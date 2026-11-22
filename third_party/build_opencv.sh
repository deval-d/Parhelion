#!/bin/sh
set -eu

third_party=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(dirname "$third_party")

if [ ! -f "$third_party/opencv/CMakeLists.txt" ]; then
    git -C "$root" submodule update --init --depth 1 third_party/opencv
fi

if command -v cmake >/dev/null 2>&1; then
    cmake_bin=$(command -v cmake)
else
    cmake_bin="$third_party/.tools/bin/cmake"
    if [ ! -x "$cmake_bin" ]; then
        python3 -m venv "$third_party/.tools"
        "$third_party/.tools/bin/python" -m pip install cmake==3.31.6
    fi
fi

"$cmake_bin" -S "$third_party/opencv" -B "$third_party/opencv-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DBUILD_LIST=core -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF \
    -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF \
    -DWITH_LAPACK=OFF -DWITH_EIGEN=OFF

"$cmake_bin" --build "$third_party/opencv-build" --parallel 6
