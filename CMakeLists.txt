cmake_minimum_required(VERSION 3.21)

project(parhelion VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

option(PARHELION_BUILD_TESTS "build correctness tests" ${PROJECT_IS_TOP_LEVEL})
option(PARHELION_BUILD_BENCHMARKS "build library comparison benchmarks" OFF)
set(PARHELION_ARCH "armv8-a+sme2" CACHE STRING "arm architecture and required extensions")
set(PARHELION_CPU "" CACHE STRING "optional cpu-specific tuning target")

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
    message(FATAL_ERROR "parhelion currently requires an aarch64 target with neon and sme2")
endif()

set(parhelion_arch_options "-march=${PARHELION_ARCH}")
if(PARHELION_CPU)
    list(PREPEND parhelion_arch_options "-mcpu=${PARHELION_CPU}")
endif()

# check the acle intrinsics and streaming attributes without executing sme code.
cmake_push_check_state(RESET)
string(JOIN " " CMAKE_REQUIRED_FLAGS ${parhelion_arch_options})
unset(PARHELION_HAS_SME2_ACLE CACHE)
check_cxx_source_compiles([=[
    #include <arm_sme.h>
    #include <arm_sve.h>
    __arm_locally_streaming __arm_new("za")
    void probe(float* output) {
        svfloat32_t value = svdup_f32(1.0f);
        svfloat32x4_t group = svcreate4_f32(value, value, value, value);
        svzero_za();
        svmla_single_za32_f32_vg1x4(0, group, value);
        group = svread_za32_f32_vg1x4(0);
        svst1_f32(svptrue_b32(), output, svget4_f32(group, 0));
    }
    int main() { return 0; }
]=] PARHELION_HAS_SME2_ACLE)
cmake_pop_check_state()
if(NOT PARHELION_HAS_SME2_ACLE)
    message(FATAL_ERROR
        "the compiler must support sme2 acle intrinsics and arm streaming attributes; "
        "check PARHELION_ARCH, PARHELION_CPU and the compiler diagnostics")
endif()

add_library(parhelion STATIC
    src/AoS_SoA.cpp
    src/image_stb.cpp
    src/parhelion-1/affine.cpp
    src/parhelion-1/blend.cpp
    src/parhelion-1/brighten.cpp
    src/parhelion-1/copy.cpp
    src/parhelion-1/fill.cpp
    src/parhelion-1/grayscale.cpp
    src/parhelion-1/histogram.cpp
    src/parhelion-1/minmax.cpp
    src/parhelion-1/premultiply_alpha.cpp
    src/parhelion-1/scale.cpp
    src/parhelion-1/threshold.cpp
    src/parhelion-2/box_blur.cpp
    src/parhelion-2/convolution.cpp
    src/parhelion-2/gaussian_blur.cpp
    src/parhelion-2/laplacian.cpp
    src/parhelion-2/sharpen.cpp
    src/parhelion-2/sobel.cpp
)
add_library(parhelion::parhelion ALIAS parhelion)

target_compile_features(parhelion PUBLIC cxx_std_20)
target_compile_options(parhelion PRIVATE ${parhelion_arch_options})
target_include_directories(parhelion PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/parhelion>
)
set_target_properties(parhelion PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

if(PARHELION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(PARHELION_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

install(TARGETS parhelion EXPORT parhelionTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parhelion
    FILES_MATCHING PATTERN "*.hpp"
)
# public image i/o headers include stb directly.
install(FILES third_party/stb/stb_image.h third_party/stb/stb_image_write.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parhelion
)

set(parhelion_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/parhelion")
configure_package_config_file(cmake/parhelionConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/parhelionConfig.cmake"
    INSTALL_DESTINATION "${parhelion_cmake_dir}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/parhelionConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(EXPORT parhelionTargets NAMESPACE parhelion::
    DESTINATION "${parhelion_cmake_dir}"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/parhelionConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/parhelionConfigVersion.cmake"
    DESTINATION "${parhelion_cmake_dir}"
)
