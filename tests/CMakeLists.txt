function(parhelion_test name source)
    add_executable(parhelion_test_${name} "${source}")
    target_link_libraries(parhelion_test_${name} PRIVATE parhelion::parhelion)
    add_test(NAME parhelion.${name} COMMAND parhelion_test_${name})
endfunction()

foreach(kernel IN ITEMS affine blend brighten copy fill histogram minmax scale threshold)
    parhelion_test(${kernel} "parhelion-1/${kernel}.cpp")
endforeach()
foreach(kernel IN ITEMS box_blur gaussian_blur laplacian sharpen sobel)
    parhelion_test(${kernel} "parhelion-2/${kernel}.cpp")
endforeach()
parhelion_test(parallel_pointwise threading/parhelion-1.cpp)
parhelion_test(layout AoS_SoA/main.cpp)
parhelion_test(image_io ls/main.cpp)

# image i/o tests write output files; keep those writes inside the build tree.
file(COPY ls/files DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/image_io")
set_tests_properties(parhelion.image_io PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/image_io"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    parhelion_test(metal_pointwise metal/parhelion-1.cpp)
endif()
