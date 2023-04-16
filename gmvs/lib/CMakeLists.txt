cmake_minimum_required(VERSION 3.1)
project(
    PLANAR_INIT
    VERSION 1.0
    LANGUAGES CXX
)

find_package(CUDA 6.0 REQUIRED) # For Cuda Managed Memory and c++11
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)

set(
    CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++14 --compiler-options -Wall -gencode arch=compute_86,code=sm_86)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++14)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
endif()

include_directories(
    include
    ${NUMPY_INCLUDE_DIR}
)

# add pybind11 module
add_subdirectory(dependencies/pybind11)

# add the cuda sources
cuda_add_library(
    planar_initer SHARED
    src/bindings.cpp
    src/ndarray_converter.cpp
    src/planar_initer.cpp
    src/planar_initer.cu
    src/patch_match_kernel.cu)

target_link_libraries(
    planar_initer
    ${OpenCV_LIBS}
    pybind11::module)
pybind11_extension(planar_initer)
