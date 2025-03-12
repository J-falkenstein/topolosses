cmake_minimum_required(VERSION 3.15)
project(topolosses)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# this is the lcoation of the package when installed on a manylinux container initatied by cibuildwheel, not sure how this affects the cmake if the toml is not called with cibuildwheel but directly? Then we are not on a manylinux container
set(pybind11_DIR "/opt/python/cp39-cp39/lib/python3.9/site-packages/pybind11/share/cmake/pybind11")
list(APPEND CMAKE_PREFIX_PATH "/opt/python/cp39-cp39/lib/python3.9/site-packages/pybind11/share/cmake/pybind11")
set(OpenCV_DIR "/opt/miniconda/pkgs/opencv-4.10.0-py312hbc97090_2/lib/cmake/opencv4")
# set(Eigen3_DIR "${CMAKE_SOURCE_DIR}/external/eigen3/cmake")
# list(APPEND CMAKE_PREFIX_PATH "${Eigen3_DIR}")
# find_package(Eigen3 REQUIRED)

# Find Pacakes that need to beinstalled

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)


# Specify the extension module you want to build
add_library(_topograph MODULE 
    topolosses/losses/topograph/src/ext/topograph.cpp
    topolosses/losses/topograph/src/ext/_topograph.cpp
)

target_link_libraries(_topograph PRIVATE pybind11::module)

# Install the module to Python site packages
install(TARGETS _topograph
    LIBRARY DESTINATION ./topolosses/losses/topograph
)

if(OpenMP_CXX_FOUND)
  target_compile_options(_topograph PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(_topograph PRIVATE OpenMP::OpenMP_CXX)
endif()