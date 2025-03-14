cmake_minimum_required(VERSION 3.15)
project(topolosses)


## All of these links are not needed for local build, but inside the containers somehow cmake cannt find the libraries
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# # this is the lcoation of the package when installed on a manylinux container initatied by cibuildwheel, not sure how this affects the cmake if the toml is not called with cibuildwheel but directly? Then we are not on a manylinux container
# set(pybind11_DIR "/opt/python/cp39-cp39/lib/python3.9/site-packages/pybind11/share/cmake/pybind11")
# list(APPEND CMAKE_PREFIX_PATH "/opt/python/cp39-cp39/lib/python3.9/site-packages/pybind11/share/cmake/pybind11")
# set(OpenCV_DIR "/opt/miniconda/pkgs/opencv-4.10.0-py312hbc97090_2/lib/cmake/opencv4")
# message(STATUS "Source dir: ${CMAKE_SOURCE_DIR}")
# set(Eigen3_DIR "${CMAKE_SOURCE_DIR}/topolosses/external/eigen3/cmake")
# message(STATUS "Eigen3 found at: ${EIGEN3_INCLUDE_DIR}")
# list(APPEND CMAKE_PREFIX_PATH "${Eigen3_DIR}")


find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED) # get from conda environemnt
# find out where i get my eigen and boost from when building locally 
find_package(Boost REQUIRED) 
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)


include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

pybind11_add_module(_topograph 
    topolosses/losses/topograph/src/ext/topograph.cpp
    topolosses/losses/topograph/src/ext/_topograph.cpp
)
install(TARGETS _topograph DESTINATION topolosses/losses/topograph/src)

target_link_libraries(_topograph PRIVATE ${OpenCV_LIBS})

if(OpenMP_CXX_FOUND)
  target_compile_options(_topograph PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(_topograph PRIVATE OpenMP::OpenMP_CXX)
endif()