cmake_minimum_required(VERSION 3.15)
project(topolosses)

list(APPEND CMAKE_PREFIX_PATH "/opt/miniconda")
list(APPEND CMAKE_PREFIX_PATH "/home/computacenter/miniconda3")


find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR} (version: ${EIGEN3_VERSION})")
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED) 
message(STATUS "Found OpenCV: ${OpenCV_DIR} (version: ${OpenCV_VERSION})")

include_directories(
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