cmake_minimum_required(VERSION 3.15)
project(topolosses)

# Find Pacakes that need to beinstalled
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
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
add_library(topograph MODULE 
    topolosses/losses/topograph/src/ext/topograph.cpp
    topolosses/losses/topograph/src/ext/_topograph.cpp
)

target_link_libraries(topograph PRIVATE pybind11::module)

# Install the module to Python site packages
install(TARGETS topograph
    LIBRARY DESTINATION ./topolosses/losses/topograph/src
)

if(OpenMP_CXX_FOUND)
  target_compile_options(topograph PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(topograph PRIVATE OpenMP::OpenMP_CXX)
endif()