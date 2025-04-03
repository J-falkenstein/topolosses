cmake_minimum_required(VERSION 3.15)
project(topolosses)


# TODO these could probably be set by env varibale in yml or toml file
list(APPEND CMAKE_PREFIX_PATH "/opt/miniconda")
list(APPEND CMAKE_PREFIX_PATH "/home/computacenter/miniconda3")
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp:/opt/homebrew/opt/llvm")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/miniconda")

## from claude
# Set C++17 standard for all targets
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

## from original cmakelist betti matching
# cmake_minimum_required(VERSION 3.5)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
# set(Python_VIRTUALENV FIRST)
# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE "Release")
# endif()


find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR} (version: ${EIGEN3_VERSION})")
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED) 
message(STATUS "Found OpenCV: ${OpenCV_DIR} (version: ${OpenCV_VERSION})")
# for betti matching
find_package(Threads REQUIRED)
message(STATUS "Found Threads: ${CMAKE_THREAD_LIBS_INIT}")

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

### from here on alle new betti matching stuff
# Build betti_matching extension
# todo adjust to 
pybind11_add_module(betti_matching
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/_BettiMatching.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/BettiMatching.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/utils.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/data_structures.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_1D/BettiMatching.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_1D/dimension_0.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_1D/data_structures.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_2D/BettiMatching.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_2D/dimension_0.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_2D/dimension_1.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_2D/data_structures.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_3D/BettiMatching.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_3D/dimension_0.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_3D/dimension_1.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_3D/dimension_2.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_3D/data_structures.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_3D/enumerators.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_nD/BettiMatching.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_nD/dimension_0.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_nD/inter_dimensions.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_nD/top_dimension.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_nD/data_structures.cpp
    topolosses/losses/betti_matching/src/ext/Betti-Matching-3D-global-build/src/src_nD/enumerators.cpp
)

set_target_properties(betti_matching
    PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
)
target_link_libraries(betti_matching PRIVATE Threads::Threads)

# Install betti_matching module -> TODO does it have an effect that it is done after target_link_libraries? 
install(TARGETS betti_matching DESTINATION topolosses/losses/betti_matching/src)
