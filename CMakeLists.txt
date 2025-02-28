cmake_minimum_required(VERSION 3.15)
project(topolosses)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)


# Define the C++ extension
add_library(topograph MODULE 
    topolosses/losses/topograph/src/ext/topograph.cpp 
    topolosses/losses/topograph/src/ext/_topograph.cpp
)


# Link the pybind11 library to your C++ extension
target_link_libraries(topograph PRIVATE pybind11::module)

# Set any necessary compile flags
target_compile_options(topograph PRIVATE -O3)

# Ensure the generated module has the proper file extension for Python on your platform
set_target_properties(topograph PROPERTIES
    PREFIX ""
    SUFFIX ".so"   # Use .pyd for Windows, .so for Linux/MacOS
)

# Include directories for pybind11 and your own headers
target_include_directories(topograph PRIVATE 
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    topolosses/losses/topograph/src/ext
)

# Install the Python module
install(TARGETS topograph DESTINATION ${PYTHON_SITE_PACKAGES}/topolosses/losses/topograph)

