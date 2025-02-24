cmake_minimum_required(VERSION 3.17)
project(topolosses)

# Find the Python development files
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Add the module to compile
Python3_add_library(_add_module MODULE "topolosses/losses/topograph/src/ext/topograph.cpp" WITH_SOABI)

# Install the module
install(TARGETS _add_module
        EXCLUDE_FROM_ALL
        COMPONENT python_modules
        DESTINATION ${PY_BUILD_CMAKE_IMPORT_NAME})




