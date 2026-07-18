cmake_minimum_required(VERSION 3.20)
project(r2ie_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Locate Python dev headers (provides Python::Module target).
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Resolve pybind11 include dir (works with pip-installed pybind11, which does
# not always ship a CMake config package).
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
    "import pybind11, sys; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(pybind11_headers INTERFACE)
target_include_directories(pybind11_headers INTERFACE "${PYBIND11_INCLUDE}")

# Helper: define a pybind11 extension module without depending on the
# pybind11 CMake config package (which may be absent when pip-installed).
function(pybind11_module_target TARGET SRC)
    add_library(${TARGET} MODULE ${SRC})
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${TARGET} PRIVATE Python::Module pybind11_headers)
    # On Linux/macOS, extension modules must not be prefixed with "lib".
    set_target_properties(${TARGET} PROPERTIES PREFIX "")
endfunction()

pybind11_module_target(_r2ie_cpp bindings.cpp hdq_memory.hpp)

# Standalone C++ unit tests (header-only asserts, no external deps).
add_executable(test_hdq_memory ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/cpp/test_hdq_memory.cpp)
target_include_directories(test_hdq_memory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
enable_testing()
add_test(NAME test_hdq_memory COMMAND test_hdq_memory)
