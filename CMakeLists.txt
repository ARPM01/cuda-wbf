cmake_minimum_required(VERSION 3.24)
project(cuda_wbf VERSION 0.1.0 LANGUAGES CXX)

option(WBF_ENABLE_CUDA "Build CUDA fusion backends" OFF)
if(WBF_ENABLE_CUDA)
  message(FATAL_ERROR "CUDA backends are not implemented yet. Configure with -DWBF_ENABLE_CUDA=OFF.")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
if(NOT Python_VERSION VERSION_LESS 3.12)
  message(FATAL_ERROR "The pinned WBF environment requires Python 3.11. Set Python_EXECUTABLE to its interpreter.")
endif()
find_package(pybind11 2.13 CONFIG REQUIRED)

pybind11_add_module(_native MODULE NO_EXTRAS src/bindings.cpp)
target_compile_features(_native PRIVATE cxx_std_17)
set_target_properties(_native PROPERTIES CXX_EXTENSIONS OFF)
if(MSVC)
  target_compile_options(_native PRIVATE /W4 /fp:strict)
else()
  target_compile_options(_native PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
endif()

# The install prefix itself is added to PYTHONPATH, on both Unix and Windows.
install(TARGETS _native LIBRARY DESTINATION cuda_wbf RUNTIME DESTINATION cuda_wbf)
install(DIRECTORY python/cuda_wbf/ DESTINATION cuda_wbf FILES_MATCHING PATTERN "*.py")
