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

# Build the fusion algorithm as a static library so its compiled code can be
# linked into the _native extension that Python imports.
add_library(wbf_core STATIC src/scalar.cpp)
set_target_properties(wbf_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(wbf_core PUBLIC src)
pybind11_add_module(_native MODULE NO_EXTRAS src/bindings.cpp)
target_link_libraries(_native PRIVATE wbf_core)
# Apply the same floating-point settings to both targets. Compiler shortcuts
# such as fusing multiplication and addition can change when values round.
foreach(target wbf_core _native)
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /fp:strict)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
  endif()
endforeach()

# The install prefix itself is added to PYTHONPATH, on both Unix and Windows.
install(TARGETS _native LIBRARY DESTINATION cuda_wbf RUNTIME DESTINATION cuda_wbf)
install(DIRECTORY python/cuda_wbf/ DESTINATION cuda_wbf FILES_MATCHING PATTERN "*.py")
