cmake_minimum_required(VERSION 3.24)
project(cuda_wbf VERSION 0.1.0 LANGUAGES CXX)

option(WBF_ENABLE_CUDA "Build CUDA fusion backends" OFF)

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
if(WBF_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 11.8 REQUIRED)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8)
    message(FATAL_ERROR "CUDA 11.8 or newer is required.")
  endif()
  target_sources(wbf_core PRIVATE src/cuda.cu)
  target_compile_definitions(wbf_core PUBLIC WBF_HAS_CUDA=1)
  target_link_libraries(wbf_core PUBLIC CUDA::cudart)
  set_target_properties(wbf_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
  target_compile_options(wbf_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
  if(MSVC)
    target_compile_options(wbf_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/fp:strict>)
  else()
    target_compile_options(wbf_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math,-ffp-contract=off>)
  endif()
endif()
pybind11_add_module(_native MODULE NO_EXTRAS src/bindings.cpp)
target_link_libraries(_native PRIVATE wbf_core)
# Apply the same floating-point settings to both targets. Compiler shortcuts
# such as fusing multiplication and addition can change when values round.
foreach(target wbf_core _native)
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
  if(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4;/fp:strict>)
  else()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-fno-fast-math;-ffp-contract=off>)
  endif()
endforeach()

# The install prefix itself is added to PYTHONPATH, on both Unix and Windows.
install(TARGETS _native LIBRARY DESTINATION cuda_wbf RUNTIME DESTINATION cuda_wbf)
install(DIRECTORY python/cuda_wbf/ DESTINATION cuda_wbf FILES_MATCHING PATTERN "*.py")
