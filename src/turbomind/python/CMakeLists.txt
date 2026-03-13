# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)
project(_turbomind LANGUAGES CXX CUDA)

find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    execute_process(COMMAND "pybind11-config" "--cmakedir"
                    RESULT_VARIABLE _COMMAND_SUCCESS
                    OUTPUT_VARIABLE pybind11_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG)
endif()

pybind11_add_module(${PROJECT_NAME} bind.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE turbomind xgrammar)

pybind11_add_module(_xgrammar xgrammar_bind.cpp)
target_link_libraries(_xgrammar PRIVATE core xgrammar)
target_compile_features(_xgrammar PRIVATE cxx_std_14)

if (CALL_FROM_SETUP_PY)
  string(REPLACE "." ";" _ver ${CMAKE_CUDA_COMPILER_VERSION})
  list(GET _ver 0 CUDA_MAJOR)

  if(CUDA_MAJOR GREATER_EQUAL "13")
    set(_INSTALL_CUDA_RPATH
        "\$ORIGIN"
        "\$ORIGIN/../../nvidia/nccl/lib/"
        "\$ORIGIN/../../nvidia/cu${CUDA_MAJOR}/lib/"
    )
  else()
    set(_INSTALL_CUDA_RPATH
        "\$ORIGIN"
        "\$ORIGIN/../../nvidia/nccl/lib/"
        "\$ORIGIN/../../nvidia/cuda_runtime/lib/"
        "\$ORIGIN/../../nvidia/cublas/lib/"
        "\$ORIGIN/../../nvidia/curand/lib/"
    )
  endif()
  set_target_properties(${PROJECT_NAME} PROPERTIES
      BUILD_RPATH "\$ORIGIN"
      INSTALL_RPATH "${_INSTALL_CUDA_RPATH}"
  )
endif ()


if (TRUE)
    execute_process(
      COMMAND python -c "import importlib.util; print(importlib.util.find_spec('torch').submodule_search_locations[0])"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE torch_path
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT (result EQUAL 0) OR (torch_path STREQUAL ""))
        message(FATAL_ERROR "Can't find pytorch")
    endif()
    find_package(Torch REQUIRED HINTS ${torch_path}/share/cmake/Torch)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${torch_path}/lib)

    pybind11_add_module(_moe test_moe.cc)
    target_link_libraries(_moe PRIVATE LlamaTritonBackend ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
endif()
