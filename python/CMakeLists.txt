# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME _pycudaqlib)
if (NOT SKBUILD)
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ORIGIN/../../lib")
else()
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")
endif()

find_package(Python COMPONENTS Interpreter)

if (NOT Python_FOUND)
  message(STATUS "Python interpreter not found. Skipping Python API build.")
  return()
endif()

if (${Python_VERSION} VERSION_LESS 3.0.0)
  message(STATUS "Found Python version ${Python_VERSION}. Version must be greater than 3.0.0, skipping Python API build.")
  return()
endif()

message(STATUS "Found Python version ${Python_VERSION}. Building CUDA Quantum Python API with ${Python_INCLUDE_DIRS}")

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

pybind11_add_module(_pycudaqlib
                     bindings/cudaqlib.cpp
                     bindings/gse/py_gse.cpp
                     bindings/optim/py_optimizer.cpp
                     bindings/operators/py_operators.cpp)

target_compile_options(_pycudaqlib PRIVATE -Wno-attributes)                     
target_include_directories(_pycudaqlib PRIVATE .)
target_link_libraries(_pycudaqlib 
    PRIVATE 
      cudaq::cudaq
      cudaq-operators
      cudaq-optim
      cudaq-qec)

if (NOT SKBUILD)
  install(DIRECTORY cudaqlib DESTINATION python/cudaqlib)
  install(TARGETS _pycudaqlib DESTINATION python/cudaqlib/cudaqlib)
  # Also move these to the build directory for testing.
  file (COPY cudaqlib DESTINATION ${CMAKE_BINARY_DIR}/python/cudaqlib)
  add_custom_command(TARGET _pycudaqlib POST_BUILD 
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_pycudaqlib> 
              ${CMAKE_BINARY_DIR}/python/cudaqlib/cudaqlib)
else()
  install(TARGETS _pycudaqlib LIBRARY DESTINATION cudaqlib)
endif()

