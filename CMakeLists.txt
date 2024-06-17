# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

# Policies
# ==============================================================================
if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# Project setup
# ==============================================================================
project(cudaqlib LANGUAGES CXX C Fortran)

# Going the extra mile to prevent the user from playing tricks with symlinks.
get_filename_component(REAL_PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}" REALPATH)
get_filename_component(REAL_PROJECT_BINARY_DIR "${PROJECT_BINARY_DIR}" REALPATH)

if("${REAL_PROJECT_SOURCE_DIR}" STREQUAL "${REAL_PROJECT_BINARY_DIR}")
  message(FATAL_ERROR
    "In-source builds are not permitted. You must run cmake in a separeted "
    "directory, e.g.:\n"
    "    mkdir build && cd build && cmake ..\n"
    "NOTE: Remeber to clean up the source tree by deleting the files already "
    "created by CMake, e.g, CMakeCache.txt and cmake.check_cache")
endif()

# Project globals
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CUDAQLIB_TPLS_DIR ${CMAKE_SOURCE_DIR}/tpls)

# Set warnings as errors by default. 
# Individual targets and the command line invocation can override this behavior.
if(NOT DEFINED CMAKE_COMPILE_WARNING_AS_ERROR)
  set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
endif()

# Generate a CompilationDatabase (compile_commands.json file) for our build,
# for use by clang_complete, YouCompleteMe, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} -c submodule.tpls/llvm.update=none submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "Cloning git submodules failed with ${GIT_SUBMOD_RESULT}, please checkout submodules manually")
    endif()
  endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/tpls/googletest/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# Options
# ==============================================================================
option(CUDAQLIB_BUILD_TESTS "Build cudaqlib tests" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

# Find CUDA Quantum
# FIXME for now, we only use library mode
set(CUDAQ_LIBRARY_MODE ON)
find_package(CUDAQ REQUIRED)

# We should always find OpenMP
find_package(OpenMP REQUIRED) 

set(BLA_STATIC ON)
find_package(BLAS REQUIRED)

# Check for CUDA Support
# ==============================================================================
include(CheckLanguage)
check_language(CUDA)
set(CUDA_FOUND FALSE)
# Generate -gencode arch=compute_XX,code=sm_XX for list of supported
# arch values.
# List should be sorted in increasing order.
function(CUDA_get_gencode_args out_args_string arch_values)
  # allow the user to pass the list like a normal variable
  set(arch_list ${arch_values} ${ARGN})
  set(out "")
  foreach(arch IN LISTS arch_list)
    set(out "${out} -gencode arch=compute_${arch},code=sm_${arch}")
  endforeach(arch)

  # Repeat the last one as to ensure the generation of PTX for most
  # recent virtual architecture for forward compatibility
  list(GET arch_list -1 last_arch)
  set(out "${out} -gencode arch=compute_${last_arch},code=compute_${last_arch}")
  set(${out_args_string} ${out} PARENT_SCOPE)
endfunction()

if(CMAKE_CUDA_COMPILER)
  if (NOT CUDA_TARGET_ARCHS)
    # Volta, Ampere, Hopper
    set(CUDA_TARGET_ARCHS  "70;80;90")
  endif()
  CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_TARGET_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -shared -std=c++17 ${CUDA_gencode_flags} --compiler-options -fPIC")

  enable_language(CUDA)
  set(CUDA_FOUND TRUE)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  message(STATUS "Cuda language found.")
endif()

add_subdirectory(cmake)

# Directory setup
# ==============================================================================
add_subdirectory(tpls/pybind11)
add_subdirectory(tpls)
add_subdirectory(cudaq-x)
add_subdirectory(tools)
add_subdirectory(python)

# ==============================================================================
if(CUDAQLIB_BUILD_TESTS)
  set(INSTALL_GTEST OFF)
  add_subdirectory(tpls/googletest)
  # Bug in GCC 12 leads to spurious warnings (-Wrestrict)
  # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
  if (CMAKE_COMPILER_IS_GNUCXX 
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0 
    AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
    target_compile_options(gtest PUBLIC --param=evrp-mode=legacy)
  endif()
  if (CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    target_compile_options(gtest PUBLIC -Wno-covered-switch-default)
  endif()
  include(GoogleTest)
  include(CTest)
  add_subdirectory(tests)
endif()


install (DIRECTORY cudaqlib DESTINATION include 
            FILES_MATCHING PATTERN "*.h"
            PATTERN "liblbfgs" EXCLUDE
            PATTERN "prima/c" EXCLUDE
            PATTERN "prima/fortran" EXCLUDE)