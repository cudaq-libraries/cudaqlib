# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_compile_options(-Wno-attributes)
add_executable(test_optim main.cpp optim/OptimTester.cpp)

# cudaq-default-cpu is an interface target composed
# of cudaq, the default platform, and the cpu-based simulator
target_link_libraries(test_optim
  PRIVATE 
  cudaq-optim
  gtest_main)
gtest_discover_tests(test_optim)

add_executable(test_gse main.cpp gse/VQETester.cpp gse/AdaptTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_gse PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(test_gse
  PRIVATE 
  cudaq-optim
  gtest_main)
target_link_libraries(test_gse INTERFACE cudaq-gse)
gtest_discover_tests(test_gse)


add_executable(test_operators main.cpp operators/OperatorsTester.cpp)
target_include_directories(test_operators PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_operators
  PRIVATE 
  cudaq-operators
  gtest_main)
gtest_discover_tests(test_operators)

add_executable(test_kernels main.cpp kernels/UCCSDTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_kernels PRIVATE -Wl,--no-as-needed)
endif()
target_include_directories(test_kernels PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_kernels
  PRIVATE 
  cudaq::cudaq-builder
  cudaq-operators
  cudaq-optim
  gtest_main)
gtest_discover_tests(test_kernels)
