# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq_prima)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(prima EXCLUDE_FROM_ALL)
add_library(${LIBRARY_NAME} SHARED cobyla.cpp)
target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/prima/c/include>)
target_link_libraries(${LIBRARY_NAME} $<BUILD_INTERFACE:primac>)
# target_link_options(${LIBRARY_NAME})
install(
    TARGETS ${LIBRARY_NAME}
    EXPORT ${LIBRARY_NAME}Targets
    DESTINATION lib)
    
install(
    EXPORT ${LIBRARY_NAME}Targets
    FILE ${LIBRARY_NAME}Targets.cmake
    NAMESPACE cudaq::
    DESTINATION lib/cmake/optim)