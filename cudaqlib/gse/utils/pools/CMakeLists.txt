# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_compile_options(-Wno-attributes)
set(LIBRARY_NAME cudaq-operator-pools)
add_library(${LIBRARY_NAME} SHARED uccsd_pool.cpp spin_complement_gsd.cpp)
target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)
target_link_libraries(${LIBRARY_NAME} PRIVATE cudaq::cudaq-spin)

install(
    TARGETS ${LIBRARY_NAME}
    EXPORT ${LIBRARY_NAME}Targets
    DESTINATION lib)
    
install(
    EXPORT ${LIBRARY_NAME}Targets
    FILE ${LIBRARY_NAME}Targets.cmake
    NAMESPACE cudaq::
    DESTINATION lib/cmake/gse)