# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-gse)

add_library(${LIBRARY_NAME} INTERFACE)
target_include_directories(${LIBRARY_NAME} INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> 
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(${LIBRARY_NAME} INTERFACE cudaq::cudaq)

install(
    TARGETS ${LIBRARY_NAME}
    EXPORT ${LIBRARY_NAME}Targets
    DESTINATION lib)
    
install(
    EXPORT ${LIBRARY_NAME}Targets
    FILE ${LIBRARY_NAME}Targets.cmake
    NAMESPACE cudaq::
    DESTINATION lib/cmake/gse)

install(FILES "${CMAKE_SOURCE_DIR}/cmake/${LIBRARY_NAME}Config.cmake"
        DESTINATION lib/cmake/gse)