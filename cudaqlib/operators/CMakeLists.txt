# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-operators)

add_library(${LIBRARY_NAME} SHARED 
  # Base implementation files
  chemistry/molecule.cpp
  fermion/fermion_op.cpp
  # fermion transformations
  fermion/transformations/jordan_wigner.cpp
  # chemistry package drivers
  chemistry/drivers/pyscf/ExternalPySCFDriver.cpp
  )


target_link_libraries(${LIBRARY_NAME} PUBLIC cudaq::cudaq)
target_include_directories(${LIBRARY_NAME} 
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> 
    $<BUILD_INTERFACE:${CUDAQ_INCLUDE_DIR}> 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/json>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/cppitertools>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/xtl/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/xtensor/include>
  PUBLIC
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

install(
    TARGETS ${LIBRARY_NAME}
    EXPORT ${LIBRARY_NAME}Targets
    DESTINATION lib)
    
install(
    EXPORT ${LIBRARY_NAME}Targets
    FILE ${LIBRARY_NAME}Targets.cmake
    NAMESPACE cudaq::
    DESTINATION lib/cmake/operators)

install(FILES "${CMAKE_SOURCE_DIR}/cmake/${LIBRARY_NAME}Config.cmake"
        DESTINATION lib/cmake/operators)