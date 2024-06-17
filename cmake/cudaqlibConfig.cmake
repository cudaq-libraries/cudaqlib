# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
include("${CMAKE_CURRENT_LIST_DIR}/../operators/cudaq-operatorsTargets.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/../optim/cudaq_lbfgsTargets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../optim/cudaq_primaTargets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../optim/cudaq-optimTargets.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/../gse/cudaq-gseTargets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../kernels/cudaq-kernelsTargets.cmake")

get_filename_component(PARENT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
get_filename_component(CUDAQLIB_LIBRARY_DIR ${PARENT_DIRECTORY} DIRECTORY)
get_filename_component(CUDAQLIB_INSTALL_DIR ${CUDAQ_LIBRARY_DIR} DIRECTORY)

add_compile_options(-Wno-attributes)
