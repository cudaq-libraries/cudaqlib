# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include("${CMAKE_CURRENT_LIST_DIR}/cudaq-kernelsTargets.cmake")

get_filename_component(PARENT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
get_filename_component(CUDAQLIB_LIBRARY_DIR ${PARENT_DIRECTORY} DIRECTORY)
get_filename_component(CUDAQLIB_INSTALL_DIR ${CUDAQ_LIBRARY_DIR} DIRECTORY)