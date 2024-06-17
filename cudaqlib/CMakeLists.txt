# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set (CUDAQLIB_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(operators)
add_subdirectory(optim)
add_subdirectory(gse)
add_subdirectory(kernels)
