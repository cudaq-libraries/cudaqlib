# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
install (PROGRAMS cudaq-pyscf.py DESTINATION bin/)
file (COPY cudaq-pyscf.py DESTINATION ${CMAKE_BINARY_DIR}/bin)

