# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

FROM ghcr.io/nvidia/cuda-quantum-devdeps:ext-gcc12-main

RUN cd .. && git clone -b cudaqlib https://github.com/amccaskey/cuda-quantum \
  && cd cuda-quantum && mkdir build && cd build \
  && CC=gcc-12 CXX=g++-12 cmake -G Ninja .. -DLLVM_DIR=/opt/llvm/lib/cmake \
     -DCUDAQ_ENABLE_PYTHON=TRUE -DCMAKE_INSTALL_PREFIX=$HOME/.cudaq \
  && ninja install \
  && apt-get update && apt-get install -y gfortran libblas-dev \
  && python3 -m pip install cppe torch lightning ml-collections