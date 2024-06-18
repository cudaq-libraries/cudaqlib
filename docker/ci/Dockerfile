# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

FROM ghcr.io/cudaq-libraries/cudaqlib-base:latest

RUN apt-get update && apt-get install -y gfortran libblas-dev \
   && git clone https://github.com/cudaq-libraries/cudaqlib && cd cudaqlib && mkdir build && cd build \
   && CC=gcc-12 CXX=g++-12 cmake -G Ninja .. -DCUDAQ_DIR=$HOME/.cudaq/lib/cmake/cudaq \
   && ninja install \
   && python3 -m pip install cppe
ENV PYTHONPATH=$PYTHONPATH:/cudaqlib/build/python/cudaqlib:$HOME/.cudaq