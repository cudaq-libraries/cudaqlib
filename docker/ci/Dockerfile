# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

FROM ghcr.io/cudaq-libraries/cudaqlib-base:latest

ARG repo=cudaq-libraries/cudaqlib 
ARG branch=main 

RUN git clone -b ${branch} https://github.com/${repo} cudaqlib && cd cudaqlib && mkdir build && cd build \
   && CC=gcc-12 CXX=g++-12 cmake -G Ninja .. -DCUDAQ_DIR=$HOME/.cudaq/lib/cmake/cudaq -DCUDAQ_TARGET=qpp-cpu \
   && ninja install \
   && python3 -m pip install transformers mpi4py torch lightning ml_collections
ENV OMPI_MCA_pml=
ENV PYTHONPATH=$PYTHONPATH:/cudaqlib/build/python/cudaqlib:$HOME/.cudaq
