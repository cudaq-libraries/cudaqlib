#!/bin/bash 
numGPUS=$(nvidia-smi --list-gpus | wc -l)
echo $numGPUS
CUDA_VISIBLE_DEVICES=$(($OMPI_COMM_WORLD_RANK % $numGPUS)) python3 gqe_h2_async_mpi.py