# Welcome to the CUDA-Q Libraries repository

This repository contains a set of prototype application libraries that build off 
of NVIDIA CUDA-Q. These libraries enable the development of hybrid quantum-classical 
application code leveraging state-of-the-art CPUs, GPUs, and QPUs. 

## License

The code in this repository is licensed under [Apache License 2.0](./LICENSE).

Contributing a pull request to this repository requires accepting the
Contributor License Agreement (CLA) declaring that you have the right to, and
actually do, grant us the rights to use your contribution. A CLA-bot will
automatically determine whether you need to provide a CLA and decorate the PR
appropriately. Simply follow the instructions provided by the bot. You will only
need to do this once.

## Build 

This library requires the `cudaqlib` branch at this [fork](https://github.com/amccaskey/cuda-quantum). 

```bash 
export CC=gcc-12
export CXX=g++-12

git clone -b cudaqlib https://github.com/amccaskey/cuda-quantum 
cd cuda-quantum && mkdir build && cd build 
cmake .. -G Ninja \
   -DLLVM_DIR=/opt/llvm/lib/cmake/llvm \
   -DCUDAQ_ENABLE_PYTHON=TRUE -DCMAKE_INSTALL_PREFIX=$HOME/.cudaq 
ninja install 

apt-get install libblas-dev gfortran
git clone https://github.com/cudaq-libraries/cudaqlib
cd cudaqlib && mkdir build && cd build 
cmake .. -G Ninja -DCUDAQ_DIR=$HOME/.cudaq/lib/cmake/cudaq 
ninja 
export PYTHONPATH=$HOME/.cudaq:$PWD/python/cudaqlib
ctest 
```
