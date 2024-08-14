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

## Use
GQE class usage: `gqe(cost, pool, config=None, **kwargs)` can take a `config` object and\or additional `**kwargs`

The `config` object is of type `ConfigDict` imported from `ml_collections`.  

Example usage:
```
from ml_collections import ConfigDict
cfg = ConfigDict()
cfg.seed = 3047
```

The available config options are provided in the table below:

| **Parameter**          | **Default Value**   | **Description**   |
|------------------------|---------------------|-------------------|
| `cfg.num_samples`      | `5`                 | Number of circuits to generate during each epoch/batch |
| `cfg.max_iters`        | `100`               | Number of epochs to run |
| `cfg.ngates`           | `20`                | Number of gates that make up each generated circuit |
| `cfg.seed`             | `3047`              | Random seed |
| `cfg.lr`               | `5e-7`              | Learning rate used by the optimizer |
| `cfg.energy_offset`    | `0.0`               | Offset added to expectation value of the cirucit (Energy) for numerical stability, see [K. Nakaji et al. (2024)](https://arxiv.org/abs/2401.09253) Sec. 3 |
| `cfg.grad_norm_clip`   | `1.0`               | max_norm for clipping gradients, see [Ligthning docs](https://lightning.ai/docs/fabric/stable/api/fabric_methods.html#clip-gradients) |
| `cfg.temperature`      | `5.0`               | Starting inverse temperature $\beta$ as described in [K. Nakaji et al. (2024)](https://arxiv.org/abs/2401.09253) Sec. 2.2 |
| `cfg.del_temperature`  | `0.05`              | Temperature increase after each epoch |
| `cfg.resid_pdrop`      | `0.0`               | The dropout probability for all fully connected layers in the embeddings, encoder, and pooler, see [GPT2Config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py) |
| `cfg.embd_pdrop`       | `0.0`               | The dropout ratio for the embeddings, see [GPT2Config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py) |
| `cfg.attn_pdrop`       | `0.0`               | The dropout ratio for the attention, see [GPT2Config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py) |
| `cfg.small`            | `False`             | True: Uses a small transfomer (6 hidden layers and 6 attention heads as opposed to the default transformer of 12 of each) |
| `cfg.save_dir`         | `"./output/"`       | Path to save files |

The `**kwargs` takes the following args:
| **arg**                | **Description**   |
|------------------------|-------------------|
| `max_iters`            | Overrides cfg.max_iters for total number of epochs to run |
| `energy_offset`        | Overrides cfg.energy_offset for offset to add to expectation value |
| `model`                | Can pass in an already constructed transformer |
| `optimizer`            | Can pass in an already constructed optimizer |
