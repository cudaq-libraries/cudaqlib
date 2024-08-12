from .transformer import Transformer
import torch
import lightning as L
from abc import ABC, abstractmethod
import math, os, json, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import time
from ml_collections import ConfigDict
import cudaq 

torch.set_float32_matmul_precision('high')


def key(distance):
    v = str(distance).replace(".", "_")
    return v


def pretrain_file(cfg):
    return f"{cfg.save_dir}{cfg.name}_{cfg.seed}_checkpoint_pretrain.ckpt"


def train_file(cfg):
    return f"{cfg.save_dir}{cfg.name}_{cfg.seed}.txt"


def trajectory_file(cfg, distance):
    return f"{cfg.save_dir}{cfg.name}_trajectory_{distance}.ckpt"


def image_file(cfg, errors):
    suffix = ""
    if errors is not None:
        suffix = "-detail"
    return f"{cfg.save_dir}result-{cfg.name}{suffix}.pdf"


def ground_state_file(cfg):
    return f"{cfg.save_dir}gs_{cfg.name}.txt"


def random_file(cfg, seed):
    return f'{cfg.save_dir}{cfg.name}_random_{seed}.txt'


class TemperatureScheduler(ABC):

    @abstractmethod
    def get(self, iter):
        pass


class DefaultScheduler(TemperatureScheduler):

    def __init__(self, start, delta) -> None:
        self.start = start
        self.delta = delta

    def get(self, iter):
        return self.start + iter * self.delta


class CosineScheduler(TemperatureScheduler):

    def __init__(self, minimum, maximum, frequency) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.frequency = frequency

    def get(self, iter):
        return (self.maximum + self.minimum) / 2 - (
            self.maximum - self.minimum) / 2 * math.cos(
                2 * math.pi * iter / self.frequency)


class TrajectoryData:

    def __init__(self, iter_num, loss, indices, energies):
        self.iter_num = iter_num
        self.loss = loss
        self.indices = indices
        self.energies = energies

    def to_json(self):
        map = {
            "iter": self.iter_num,
            "loss": self.loss,
            "indices": self.indices,
            "energies": self.energies
        }
        return json.dumps(map)

    @classmethod
    def from_json(self, string):
        if string.startswith('"'):
            string = string[1:len(string) - 1]
            string = string.replace("\\", "")
        map = json.loads(string)
        return TrajectoryData(map["iter"], map["loss"], map["indices"],
                              map["energies"])


class EnergyDataset(Dataset):

    def __init__(self, file_paths, threshold=sys.maxsize):
        tensor_x = []
        tensor_y = []
        self.min_energy = sys.maxsize
        self.min_indices = None
        for file_path in file_paths:
            with open(file_path) as f:
                for l in f.readlines():
                    data = TrajectoryData.from_json(l.rstrip())
                    for indices, energy in zip(data.indices, data.energies):
                        if threshold < energy:
                            continue
                        if self.min_energy > energy:
                            self.min_energy = energy
                            self.min_indices = indices
                        result = [0]
                        result.extend(indices)
                        tensor_x.append(result)
                        tensor_y.append(energy)
        self.tensors = (torch.tensor(tensor_x, dtype=torch.int64),
                        torch.tensor(tensor_y, dtype=torch.float))

    def __getitem__(self, index):
        result = self.tensors[0][index], self.tensors[1][index]
        return result

    def __len__(self):
        return self.tensors[0].size(0)


class FileMonitor:

    def __init__(self):
        self.lines = []

    def record(self, iter_num, loss, energies, indices):
        energies = energies.cpu().numpy().tolist()
        indices = indices.cpu().numpy().tolist()
        data = TrajectoryData(iter_num, loss.item(), indices, energies)
        self.lines.append(data.to_json())

    def save(self, path):
        with open(path, 'w') as f:
            for l in self.lines:
                f.write(f"{l}\n")


def get_default_config():
    cfg = ConfigDict()
    cfg.verbose = False
    cfg.num_samples = 5  # akin to batch size
    cfg.max_iters = 100
    cfg.ngates = 20
    cfg.seed = 3047
    cfg.lr = 5e-7
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 5.0
    cfg.del_temperature = 0.05
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.check_points = {}
    cfg.dry = False
    cfg.small = False
    cfg.cache = True
    cfg.save_dir = "./output/"
    return cfg


def __internal_run_gqe(temperature_scheduler: TemperatureScheduler,
                       cfg: ConfigDict, model, pool, optimizer):
    fabric = L.Fabric(accelerator="auto", devices=1)  # 1 device for now
    fabric.seed_everything(cfg.seed)
    fabric.launch()
    monitor = FileMonitor()
    model, optimizer = fabric.setup(model, optimizer)
    model.mark_forward_method('train_step')  #model.train_step)
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total trainable params: {pytorch_total_params / 1e6:.2f}M")
    min_energy = sys.maxsize
    min_indices = None
    for epoch in range(cfg.max_iters):
        optimizer.zero_grad()
        l = None
        start = time.time()
        loss, energies, indices, log_values = model.train_step(pool)
        print('epoch', epoch, 'model.train_step time:',
              time.time() - start, torch.min(energies))
        if l is None:
            l = loss
        else:
            l = l + loss
        monitor.record(epoch, loss, energies, indices)
        for e, indices in zip(energies, indices):
            energy = e.item()
            if energy < min_energy:
                min_energy = e.item()
                min_indices = indices
        log_values[f"min_energy at"] = min_energy
        log_values[f"temperature at"] = model.temperature
        fabric.log_dict(log_values, step=epoch)
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
        optimizer.step()
        if temperature_scheduler is not None:
            model.temperature = temperature_scheduler.get(epoch)
        else:
            model.temperature += cfg.del_temperature
    model.set_cost(None)
    # state = {"model": model, "optimizer": optimizer, "hparams": model.hparams}
    # fabric.save(cfg.save_dir + f"checkpoint_{distance}.ckpt", state)
    # if cfg.save_data:
    # monitor.save(trajectory_file(cfg, distance))
    min_indices = min_indices.cpu().numpy().tolist()
    # return min_energy, indices
    fabric.log('circuit', json.dumps(min_indices))
    return min_energy, min_indices

def gqe(cost, pool, config=None, **kwargs):
    cfg = get_default_config() 
    if config == None: 
        [setattr(cfg, a, kwargs[a]) for a in dir(cfg) if not a.startswith('_') and a in kwargs]
    else:
        cfg = config

    # Don't let someone override the vocab_size
    cfg.vocab_size = len(pool)
    cudaqTarget = cudaq.get_target() 
    numQPUs = cudaqTarget.num_qpus() 
    model = Transformer(
        cfg, cost, loss='exp', numQPUs=numQPUs) if 'model' not in kwargs else kwargs['model']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr) if 'optimizer' not in kwargs else kwargs['optimizer']
    return __internal_run_gqe(None, cfg, model, pool, optimizer)
