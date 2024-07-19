from .transformer import Transformer
import torch
import lightning as L
from abc import ABC, abstractmethod
import math, os, json, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import time

torch.set_float32_matmul_precision('high')


def key(distance):
    v = str(distance).replace(".", "_")
    return v


def plot_trajectory(trajectory_path):
    data = []
    with open(trajectory_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    iterations = []
    losses = []
    avg_energies = []
    min_energies = []
    global_min = 0
    for entry in data:
        iterations.append(entry['iter'])
        losses.append(entry['loss'])
        avg_energies.append(sum(entry['energies']) / len(entry['energies']))
        if min(entry['energies']) < global_min:
            global_min = min(entry['energies'])
        min_energies.append(global_min)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(iterations, losses, marker='o', linestyle='-', color='b')
    plt.title('Loss per Epoch')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(3, 1, 2)
    plt.plot(iterations, avg_energies, marker='o', linestyle='-', color='g')
    plt.title('Average Energy per Epoch')
    plt.xlabel('Iteration')
    plt.ylabel('Average Energy')

    plt.subplot(3, 1, 3)
    plt.plot(iterations, min_energies, marker='o', linestyle='-', color='r')
    plt.title('Minimum Energy per Epoch')
    plt.xlabel('Iteration')
    plt.ylabel('Minimum Energy')

    plt.tight_layout()
    plt.savefig('trajectory_plot.png')


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


class GPTQETaskBase():

    def __init__(self,
                 temperature_scheduler: TemperatureScheduler = None) -> None:
        self.temperature_scheduler = temperature_scheduler

    # def train(self, cfg, model, optimizer, numQPUs=1):
    #     self.numQPUs = numQPUs

    #     # Not yet...
    #     # from mpi4py import MPI

    #     # Initialize MPI
    #     # self.mpiComm = MPI.COMM_WORLD
    #     # self.mpiRank = self.mpiComm.Get_rank()
    #     # self.mpiSize = self.mpiComm.Get_size()
    #     fabric = L.Fabric(accelerator="auto", devices=1)
    #     fabric.seed_everything(cfg.seed)
    #     fabric.launch()

    #     min_indices_dict = {}
    #     distances = cfg.distances
    #     filename = train_file(cfg)
    #     m = {}
    #     if os.path.exists(filename):
    #         with open(filename) as f:
    #             for l in f.readlines():
    #                 items = l.rstrip().split('\t')
    #                 if len(items) != 2:
    #                     continue
    #                 distance, energy = items
    #                 distance = float(distance)
    #                 energy = float(energy)
    #                 m[distance] = energy
    #     with open(filename, 'w') as f:
    #         for distance in distances:
    #             print("distance:", distance)
    #             if cfg.cache and distance in m:
    #                 min_energy = m[distance]
    #                 print("already computed, skipped", distance)
    #             else:
    #                 indices, min_energy = self.do_train(cfg, distance, fabric, model, optimizer)
    #                 min_indices_dict[str(distance)] = indices
    #             f.write(f"{distance}\t{min_energy}\n")
    #     fabric.log('circuit', json.dumps(min_indices_dict))
    #     # if (self.mpiComm):
    #         # MPI.Finalize()
    #     return min_indices_dict

    def train(self, cfg, model, optimizer):
        fabric = L.Fabric(accelerator="auto")
        fabric.seed_everything(cfg.seed)
        fabric.launch()

        min_indices, energy = self.do_train(cfg, fabric, model,
                                            optimizer)
        fabric.log('circuit', json.dumps(min_indices))
        return min_indices, energy

    def do_train(self, cfg, fabric, model, optimizer):
        print(cfg)
        monitor = FileMonitor()
        model, optimizer = fabric.setup(model, optimizer)
        model.mark_forward_method(model.train_step)
        
        # if key(distance) in cfg.check_points:
        #     print("loaded from the checkpoint")
        #     cp = fabric.load(cfg.check_points[key(distance)])
        #     model.load_state_dict(cp["model"])
        #     optimizer.load_state_dict(cp["optimizer"])
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total trainable params: {pytorch_total_params / 1e6:.2f}M")
        min_energy = sys.maxsize
        min_indices = None
        for epoch in range(cfg.max_iters):
            optimizer.zero_grad()
            l = None
            start = time.time()
            loss, energies, indices, log_values = model.train_step()
                # numQPUs=self.numQPUs, comm=self.mpiComm)
            print('epoch', epoch, 'model.train_step time:', time.time() - start, torch.min(energies))
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
            if cfg.verbose:
                print(f"energies: {energies}")
                print(f"temperature: {model.temperature}")
            fabric.log_dict(log_values, step=epoch)
            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
            optimizer.step()
            # scheduler.step()
            if self.temperature_scheduler is not None:
                model.temperature = self.temperature_scheduler.get(epoch)
            else:
                model.temperature += cfg.del_temperature
        model.set_cost(None)
        # state = {"model": model, "optimizer": optimizer, "hparams": model.hparams}
        # fabric.save(cfg.save_dir + f"checkpoint_{distance}.ckpt", state)
        # if cfg.save_data:
        #     monitor.save(trajectory_file(cfg, distance))
        #     if cfg.plot:
        #         plot_trajectory(
        #             str(cfg.save_dir) + str(cfg.name) + "_trajectory_" +
        #             str(distance) + ".ckpt")
        indices = min_indices.cpu().numpy().tolist()
        return min_energy, indices


def get_default_chemistry_configs():
    from ml_collections import ConfigDict
    cfg = ConfigDict()
    cfg.verbose = False
    cfg.save_data = False
    cfg.print_exact = True
    cfg.name = "gptqe"
    cfg.warmup_steps = 100
    cfg.num_samples = 5  # akin to batch size
    cfg.max_iters = 100
    cfg.nqubit = 4
    cfg.ngates = 20
    cfg.nshot = 0
    cfg.seed = 3047
    cfg.distances = [0.5, 0.6, 0.7, 0.7414, 0.8, 0.9, 1.0, 1.5,
                     2.0]  # choices of the distance between two atoms
    cfg.time_pool = [
        0.003125, -0.003125, 0.00625, -0.00625, 0.0125, -0.0125, 0.025, -0.025,
        0.05, -0.05, 0.1, -0.1
    ]
    cfg.time_factor = 1.0
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


def gqe(cost, **kwargs):
    cfg = get_default_chemistry_configs()
    cfg.vocab_size = 12
    distance = .7474
    model = Transformer(cfg, distance, cost, loss='exp')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gqe = GPTQETaskBase()
    return gqe.train(cfg, model, optimizer)#, numQPUs=1)
