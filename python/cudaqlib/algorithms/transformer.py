import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Config
from lightning import LightningModule
from .loss import ExpLogitMatching, GFlowLogitMatching

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class SmallConfig(GPT2Config):

    def __init__(self, **kwargs):
        super().__init__(n_layer=6, n_head=6, **kwargs)


class Transformer(LightningModule):

    def __init__(self, cfg, cost,  loss="exp"):
        super().__init__()
        self._label = 'label_stand_in'
        self.cfg = cfg
        gpt2cfg = GPT2Config(
            **{k: cfg[k] for k in GPT2Config().to_dict().keys() & cfg.keys()})
        if cfg.small:
            gpt2cfg = SmallConfig(
                **
                {k: cfg[k] for k in GPT2Config().to_dict().keys() & cfg.keys()})
        self.transformer = GPT2LMHeadModel(gpt2cfg).to(get_device())
        self.ngates = cfg.ngates
        self.num_samples = cfg.num_samples
        self.temperature = cfg.temperature
        self.save_hyperparameters()
        self._starting_idx = torch.zeros(self.num_samples,
                                         1,
                                         dtype=torch.int,
                                         device=get_device())
        if loss == "exp":
            self.loss = ExpLogitMatching(cfg.energy_offset, self._label)
            print("ExpMatching")
        else:
            self.loss = GFlowLogitMatching(cfg.energy_offset, get_device(),
                                           self._label, self)
            print("GFlow")
        self._cost = cost

    def generate_logits(self, idx):
        logits = self.transformer(idx)[0]
        return logits

    def set_cost(self, cost):
        self._cost = cost

    def gather(self, idx, logits_base):
        b_size = idx.shape[0]
        return torch.gather(logits_base, 2, idx.reshape(b_size, -1,
                                                        1)).reshape(b_size, -1)

    def computeCost(self, idx_output, pool, **kwargs):
        return torch.tensor([self._cost([pool[i] for i in row]) for row in idx_output],
                            dtype=torch.float)

    def train_step(self, pool, indices=None, energies=None, numQPUs=None, comm=None):
        log_values = {}
        if energies is not None:
            assert indices is not None
            idx_output = indices[:, 1:]
            logits_base = self.generate_logits(idx_output)
        else:
            idx_output, logits_base = self.generate()
            energies = self.computeCost(idx_output, pool, numQPUs=numQPUs, comm=comm)
        logits_tensor = self.gather(idx_output, logits_base)
        allLogits = logits_tensor

        loss = self.loss.compute(energies, allLogits, self.temperature,
                                 log_values)
        log_values[f"loss at {self._label}"] = loss
        return loss, energies, idx_output, log_values

    def generate(self, idx=None, ngates=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if idx is None:
            idx = self._starting_idx.clone()
        condition_length = idx.size(dim=1)
        if ngates is None:
            ngates = self.ngates
        for _ in range(ngates):
            idx_cond = idx
            logits_base = self.generate_logits(idx_cond)
            logits = logits_base[:, -1, :]
            probs = F.softmax(-self.temperature * logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        idx = idx[:, condition_length:]
        return idx, logits_base

    def forward(self):
        return self.train_step()
