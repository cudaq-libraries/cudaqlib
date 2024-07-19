from abc import abstractmethod, ABC
import torch


class LogitMatchingLoss(ABC):

    @abstractmethod
    def compute(self, energies, logits_tensor, temperature, log_values):
        pass


class ExpLogitMatching(LogitMatchingLoss):

    def __init__(self, energy_offset, label) -> None:
        self._label = label
        self.energy_offset = energy_offset
        self.loss_fn = torch.nn.MSELoss()

    def compute(self, energies, logits_tensor, temperature, log_values):
        mean_logits = torch.mean(logits_tensor, 1)
        log_values[f"mean_logits at {self._label}"] = torch.mean(
            mean_logits - self.energy_offset)
        log_values[f"mean energy at {self._label}"] = torch.mean(energies)
        mean_logits = torch.mean(logits_tensor, 1)
        device = mean_logits.device
        return self.loss_fn(
            torch.exp(-mean_logits),
            torch.exp(-energies.to(device) - self.energy_offset))


class GFlowLogitMatching(LogitMatchingLoss):

    def __init__(self, energy_offset, device, label, nn: torch.nn) -> None:
        self._label = label
        self.loss_fn = torch.nn.MSELoss()
        self.energy_offset = energy_offset
        self.normalization = 10**-5
        self.param = torch.nn.Parameter(torch.tensor([0.0]).to(device))
        nn.register_parameter(name="energy_offset", param=self.param)

    def compute(self, energies, logits_tensor, temperature, log_values):
        mean_logits = torch.mean(logits_tensor, 1)
        energy_offset = self.energy_offset + self.param / self.normalization
        log_values[f"energy_offset at {self._label}"] = energy_offset
        log_values[f"mean_logits at {self._label}"] = torch.mean(mean_logits -
                                                                 energy_offset)
        log_values[f"mean energy at {self._label}"] = torch.mean(energies)
        mean_logits = torch.mean(logits_tensor, 1)
        device = mean_logits.device
        loss = self.loss_fn(
            torch.exp(-mean_logits),
            torch.exp(-(energies.to(device) + energy_offset.to(device))))
        return loss
