import torch.nn as nn
from copy import deepcopy
import torch

class EMA(nn.Module):
    # code partially borrowed from ema-pytorch
    def __init__(self, model, beta, update_every, update_after_step):
        super().__init__()
        self.model = model
        self.ema_model = deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.detach_()

        self.decay = beta
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.register_buffer('step', torch.tensor(0))

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)

    def update(self):
        step = self.step.item()
        self.step += 1

        should_update = step % self.update_every == 0
        if should_update and step <= self.update_after_step:
            self.ema_model.load_state_dict(self.model.state_dict())
            return

        if should_update:
            self.update_model_average(self.ema_model, self.model)
        return

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)