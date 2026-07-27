"""Learning-rate schedules."""

import torch


def cosine_scheduler(optimizer: torch.optim.Optimizer, epochs: int) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

