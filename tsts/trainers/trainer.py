from typing import List

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from tsts.cfg import CfgNode as CN
from tsts.core import TRAINERS
from tsts.losses import Loss
from tsts.optimizers import Optimizer

__all__ = ["SupervisedTrainer", "Trainer"]


class Trainer(object):
    def __init__(
        self,
        model: Module,
        losses: List[Loss],
        weight_per_loss: List[float],
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        self.model = model
        self.losses = losses
        self.weight_per_loss = weight_per_loss
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def step(self) -> float:
        raise NotImplementedError


@TRAINERS.register()
class SupervisedTrainer(Trainer):
    @classmethod
    def from_cfg(
        cls,
        model: Module,
        losses: List[Loss],
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        cfg: CN,
    ) -> "SupervisedTrainer":
        weight_per_loss = cfg.LOSSES.WEIGHT_PER_LOSS
        trainer = cls(
            model,
            losses,
            weight_per_loss,
            optimizer,
            train_dataloader,
            val_dataloader,
        )
        return trainer

    def step(self) -> float:
        self.model.train()
        total_loss_val = 0.0
        for (X, y) in tqdm(self.train_dataloader):
            Z = self.model(X)
            self.optimizer.zero_grad()
            device = Z.device
            loss_val = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=device,
            )
            for (i, loss) in enumerate(self.losses):
                weight = self.weight_per_loss[i]
                loss_val += weight * loss(Z, y)
            loss_val.backward()
            self.optimizer.step()
            total_loss_val += loss_val.item()
        self.model.eval()
        total_score = 0.0
        for (X, y) in tqdm(self.val_dataloader):
            Z = self.model(X)
            score = (100 * torch.abs((Z - y) / y)).sum()
            total_score += score.item()
        print(total_score)
        print(total_loss_val)
        return total_loss_val
