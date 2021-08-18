from typing import List

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from tsts.cfg import CfgNode as CN
from tsts.core import TRAINERS
from tsts.losses import Loss
from tsts.metrics import Metric
from tsts.optimizers import Optimizer

__all__ = ["SupervisedTrainer", "Trainer"]


class Trainer(object):
    def __init__(
        self,
        model: Module,
        losses: List[Loss],
        weight_per_loss: List[float],
        metrics: List[Metric],
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        self.model = model
        self.losses = losses
        self.weight_per_loss = weight_per_loss
        self.metrics = metrics
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
        metrics: List[Metric],
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
            metrics,
            optimizer,
            train_dataloader,
            val_dataloader,
        )
        return trainer

    def step(self) -> float:
        self.model.train()
        total_loss_val = 0.0
        for (X, y, X_mask, y_mask) in tqdm(self.train_dataloader):
            Z = self.model(X, X_mask)
            self.optimizer.zero_grad()
            device = Z.device
            loss_val = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=device,
            )
            for (i, loss) in enumerate(self.losses):
                weight = self.weight_per_loss[i]
                loss_val += weight * loss(Z, y, y_mask)
            loss_val.backward()
            self.optimizer.step()
            total_loss_val += loss_val.item()
        self.model.eval()
        for (X, y, X_mask, y_mask) in tqdm(self.val_dataloader):
            Z = self.model(X, X_mask)
            for (i, metric) in enumerate(self.metrics):
                metric.update(Z, y, y_mask)
        ave_scores = []
        for (i, metric) in enumerate(self.metrics):
            ave_score = metric()
            ave_scores.append(ave_score)
        print(ave_scores)
        print(total_loss_val)
        return total_loss_val
