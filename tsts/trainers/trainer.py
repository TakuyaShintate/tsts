import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from tsts.cfg import CfgNode as CN
from tsts.core import TRAINERS
from tsts.optimizers import Optimizer

__all__ = ["SupervisedTrainer", "Trainer"]


class Trainer(object):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        self.model = model
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
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        cfg: CN,
    ) -> "SupervisedTrainer":
        trainer = cls(
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
        )
        return trainer

    def step(self) -> float:
        self.model.train()
        total_loss = 0.0
        for (X, y) in tqdm(self.train_dataloader):
            Z = self.model(X)
            loss = F.mse_loss(Z, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        self.model.eval()
        total_score = 0.0
        for (X, y) in tqdm(self.val_dataloader):
            Z = self.model(X)
            score = (100 * torch.abs((Z - y) / y)).sum()
            total_score += score.item()
        print(total_score)
        print(total_loss)
        return total_loss
