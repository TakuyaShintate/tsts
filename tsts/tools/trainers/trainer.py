from typing import Dict

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

__all__ = ["SupervisedTrainer"]


class Trainer(object):
    def __init__(
        self,
        model: Module,
        optimizer: SGD,
        train_dl: DataLoader,
        val_dl: DataLoader,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.val_dl = val_dl


class SupervisedTrainer(Trainer):
    def step(self) -> float:
        self.model.train()
        total_loss = 0.0
        for (X, y) in tqdm(self.train_dl):
            Z = self.model(X)
            loss = F.mse_loss(Z, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        self.model.eval()
        total_score = 0.0
        for (X, y) in tqdm(self.val_dl):
            Z = self.model(X)
            score = (100 * torch.abs((Z - y) / y)).sum()
            total_score += score.item()
        print(total_score)
        print(total_loss)
        return total_loss
