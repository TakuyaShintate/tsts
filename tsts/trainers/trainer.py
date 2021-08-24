from typing import List, Tuple

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from tsts.cfg import CfgNode as CN
from tsts.core import TRAINERS
from tsts.losses import Loss
from tsts.metrics import Metric
from tsts.optimizers import Optimizer
from tsts.scalers import Scaler

__all__ = ["SupervisedTrainer", "Trainer"]


class Trainer(object):
    def __init__(
        self,
        model: Module,
        losses: List[Loss],
        weight_per_loss: List[float],
        metrics: List[Metric],
        optimizer: Optimizer,
        train_dataloaders: List[DataLoader],
        valid_dataloaders: List[DataLoader],
        max_grad_norm: float,
        device: str,
        scaler: Scaler,
    ) -> None:
        self.model = model
        self.losses = losses
        self.weight_per_loss = weight_per_loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.train_dataloaders = train_dataloaders
        self.valid_dataloaders = valid_dataloaders
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.scaler = scaler

    def on_train(self) -> List[float]:
        raise NotImplementedError

    def on_val(self) -> List[float]:
        raise NotImplementedError

    def step(self) -> Tuple[List[float], List[float]]:
        ave_loss_vs = self.on_train()
        ave_scores = self.on_val()
        return (ave_loss_vs, ave_scores)


@TRAINERS.register()
class SupervisedTrainer(Trainer):
    @classmethod
    def from_cfg(
        cls,
        model: Module,
        losses: List[Loss],
        metrics: List[Metric],
        optimizer: Optimizer,
        train_dataloaders: List[DataLoader],
        valid_dataloaders: List[DataLoader],
        scaler: Scaler,
        cfg: CN,
    ) -> "SupervisedTrainer":
        weight_per_loss = cfg.LOSSES.WEIGHT_PER_LOSS
        max_grad_norm = cfg.TRAINER.MAX_GRAD_NORM
        device = cfg.DEVICE
        trainer = cls(
            model,
            losses,
            weight_per_loss,
            metrics,
            optimizer,
            train_dataloaders,
            valid_dataloaders,
            max_grad_norm,
            device,
            scaler,
        )
        return trainer

    def on_train(self) -> List[float]:
        self.model.train()
        ave_loss_vs = [0.0 for _ in range(len(self.losses))]
        num_dataloaders = len(self.train_dataloaders)
        num_samples = 0.0
        for i in range(num_dataloaders):
            for (X, y, X_mask, y_mask) in tqdm(self.train_dataloaders[i]):
                X = X.to(self.device)
                y = y.to(self.device)
                X_mask = X_mask.to(self.device)
                y_mask = y_mask.to(self.device)
                Z = self.model(X, X_mask)
                self.optimizer.zero_grad()
                device = Z.device
                total_loss_v = torch.tensor(
                    0.0,
                    dtype=torch.float32,
                    device=device,
                )
                for (i, loss) in enumerate(self.losses):
                    weight = self.weight_per_loss[i]
                    loss_v = loss(Z, y, y_mask)
                    total_loss_v += weight * loss_v
                    ave_loss_vs[i] += loss_v.item()
                total_loss_v.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                self.optimizer.step()
            num_samples += len(self.train_dataloaders[i])
        for i in range(len(self.losses)):
            ave_loss_vs[i] /= num_samples
        return ave_loss_vs

    def on_val(self) -> List[float]:
        """Evaluate model on validation dataset.

        Notes
        -----
        Currently, it evaluates model per instance (not whole time series) and averages the score.
        Evaluation on whole time series is the future work.

        Returns
        -------
        List[float]
            List of averaged scores
        """
        self.model.eval()
        num_dataloaders = len(self.valid_dataloaders)
        for i in range(num_dataloaders):
            dataloader = self.valid_dataloaders[i]
            for (X, y, X_mask, y_mask) in tqdm(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                X_mask = X_mask.to(self.device)
                y_mask = y_mask.to(self.device)
                with torch.no_grad():
                    Z = self.model(X, X_mask)
                Z = self.scaler.inv_transform(Z)
                y = self.scaler.inv_transform(y)
                for metric in self.metrics:
                    metric.update(Z, y, y_mask)
        ave_scores = []
        for metric in self.metrics:
            ave_score = metric()
            ave_scores.append(ave_score)
        return ave_scores
