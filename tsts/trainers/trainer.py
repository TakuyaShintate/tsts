import os
import uuid
from typing import Any, Dict, List

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
        log_dir: str,
        max_grad_norm: float,
        device: str,
    ) -> None:
        self.model = model
        self.losses = losses
        self.weight_per_loss = weight_per_loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.log_dir = log_dir
        self.max_grad_norm = max_grad_norm
        self.device = device
        self._init_internal_state()
        self._init_log_dir()

    def _init_internal_state(self) -> None:
        self.epoch = 0
        self.best_ave_score = float("inf")

    def _init_log_dir(self) -> None:
        os.mkdir(self.log_dir)

    def log(
        self,
        ave_loss_vals: List[float],
        ave_scores: List[float],
    ) -> None:
        # Update model params
        current_ave_score = sum(ave_scores) / len(ave_scores)
        if current_ave_score < self.best_ave_score:
            self.best_ave_score = current_ave_score
            root = os.path.join(self.log_dir, "model.pth")
            torch.save(self.model.state_dict(), root)
        # Add new record to log file
        record: Dict[str, Any] = {
            "epoch": self.epoch,
            "loss": {},
            "metric": {},
        }
        for (i, loss) in enumerate(self.losses):
            loss_name = loss.__class__.__name__
            ave_loss_val = ave_loss_vals[i]
            record["loss"][loss_name] = ave_loss_val
        for (i, metric) in enumerate(self.metrics):
            metric_name = metric.__class__.__name__
            ave_score = ave_scores[i]
            record["metric"][metric_name] = ave_score
        log_file = os.path.join(self.log_dir, "log.txt")
        with open(log_file, "a") as f:
            f.write(str(record) + "\n")

    def on_train(self) -> List[float]:
        raise NotImplementedError

    def on_val(self) -> List[float]:
        raise NotImplementedError

    def step(self) -> None:
        ave_loss_vals = self.on_train()
        ave_scores = self.on_val()
        self.log(ave_loss_vals, ave_scores)
        self.epoch += 1


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
        log_dir = cfg.LOG_DIR
        if log_dir == "auto":
            log_dir = str(uuid.uuid4())
        max_grad_norm = cfg.TRAINER.MAX_GRAD_NORM
        device = cfg.DEVICE
        trainer = cls(
            model,
            losses,
            weight_per_loss,
            metrics,
            optimizer,
            train_dataloader,
            val_dataloader,
            log_dir,
            max_grad_norm,
            device,
        )
        return trainer

    def on_train(self) -> List[float]:
        self.model.train()
        ave_loss_vals = [0.0 for _ in range(len(self.losses))]
        for (X, y, X_mask, y_mask) in tqdm(self.train_dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            X_mask = X_mask.to(self.device)
            y_mask = y_mask.to(self.device)
            Z = self.model(X, X_mask)
            self.optimizer.zero_grad()
            device = Z.device
            total_loss_val = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=device,
            )
            for (i, loss) in enumerate(self.losses):
                weight = self.weight_per_loss[i]
                loss_val = loss(Z, y, y_mask)
                total_loss_val += weight * loss_val
                ave_loss_vals[i] += loss_val.item()
            total_loss_val.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
            self.optimizer.step()
        num_samples = len(self.train_dataloader)
        for i in range(len(self.losses)):
            ave_loss_vals[i] /= num_samples
        return ave_loss_vals

    def on_val(self) -> List[float]:
        self.model.eval()
        for (X, y, X_mask, y_mask) in tqdm(self.val_dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            X_mask = X_mask.to(self.device)
            y_mask = y_mask.to(self.device)
            Z = self.model(X, X_mask)
            for metric in self.metrics:
                metric.update(Z, y, y_mask)
        ave_scores = []
        for metric in self.metrics:
            ave_score = metric()
            ave_scores.append(ave_score)
        return ave_scores
