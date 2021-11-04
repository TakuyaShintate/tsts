from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from tsts.cfg import CfgNode as CN
from tsts.core import TRAINERS
from tsts.losses import Loss
from tsts.metrics import Metric
from tsts.optimizers import Optimizer
from tsts.schedulers import Scheduler

__all__ = ["SupervisedTrainer", "Trainer"]


class Trainer(object):
    def __init__(
        self,
        model: Module,
        local_scaler: Module,
        losses: List[Loss],
        weight_per_loss: List[float],
        metrics: List[Metric],
        optimizer: Optimizer,
        scheduler: Scheduler,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        max_grad_norm: float,
        device: str,
        denorm: bool,
    ) -> None:
        self.model = model
        self.local_scaler = local_scaler
        self.losses = losses
        self.weight_per_loss = weight_per_loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.denorm = denorm

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
        local_scaler: Module,
        losses: List[Loss],
        metrics: List[Metric],
        optimizer: Optimizer,
        scheduler: Scheduler,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        cfg: CN,
    ) -> "SupervisedTrainer":
        weight_per_loss = cfg.LOSSES.WEIGHT_PER_LOSS
        max_grad_norm = cfg.TRAINER.MAX_GRAD_NORM
        device = cfg.DEVICE
        denorm = cfg.TRAINER.DENORM
        trainer = cls(
            model,
            local_scaler,
            losses,
            weight_per_loss,
            metrics,
            optimizer,
            scheduler,
            train_dataloader,
            valid_dataloader,
            max_grad_norm,
            device,
            denorm,
        )
        return trainer

    def on_train(self) -> List[float]:
        self.model.train()
        self.local_scaler.train()
        ave_loss_vs = [0.0 for _ in range(len(self.losses))]
        with tqdm(total=len(self.train_dataloader), leave=False) as pbar:
            for (
                X,
                y,
                bias,
                X_mask,
                y_mask,
                time_stamps,
                _,
                _,
            ) in self.train_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                bias = bias.to(self.device)
                X_mask = X_mask.to(self.device)
                y_mask = y_mask.to(self.device)
                if time_stamps is not None:
                    time_stamps = time_stamps.to(self.device)

                # NOTE: This is for second step optimziers like SAM
                def closure() -> Tensor:
                    Z = self.model(X, X_mask, time_stamps)
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
                    total_loss_v.backward()
                    return total_loss_v

                Z = self.model(X, X_mask, time_stamps)
                Z = Z + self.local_scaler(bias)
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
                if self.optimizer.require_second_step is True:
                    self.optimizer.step(closure)  # type: ignore
                else:
                    self.optimizer.step()
                pbar.set_description(
                    f"(loss={total_loss_v.item():.4f}, "
                    f"lr={self.scheduler.current_lr:.4f})"
                )
                pbar.update(1)
        self.scheduler.step()
        for i in range(len(self.losses)):
            ave_loss_vs[i] /= len(self.train_dataloader)
        return ave_loss_vs

    def on_val(self) -> List[float]:
        """Evaluate model on validation dataset.

        Returns
        -------
        List[float]
            List of averaged scores
        """
        self.model.eval()
        self.local_scaler.eval()
        with tqdm(total=len(self.valid_dataloader), leave=False) as pbar:
            for (
                X,
                y,
                bias,
                X_mask,
                y_mask,
                time_stamps,
                _,
                y_inv_transforms,
            ) in self.valid_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                bias = bias.to(self.device)
                X_mask = X_mask.to(self.device)
                y_mask = y_mask.to(self.device)
                if time_stamps is not None:
                    time_stamps = time_stamps.to(self.device)
                with torch.no_grad():
                    Z = self.model(X, X_mask, time_stamps)
                    Z = Z + self.local_scaler(bias)
                batch_size = X.size(0)
                if self.denorm is True:
                    for i in range(batch_size):
                        Z[i] = y_inv_transforms[i](Z[i])
                        y[i] = y_inv_transforms[i](y[i])
                for metric in self.metrics:
                    metric.update(Z, y, y_mask)
                metric_v = sum([metric(False) for metric in self.metrics])
                pbar.set_description(f"(metric={metric_v:.4f})")
                pbar.update(1)
        ave_scores = []
        for metric in self.metrics:
            ave_score = metric(True)
            ave_scores.append(ave_score)
        return ave_scores
