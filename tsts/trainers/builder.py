from typing import List

from tsts.cfg import CfgNode as CN
from tsts.core import TRAINERS
from tsts.dataloaders import DataLoader
from tsts.losses import Loss
from tsts.metrics import Metric
from tsts.models import Module
from tsts.optimizers import Optimizer
from tsts.scalers import Scaler

from .trainer import Trainer

__all__ = ["build_trainer"]


def build_trainer(
    model: Module,
    losses: List[Loss],
    metrics: List[Metric],
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    scaler: Scaler,
    cfg: CN,
) -> Trainer:
    trainer_name = cfg.TRAINER.NAME
    cls = TRAINERS[trainer_name]
    trainer = cls.from_cfg(
        model,
        losses,
        metrics,
        optimizer,
        train_dataloader,
        valid_dataloader,
        scaler,
        cfg,
    )
    return trainer
