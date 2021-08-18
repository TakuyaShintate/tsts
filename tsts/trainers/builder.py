from typing import List

from tsts.cfg import CfgNode as CN
from tsts.core import TRAINERS
from tsts.dataloaders import DataLoader
from tsts.losses import Loss
from tsts.metrics import Metric
from tsts.models import Module
from tsts.optimizers import Optimizer

from .trainer import Trainer

__all__ = ["build_trainer"]


def build_trainer(
    model: Module,
    losses: List[Loss],
    metrics: List[Metric],
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
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
        val_dataloader,
        cfg,
    )
    return trainer