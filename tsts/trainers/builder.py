from typing import Type

from tsts.cfg import CfgNode as CN
from tsts.core import TRAINERS
from tsts.dataloaders import DataLoader
from tsts.models import Module
from tsts.optimizers import Optimizer

from .trainer import Trainer

__all__ = ["build_trainer"]


def build_trainer(
    model: Type[Module],
    optimizer: Type[Optimizer],
    train_dataloader: Type[DataLoader],
    val_dataloader: Type[DataLoader],
    cfg: CN,
) -> Type[Trainer]:
    trainer_name = cfg.TRAINER.NAME
    cls = TRAINERS[trainer_name]
    trainer = cls.from_cfg(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        cfg,
    )
    return trainer
