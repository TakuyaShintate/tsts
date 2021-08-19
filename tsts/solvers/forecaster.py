import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
from tsts.cfg import CfgNode as CN
from tsts.cfg import get_cfg_defaults
from tsts.collators import Collator, build_collator
from tsts.dataloaders import DataLoader, build_dataloader
from tsts.datasets import Dataset, build_dataset
from tsts.loggers import Logger, build_logger
from tsts.losses import Loss
from tsts.losses.builder import build_losses
from tsts.metrics import Metric, build_metrics
from tsts.models import Module, build_model
from tsts.optimizers import build_optimizer
from tsts.trainers import Trainer, build_trainer
from tsts.types import RawDataset
from tsts.utils import infer_dataset_type

from .solver import Solver

__all__ = ["Forecaster"]

_TRAIN_INDEX = 0
_VAL_INDEX = 1
_INVALID_INDEX = -1
_FullRawDataset = Tuple[
    RawDataset,
    RawDataset,
    Optional[RawDataset],
    Optional[RawDataset],
]


class Forecaster(Solver):
    """Tool to solve time series forecasting."""

    def __init__(self, cfg: Optional[CN] = None) -> None:
        super(Forecaster, self).__init__()
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = get_cfg_defaults()

    def _infer_num_in_feats(self, X: RawDataset) -> int:
        num_in_feats = X[0].size(0)
        return num_in_feats

    def _infer_num_out_feats(self, y: RawDataset) -> int:
        num_out_feats = self._infer_num_in_feats(y)
        return num_out_feats

    def _build_model(
        self,
        num_in_feats: int,
        num_out_feats: int,
    ) -> Module:
        model = build_model(
            num_in_feats,
            num_out_feats,
            self.cfg,
        )
        device = self.cfg.DEVICE
        model.to(device)
        log_dir = self.cfg.LOGGER.LOG_DIR
        if os.path.exists(log_dir) is True:
            model_path = os.path.join(log_dir, "model.pth")
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
        return model

    def _build_losses(self) -> List[Loss]:
        losses = build_losses(self.cfg)
        device = self.cfg.DEVICE
        for loss in losses:
            loss.to(device)
        return losses

    def _build_metrics(self) -> List[Metric]:
        metrics = build_metrics(self.cfg)
        device = self.cfg.DEVICE
        for metric in metrics:
            metric.to(device)
        return metrics

    def _build_optimizer(self, model: Module) -> Optimizer:
        optimizer = build_optimizer(model.parameters(), self.cfg)
        return optimizer

    def _split_train_and_val_data(
        self,
        X: RawDataset,
        y: Optional[RawDataset],
    ) -> _FullRawDataset:
        dataset_type = infer_dataset_type(X)
        train_data_ratio = self.cfg.TRAINING.TRAIN_DATA_RATIO
        lookback = self.cfg.IO.LOOKBACK
        if dataset_type == "mn":
            num_samples = len(X)
            num_train_samples = int(train_data_ratio * num_samples)
            device = self.cfg.DEVICE
            indices = torch.zeros(num_samples, device=device)
            offset = num_train_samples
            indices[offset + lookback :] += _VAL_INDEX
            indices[offset : offset + lookback] += _INVALID_INDEX
        else:
            raise NotImplementedError
        X_train = X[indices == _TRAIN_INDEX]
        X_val = X[indices == _VAL_INDEX]
        if y is not None:
            y_train: Optional[Tensor] = y[indices == _TRAIN_INDEX]
            y_val: Optional[Tensor] = y[indices == _VAL_INDEX]
        else:
            y_train = None
            y_val = None
        return (X_train, X_val, y_train, y_val)

    def _build_train_dataset(
        self,
        X: RawDataset,
        y: Optional[RawDataset],
    ) -> Dataset:
        train_dataset = build_dataset(
            X,
            y,
            "train",
            self.cfg,
        )
        return train_dataset

    def _build_val_dataset(
        self,
        X: RawDataset,
        y: Optional[RawDataset],
    ) -> Dataset:
        val_dataset = build_dataset(
            X,
            y,
            "val",
            self.cfg,
        )
        return val_dataset

    def _build_collator(self) -> Collator:
        collator = build_collator(self.cfg)
        return collator

    def _build_train_dataloader(
        self,
        train_dataset: Dataset,
        collator: Collator,
    ) -> DataLoader:
        train_dataloader = build_dataloader(
            train_dataset,
            "train",
            collator,
            self.cfg,
        )
        return train_dataloader

    def _build_val_dataloader(
        self,
        val_dataset: Dataset,
        collator: Collator,
    ) -> DataLoader:
        val_dataloader = build_dataloader(
            val_dataset,
            "val",
            collator,
            self.cfg,
        )
        return val_dataloader

    def _build_trainer(
        self,
        model: Module,
        losses: List[Loss],
        metrics: List[Metric],
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> Trainer:
        trainer = build_trainer(
            model,
            losses,
            metrics,
            optimizer,
            train_dataloader,
            val_dataloader,
            self.cfg,
        )
        return trainer

    def _build_logger(
        self,
        model: Module,
        losses: List[Loss],
        metrics: List[Metric],
        meta_info: Dict[str, Any],
    ) -> Logger:
        logger = build_logger(
            model,
            losses,
            metrics,
            meta_info,
            self.cfg,
        )
        return logger

    def predict(self, X: RawDataset) -> None:
        pass

    def fit(
        self,
        X: RawDataset,
        y: Optional[RawDataset] = None,
    ) -> None:
        """Train the target model on given dataset.

        Parameters
        ----------
        X : RawDataset, (L, N, M) or (N, M)
            Input on witch the model is trained

        y : Optional[RawDataset], optional
            Target, by default None
        """
        num_in_feats = self._infer_num_in_feats(X)
        num_out_feats = self._infer_num_out_feats(y if y is not None else X)
        model = self._build_model(num_in_feats, num_out_feats)
        losses = self._build_losses()
        metrics = self._build_metrics()
        optimizer = self._build_optimizer(model)
        (X_train, X_val, y_train, y_val) = self._split_train_and_val_data(X, y)
        train_dataset = self._build_train_dataset(X_train, y_train)
        val_dataset = self._build_val_dataset(X_val, y_val)
        collator = self._build_collator()
        train_dataloader = self._build_train_dataloader(train_dataset, collator)
        val_dataloader = self._build_val_dataloader(val_dataset, collator)
        meta_info = {"num_in_feats": num_in_feats, "num_out_feats": num_out_feats}
        logger = self._build_logger(model, losses, metrics, meta_info)
        trainer = self._build_trainer(
            model,
            losses,
            metrics,
            optimizer,
            train_dataloader,
            val_dataloader,
        )
        num_epochs = self.cfg.TRAINING.NUM_EPOCHS
        for i in range(num_epochs):
            (ave_loss_vals, ave_scores) = trainer.step()
            logger.log(i, ave_loss_vals, ave_scores)
