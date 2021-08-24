import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import Optimizer
from tsts.cfg import get_cfg_defaults
from tsts.collators import Collator, build_collator
from tsts.core import SCALERS
from tsts.dataloaders import DataLoader, build_dataloader
from tsts.datasets import Dataset, build_dataset
from tsts.loggers import Logger, build_logger
from tsts.losses import Loss
from tsts.losses.builder import build_losses
from tsts.metrics import Metric, build_metrics
from tsts.models import Module, build_model
from tsts.optimizers import build_optimizer
from tsts.scalers import Scaler, build_scaler
from tsts.trainers import Trainer, build_trainer
from tsts.types import MaybeRawDataset, RawDataset

__all__ = ["Solver"]


_TRAIN_INDEX = 0
_VALID_INDEX = 1
_INVALID_INDEX = -1
_FullRawDataset = Tuple[
    RawDataset,
    RawDataset,
    MaybeRawDataset,
    MaybeRawDataset,
]


class Solver(object):
    """Base solver class."""

    def __init__(self, cfg_path: Optional[str] = None) -> None:
        super(Solver, self).__init__()
        self.cfg = get_cfg_defaults()
        if cfg_path is not None:
            self.cfg.merge_from_file(cfg_path)
        # Load pretrained model for inference
        if self.log_dir_exist() is True:
            self.meta_info = self._load_meta_info()
            self.model = self._restore_model(self.meta_info)
            self.scaler = self._restore_scaler(self.meta_info)
        self.cfg_path = cfg_path

    def _load_meta_info(self) -> Dict[str, Any]:
        """Load meta info collected during training.

        Returns
        -------
        Dict[str, Any]
            Meta info
        """
        log_dir = self.cfg.LOGGER.LOG_DIR
        meta_info_path = os.path.join(log_dir, "meta.json")
        with open(meta_info_path, "r") as f:
            meta_info = json.load(f)
        return meta_info

    def _restore_model(self, meta_info: Dict[str, Any]) -> Module:
        """Restore pretrained model by meta_info.

        Parameters
        ----------
        meta_info : Dict[str, Any]
            Meta info collected during training

        Returns
        -------
        Module
            Pretrained model
        """
        num_in_feats = meta_info["num_in_feats"]
        num_out_feats = meta_info["num_out_feats"]
        model = self.build_model(num_in_feats, num_out_feats)
        model.eval()
        return model

    def _restore_scaler(self, meta_info: Dict[str, Any]) -> Scaler:
        """Restore scaler used during training

        Parameters
        ----------
        meta_info : Dict[str, Any]
            Meta info collected during training

        Returns
        -------
        Scaler
            Scaler
        """
        scaler_name = self.cfg.SCALER.NAME
        scaler = SCALERS[scaler_name](cfg=self.cfg, **meta_info["scaler"])
        return scaler

    def infer_num_in_feats(self, X: RawDataset) -> int:
        num_in_feats = X[0].size(-1)
        return num_in_feats

    def infer_num_out_feats(self, y: RawDataset) -> int:
        num_out_feats = self.infer_num_in_feats(y)
        return num_out_feats

    def log_dir_exist(self) -> bool:
        log_dir = self.cfg.LOGGER.LOG_DIR
        return os.path.exists(log_dir)

    def build_model(
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
        if self.log_dir_exist() is True:
            model_path = os.path.join(log_dir, "model.pth")
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
        return model

    def build_losses(self) -> List[Loss]:
        losses = build_losses(self.cfg)
        device = self.cfg.DEVICE
        for loss in losses:
            loss.to(device)
        return losses

    def build_metrics(self) -> List[Metric]:
        metrics = build_metrics(self.cfg)
        device = self.cfg.DEVICE
        for metric in metrics:
            metric.to(device)
        return metrics

    def build_optimizer(self, model: Module) -> Optimizer:
        optimizer = build_optimizer(model.parameters(), self.cfg)
        return optimizer

    def split_train_and_valid_data(
        self,
        X: RawDataset,
        y: Optional[RawDataset],
    ) -> _FullRawDataset:
        train_data_ratio = self.cfg.TRAINING.TRAIN_DATA_RATIO
        lookback = self.cfg.IO.LOOKBACK
        device = self.cfg.DEVICE
        X_train = []
        X_valid = []
        y_train: MaybeRawDataset = []
        y_valid: MaybeRawDataset = []
        num_datasets = len(X)
        for i in range(num_datasets):
            num_samples = len(X[i])
            num_train_samples = int(train_data_ratio * num_samples)
            indices = torch.zeros(num_samples, device=device)
            offset = num_train_samples
            indices[offset + lookback :] += _VALID_INDEX
            indices[offset : offset + lookback] += _INVALID_INDEX
            X_train.append(X[i][indices == _TRAIN_INDEX])
            X_valid.append(X[i][indices == _VALID_INDEX])
            if y is not None:
                y_train.append(y[i][indices == _TRAIN_INDEX])
                y_valid.append(y[i][indices == _VALID_INDEX])
            else:
                y_train.append(None)
                y_valid.append(None)
        return (X_train, X_valid, y_train, y_valid)

    def build_scaler(self, X_or_y: RawDataset) -> Scaler:
        scaler = build_scaler(X_or_y, self.cfg)
        return scaler

    def build_train_datasets(
        self,
        X: RawDataset,
        y: MaybeRawDataset,
    ) -> List[Dataset]:
        train_datasets = []
        num_datasets = len(X)
        for i in range(num_datasets):
            td = build_dataset(
                X[i],
                y[i],
                "train",
                self.cfg,
            )
            train_datasets.append(td)
        return train_datasets

    def build_valid_datasets(
        self,
        X: RawDataset,
        y: MaybeRawDataset,
    ) -> List[Dataset]:
        valid_datasets = []
        num_datasets = len(X)
        for i in range(num_datasets):
            vd = build_dataset(
                X[i],
                y[i],
                "valid",
                self.cfg,
            )
            valid_datasets.append(vd)
        return valid_datasets

    def build_test_dataset(self, X: RawDataset) -> List[Dataset]:
        test_datasets = []
        num_datasets = len(X)
        for i in range(num_datasets):
            td = build_dataset(
                X[i],
                None,
                "test",
                self.cfg,
            )
            test_datasets.append(td)
        return test_datasets

    def build_collator(self) -> Collator:
        collator = build_collator(self.cfg)
        return collator

    def build_train_dataloaders(
        self,
        train_datasets: List[Dataset],
        collator: Collator,
    ) -> List[DataLoader]:
        train_dataloaders = []
        num_datasets = len(train_datasets)
        for i in range(num_datasets):
            td = build_dataloader(
                train_datasets[i],
                "train",
                collator,
                self.cfg,
            )
            train_dataloaders.append(td)
        return train_dataloaders

    def build_valid_dataloaders(
        self,
        valid_datasets: List[Dataset],
        collator: Collator,
    ) -> List[DataLoader]:
        valid_dataloaders = []
        num_datasets = len(valid_datasets)
        for i in range(num_datasets):
            vd = build_dataloader(
                valid_datasets[i],
                "valid",
                collator,
                self.cfg,
            )
            valid_dataloaders.append(vd)
        return valid_dataloaders

    def build_trainer(
        self,
        model: Module,
        losses: List[Loss],
        metrics: List[Metric],
        optimizer: Optimizer,
        train_dataloaders: List[DataLoader],
        valid_dataloaders: List[DataLoader],
        scaler: Scaler,
    ) -> Trainer:
        trainer = build_trainer(
            model,
            losses,
            metrics,
            optimizer,
            train_dataloaders,
            valid_dataloaders,
            scaler,
            self.cfg,
        )
        return trainer

    def build_logger(
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
