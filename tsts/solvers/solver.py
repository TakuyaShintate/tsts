import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset
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

    def __init__(self, cfg_path: Optional[str] = None, verbose: bool = True) -> None:
        super(Solver, self).__init__()
        self.cfg = get_cfg_defaults()
        if cfg_path is not None:
            self.cfg.merge_from_file(cfg_path)
        # Load pretrained model for inference
        if self.log_dir_exist() is True:
            if verbose is True:
                sys.stdout.write("Log directory found \n")
                sys.stdout.write("Restoring state ...")
            self.meta_info = self._load_meta_info()
            self.model = self._restore_model(self.meta_info)
            self.scaler = self._restore_scaler(self.meta_info)
            if verbose is True:
                sys.stdout.write("\t [done] \n")
        self.cfg_path = cfg_path
        self.verbose = verbose

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
        train_data_split = self.cfg.TRAINING.TRAIN_DATA_SPLIT
        num_datasets = len(X)
        if train_data_split == "col":
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
        elif train_data_split == "row":
            num_train_samples = int(train_data_ratio * num_datasets)
            X_train = X[:num_train_samples]
            X_valid = X[num_train_samples:]
            if y is not None:
                y_train = y[:num_train_samples]  # type: ignore
                y_valid = y[num_train_samples:]  # type: ignore
            else:
                y_train = [None for _ in range(len(X_train))]
                y_valid = [None for _ in range(len(X_valid))]
        else:
            raise ValueError(f"Invalid train_data_split: {train_data_split}")
        return (X_train, X_valid, y_train, y_valid)

    def build_scaler(self, X_or_y: RawDataset) -> Scaler:
        scaler = build_scaler(X_or_y, self.cfg)
        return scaler

    def build_train_dataset(
        self,
        X: RawDataset,
        y: MaybeRawDataset,
    ) -> Dataset:
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
        train_dataset = ConcatDataset(train_datasets)  # type: ignore
        return train_dataset  # type: ignore

    def build_valid_dataset(
        self,
        X: RawDataset,
        y: MaybeRawDataset,
    ) -> Dataset:
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
        valid_dataset = ConcatDataset(valid_datasets)  # type: ignore
        return valid_dataset  # type: ignore

    def build_collator(self) -> Collator:
        collator = build_collator(self.cfg)
        return collator

    def build_train_dataloader(
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

    def build_valid_dataloader(
        self,
        valid_dataset: Dataset,
        collator: Collator,
    ) -> DataLoader:
        valid_dataloader = build_dataloader(
            valid_dataset,
            "valid",
            collator,
            self.cfg,
        )
        return valid_dataloader

    def build_trainer(
        self,
        model: Module,
        losses: List[Loss],
        metrics: List[Metric],
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        scaler: Scaler,
    ) -> Trainer:
        trainer = build_trainer(
            model,
            losses,
            metrics,
            optimizer,
            train_dataloader,
            valid_dataloader,
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
