import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset
from tsts.cfg import get_cfg_defaults
from tsts.collators import Collator, build_collator
from tsts.core import SCALERS, ContextManager
from tsts.dataloaders import DataLoader, build_dataloader
from tsts.datasets import Dataset, build_dataset
from tsts.loggers import Logger, build_logger
from tsts.losses import Loss
from tsts.losses.builder import build_losses
from tsts.metrics import Metric, build_metrics
from tsts.models import Module, build_model
from tsts.optimizers import build_optimizer
from tsts.scalers import Scaler, build_scaler
from tsts.schedulers import Scheduler, build_scheduler
from tsts.trainers import Trainer, build_trainer
from tsts.types import MaybeRawDataset, RawDataset
from tsts.utils import set_random_seed

__all__ = ["Solver"]


class Solver(object):
    """Base solver class."""

    def __init__(self, cfg_path: Optional[str] = None, verbose: bool = True) -> None:
        super(Solver, self).__init__()
        self.cfg = get_cfg_defaults()
        if cfg_path is not None:
            self.cfg.merge_from_file(cfg_path)
        seed = self.cfg.SEED
        set_random_seed(seed)
        # Load pretrained model for inference
        """
        if self.log_dir_exist() is True:
            if verbose is True:
                sys.stdout.write("Log directory found \n")
                sys.stdout.write("Restoring state ...")
            self.meta_info = self._load_meta_info()
            self.model = self._restore_model(self.meta_info)
            (self.X_scaler, self.y_scaler) = self._restore_scaler(self.meta_info)
            if verbose is True:
                sys.stdout.write("\t [done] \n")
        """
        self._init_context_manager()
        self.cfg_path = cfg_path
        self.verbose = verbose

    def _init_context_manager(self) -> None:
        self.context_manager = ContextManager()

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

    def _restore_scaler(self, meta_info: Dict[str, Any]) -> Tuple[Scaler, Scaler]:
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
        X_scaler = SCALERS[scaler_name](cfg=self.cfg, **meta_info["X_scaler"])
        y_scaler = SCALERS[scaler_name](cfg=self.cfg, **meta_info["y_scaler"])
        return (X_scaler, y_scaler)

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

    def build_scheduler(self, optimizer: Optimizer) -> Scheduler:
        scheduler = build_scheduler(optimizer, self.cfg)
        return scheduler

    def build_scaler(self, X_or_y: RawDataset) -> Scaler:
        scaler = build_scaler(X_or_y, self.cfg)
        return scaler

    def build_train_dataset(
        self,
        X: RawDataset,
        y: RawDataset,
        time_stamps: MaybeRawDataset,
    ) -> Dataset:
        train_datasets = []
        num_datasets = len(X)
        for i in range(num_datasets):
            td = build_dataset(
                X[i],
                y[i],
                time_stamps[i] if time_stamps is not None else None,
                "train",
                self.cfg,
            )
            train_datasets.append(td)
        train_dataset = ConcatDataset(train_datasets)  # type: ignore
        return train_dataset  # type: ignore

    def build_valid_dataset(
        self,
        X: RawDataset,
        y: RawDataset,
        time_stamps: MaybeRawDataset,
    ) -> Dataset:
        valid_datasets = []
        num_datasets = len(X)
        for i in range(num_datasets):
            vd = build_dataset(
                X[i],
                y[i],
                time_stamps[i] if time_stamps is not None else None,
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
        scheduler: Scheduler,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        X_scaler: Scaler,
        y_scaler: Scaler,
    ) -> Trainer:
        trainer = build_trainer(
            model,
            losses,
            metrics,
            optimizer,
            scheduler,
            train_dataloader,
            valid_dataloader,
            X_scaler,
            y_scaler,
            self.cfg,
        )
        return trainer

    def build_logger(
        self,
        model: Module,
        losses: List[Loss],
        metrics: List[Metric],
        context_manager: ContextManager,
    ) -> Logger:
        logger = build_logger(
            model,
            losses,
            metrics,
            context_manager,
            self.cfg,
        )
        return logger
