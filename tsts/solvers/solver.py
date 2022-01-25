import json
import os
import sys
import warnings
from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset
from tsts.cfg import get_cfg_defaults
from tsts.collators import Collator, build_collator
from tsts.core import ContextManager
from tsts.dataloaders import DataLoader, build_dataloader
from tsts.datasets import Dataset, build_dataset
from tsts.loggers import Logger, build_logger
from tsts.losses import Loss
from tsts.losses.builder import build_losses
from tsts.metrics import Metric, build_metrics
from tsts.models import Module, build_model
from tsts.models.localscalers import build_local_scaler
from tsts.optimizers import build_optimizer
from tsts.scalers import Scaler
from tsts.schedulers import Scheduler, build_scheduler
from tsts.trainers import Trainer, build_trainer
from tsts.types import MaybeRawDataset, RawDataset
from tsts.utils import set_random_seed

__all__ = ["Solver"]


class Solver(object):
    """Base solver class.

    It has methods to build modules used to start training and inference, and has some utility
    methods.

    Parameters
    ----------
    cfg_path : str, optional
        Path to custom config file, by default None

    verbose : bool, optional
        If True, it prints meta info on console, by default True
    """

    def __init__(self, cfg_path: Optional[str] = None, verbose: bool = True) -> None:
        super(Solver, self).__init__()
        self.cfg_path = cfg_path
        self.verbose = verbose
        self._init_internal_state()

    def _init_internal_state(self) -> None:
        self._init_cfg()
        seed = self.cfg.SEED
        set_random_seed(seed)
        self._init_context_manager()
        if self.log_dir_exist() is True:
            self._load_meta_info()

    def _load_meta_info(self) -> None:
        if self.verbose is True:
            sys.stdout.write("log directory found \n")
            sys.stdout.write("restoring state...")
        log_dir = self.cfg.LOGGER.LOG_DIR
        meta_info_path = os.path.join(log_dir, "meta.json")
        with open(meta_info_path, "r") as f:
            meta_info = json.load(f)
        for (k, v) in meta_info.items():
            self.context_manager[k] = v
        if self.verbose is True:
            sys.stdout.write("\t [done] \n")

    def _init_cfg(self) -> None:
        self.cfg = get_cfg_defaults()
        if self.cfg_path is not None:
            self.cfg.merge_from_file(self.cfg_path)

    def _init_context_manager(self) -> None:
        self.context_manager = ContextManager()

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
            try:
                model_path = os.path.join(log_dir, "model.pth")
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
            except FileNotFoundError:
                warnings.warn("Failed to load pretrained model")
        return model

    def build_local_scaler(
        self,
        num_in_feats: int,
        num_out_feats: int,
    ) -> Module:
        local_scaler = build_local_scaler(
            num_in_feats,
            num_out_feats,
            self.cfg,
        )
        device = self.cfg.DEVICE
        local_scaler.to(device)
        log_dir = self.cfg.LOGGER.LOG_DIR
        if self.log_dir_exist() is True:
            try:
                local_scaler_path = os.path.join(log_dir, "local_scaler.pth")
                state_dict = torch.load(local_scaler_path)
                local_scaler.load_state_dict(state_dict)
            except FileNotFoundError:
                warnings.warn("Failed to load pretrained local scaler")
        return local_scaler

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

    def build_optimizer(self, model: Module, local_scaler: Module) -> Optimizer:
        params = list(model.parameters()) + list(local_scaler.parameters())
        optimizer = build_optimizer(params, self.cfg)
        return optimizer

    def build_scheduler(
        self,
        optimizer: Optimizer,
        iters_per_epoch: int,
    ) -> Scheduler:
        scheduler = build_scheduler(
            optimizer,  # type: ignore
            iters_per_epoch,
            self.cfg,
        )
        return scheduler

    def build_train_dataset(
        self,
        X: RawDataset,
        y: RawDataset,
        time_stamps: MaybeRawDataset,
        X_scaler: Scaler,
        y_scaler: Scaler,
    ) -> Dataset:
        train_datasets = []
        num_datasets = len(X)
        for i in range(num_datasets):
            td = build_dataset(
                X[i],
                y[i],
                time_stamps[i] if time_stamps is not None else None,
                "train",
                X_scaler,
                y_scaler,
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
        X_scaler: Scaler,
        y_scaler: Scaler,
    ) -> Dataset:
        valid_datasets = []
        num_datasets = len(X)
        for i in range(num_datasets):
            vd = build_dataset(
                X[i],
                y[i],
                time_stamps[i] if time_stamps is not None else None,
                "valid",
                X_scaler,
                y_scaler,
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
        local_scaler: Module,
        losses: List[Loss],
        metrics: List[Metric],
        optimizer: Optimizer,
        scheduler: Scheduler,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
    ) -> Trainer:
        trainer = build_trainer(
            model,
            local_scaler,
            losses,
            metrics,
            optimizer,  # type: ignore
            scheduler,
            train_dataloader,
            valid_dataloader,
            self.cfg,
        )
        return trainer

    def build_logger(
        self,
        model: Module,
        local_scaler: Module,
        losses: List[Loss],
        metrics: List[Metric],
        context_manager: ContextManager,
    ) -> Logger:
        logger = build_logger(
            model,
            local_scaler,
            losses,
            metrics,
            context_manager,
            self.cfg,
        )
        return logger
