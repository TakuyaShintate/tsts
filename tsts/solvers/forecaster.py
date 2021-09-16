import typing
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from tsts.collators import Collator
from tsts.dataloaders.dataloader import DataLoader
from tsts.datasets import Dataset
from tsts.loggers import Logger
from tsts.losses.loss import Loss
from tsts.metrics import Metric
from tsts.models.module import Module
from tsts.optimizers import Optimizer
from tsts.scalers import Scaler
from tsts.scalers.builder import build_scaler
from tsts.schedulers import Scheduler
from tsts.trainers import Trainer
from tsts.types import MaybeRawDataset, RawDataset

from .solver import Solver

__all__ = ["TimeSeriesForecaster"]

_TestData = Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]


class TimeSeriesForecaster(Solver):
    """Tool to solve time series forecasting."""

    @property
    def num_in_feats(self) -> int:
        """Get the number of input features.

        This value is inferred by a given dataset.

        Returns
        -------
        int
            Number of input features
        """
        if "num_in_feats" not in self.context_manager:
            num_in_feats = self.infer_num_in_feats(self.X)
            self.context_manager["num_in_feats"] = num_in_feats
        num_in_feats = self.context_manager["num_in_feats"]
        return num_in_feats

    @property
    def num_out_feats(self) -> int:
        """Get the number of output features.

        This value is inferred by a given dataset.

        Returns
        -------
        int
            Number of output features
        """
        if "num_out_feats" not in self.context_manager:
            if self.y is not None:
                num_out_feats = self.infer_num_out_feats(self.y)
            else:
                num_out_feats = self.infer_num_out_feats(self.X)
            self.context_manager["num_out_feats"] = num_out_feats
        num_out_feats = self.context_manager["num_out_feats"]
        return num_out_feats

    @property
    def model(self) -> Module:
        """Get a target model.

        Returns
        -------
        Module
            Target model
        """
        if "model" not in self.context_manager:
            model = self.build_model(self.num_in_feats, self.num_out_feats)
            self.context_manager["model"] = model
        model = self.context_manager["model"]
        return model

    @property
    def local_scaler(self) -> Module:
        """Get a target local scaler.

        Returns
        -------
        Module
            Target local scaler
        """
        if "local_scaler" not in self.context_manager:
            local_scaler = self.build_local_scaler(
                self.num_in_feats,
                self.num_out_feats,
            )
            self.context_manager["local_scaler"] = local_scaler
        local_scaler = self.context_manager["local_scaler"]
        return local_scaler

    @property
    def losses(self) -> List[Loss]:
        """Get a list of loss functions.

        Returns
        -------
        Loss
            List of loss functions
        """
        if "losses" not in self.context_manager:
            losses = self.build_losses()
            self.context_manager["losses"] = losses
        losses = self.context_manager["losses"]
        return losses

    @property
    def metrics(self) -> List[Metric]:
        """Get a list of metrics.

        Returns
        -------
        List[Metric]
            List of metrics
        """
        if "metrics" not in self.context_manager:
            metrics = self.build_metrics()
            self.context_manager["metrics"] = metrics
        metrics = self.context_manager["metrics"]
        return metrics

    @property
    def optimizer(self) -> Optimizer:
        """Get an optimizer.

        Returns
        -------
        Optimizer
            Optimizer
        """
        if "optimizer" not in self.context_manager:
            optimizer = self.build_optimizer(self.model, self.local_scaler)
            self.context_manager["optimizer"] = optimizer
        optimizer = self.context_manager["optimizer"]
        return optimizer

    @property
    def scheduler(self) -> Scheduler:
        """Get an scheduler.

        Returns
        -------
        Scheduler
            Learning rate scheduler
        """
        if "scheduler" not in self.context_manager:
            scheduler = self.build_scheduler(self.optimizer)
            self.context_manager["scheduler"] = scheduler
        scheduler = self.context_manager["scheduler"]
        return scheduler

    @property
    def num_train_samples(self) -> List[int]:
        """Get a list of training samples per dataset.

        There are 2 types of datasets. If TRAIN_DATA_SPLIT = "col", it returns a list of training
        samples per dataset. If TRAIN_DATA_SPLIT = "row", it returns a list to contains a
        single value. In this case, first num_train_samples[0] rows are used as training samples
        and the last n - num_train_samples[0] rows are used as validation samples where n is the
        total number of samples.

        Returns
        -------
        List[int]
            List of training samples per dataset
        """
        if "train_and_valid_indices" not in self.context_manager:
            train_data_ratio = self.cfg.TRAINING.TRAIN_DATA_RATIO
            train_data_split = self.cfg.TRAINING.TRAIN_DATA_SPLIT
            num_datasets = len(self.X)
            num_train_samples = []
            # Split like [[train, valid], [train, valid], ...]
            if train_data_split == "col":
                for i in range(num_datasets):
                    num_samples = len(self.X[i])
                    num_train_samples.append(int(train_data_ratio * num_samples))
            # Split like [train, train, valid, valid, ...]
            else:
                num_train_samples.append(int(train_data_ratio * num_datasets))
            self.context_manager["num_train_samples"] = num_train_samples
        num_train_samples = self.context_manager["num_train_samples"]
        return num_train_samples

    @property
    def X_train(self) -> RawDataset:
        """Get a training raw input dataset.

        Returns
        -------
        RawDataset
            Training raw input dataset
        """
        if "X_train" not in self.context_manager:
            train_data_split = self.cfg.TRAINING.TRAIN_DATA_SPLIT
            if train_data_split == "col":
                X_train = []
                num_datasets = len(self.X)
                for i in range(num_datasets):
                    X_train.append(self.X[i][: self.num_train_samples[i]])
            else:
                num_train_samples = self.num_train_samples[0]
                X_train = self.X[:num_train_samples]
            self.context_manager["X_train"] = X_train
        X_train = self.context_manager["X_train"]
        return X_train

    @property
    def X_valid(self) -> RawDataset:
        """Get a validation raw input dataset.

        Returns
        -------
        RawDataset
            Validation raw input dataset
        """
        if "X_valid" not in self.context_manager:
            train_data_split = self.cfg.TRAINING.TRAIN_DATA_SPLIT
            lookback = self.cfg.IO.LOOKBACK
            if train_data_split == "col":
                X_valid = []
                num_datasets = len(self.X)
                for i in range(num_datasets):
                    X_valid.append(self.X[i][self.num_train_samples[i] + lookback :])
            else:
                X_valid = self.X[self.num_train_samples[0] :]
            self.context_manager["X_valid"] = X_valid
        X_valid = self.context_manager["X_valid"]
        return X_valid

    @property
    def y_train(self) -> RawDataset:
        """Get a training raw target dataset.

        Returns
        -------
        RawDataset
            Training raw target dataset
        """
        if "y_train" not in self.context_manager:
            train_data_split = self.cfg.TRAINING.TRAIN_DATA_SPLIT
            if train_data_split == "col":
                y_train = []
                num_datasets = len(self.y)
                for i in range(num_datasets):
                    y_train.append(self.y[i][: self.num_train_samples[i]])
            else:
                y_train = self.y[: self.num_train_samples[0]]
            self.context_manager["y_train"] = y_train
        y_train = self.context_manager["y_train"]
        return y_train

    @property
    def y_valid(self) -> RawDataset:
        """Get a validation raw target dataset.

        Returns
        -------
        RawDataset
            Validation raw target dataset
        """
        if "y_valid" not in self.context_manager:
            train_data_split = self.cfg.TRAINING.TRAIN_DATA_SPLIT
            lookback = self.cfg.IO.LOOKBACK
            if train_data_split == "col":
                y_valid = []
                num_datasets = len(self.y)
                for i in range(num_datasets):
                    y_valid.append(self.y[i][self.num_train_samples[i] + lookback :])
            else:
                y_valid = self.y[self.num_train_samples[0] :]
            self.context_manager["y_valid"] = y_valid
        y_valid = self.context_manager["y_valid"]
        return y_valid

    @property
    def time_stamps_train(self) -> MaybeRawDataset:
        """Get time stamps for training samples

        Returns
        -------
        MaybeRawDataset
            time stamps for training samples
        """
        if "time_stamps_train" not in self.context_manager:
            # time_stamps is given as input
            if self.time_stamps is not None:
                train_data_split = self.cfg.TRAINING.TRAIN_DATA_SPLIT
                time_stamps_train = []
                if train_data_split == "col":
                    num_datasets = len(self.y)
                    for i in range(num_datasets):
                        if self.time_stamps is not None:
                            ts = self.time_stamps[i][: self.num_train_samples[i]]
                            time_stamps_train.append(ts)
                else:
                    if self.time_stamps is not None:
                        ts = self.time_stamps[0][: self.num_train_samples[0]]
                        time_stamps_train.append(ts)
                self.context_manager["time_stamps_train"] = time_stamps_train
            else:
                self.context_manager["time_stamps_train"] = None
        time_stamps_train = self.context_manager["time_stamps_train"]
        return typing.cast(MaybeRawDataset, time_stamps_train)

    @property
    def time_stamps_valid(self) -> MaybeRawDataset:
        """Get time stamps for validation samples

        Returns
        -------
        MaybeRawDataset
            time stamps for validation samples
        """
        if "time_stamps_valid" not in self.context_manager:
            # time_stamps is given as input
            if self.time_stamps is not None:
                train_data_split = self.cfg.TRAINING.TRAIN_DATA_SPLIT
                lookback = self.cfg.IO.LOOKBACK
                time_stamps_valid = []
                if train_data_split == "col":
                    num_datasets = len(self.y)
                    for i in range(num_datasets):
                        if self.time_stamps is not None:
                            offset = self.num_train_samples[i] + lookback
                            ts = self.time_stamps[i][offset:]
                            time_stamps_valid.append(ts)
                else:
                    if self.time_stamps is not None:
                        offset = self.num_train_samples[0]
                        ts = self.time_stamps[0][offset:]
                        time_stamps_valid.append(ts)
                self.context_manager["time_stamps_valid"] = time_stamps_valid
            else:
                self.context_manager["time_stamps_valid"] = None
        time_stamps_valid = self.context_manager["time_stamps_valid"]
        return typing.cast(MaybeRawDataset, time_stamps_valid)

    @property
    def train_dataset(self) -> Dataset:
        """Get a training dataset.

        RawDataset is a list of datasets. They are concatenated inside build_train_dataset method.

        Returns
        -------
        Dataset
            Training dataset
        """
        if "train_dataset" not in self.context_manager:
            train_dataset = self.build_train_dataset(
                self.X_train,
                self.y_train,
                self.time_stamps_train,
                self.X_scaler,
                self.y_scaler,
            )
            self.context_manager["train_dataset"] = train_dataset
        train_dataset = self.context_manager["train_dataset"]
        return train_dataset

    @property
    def valid_dataset(self) -> Dataset:
        """Get a validation dataset.

        RawDataset is a list of datasets. They are concatenated inside build_train_dataset method.

        Returns
        -------
        Dataset
            Validation dataset
        """
        if "valid_dataset" not in self.context_manager:
            valid_dataset = self.build_valid_dataset(
                self.X_valid,
                self.y_valid,
                self.time_stamps_valid,
                self.X_scaler,
                self.y_scaler,
            )
            self.context_manager["valid_dataset"] = valid_dataset
        valid_dataset = self.context_manager["valid_dataset"]
        return valid_dataset

    @property
    def X_scaler(self) -> Scaler:
        """Get a scaler for input.

        Returns
        -------
        Scaler
            Scale for input
        """
        if "X_scaler" not in self.context_manager:
            X_scaler = build_scaler(self.cfg)
            X_scaler.fit_batch(self.X_train)
            self.context_manager["X_scaler"] = X_scaler
        X_scaler = self.context_manager["X_scaler"]
        return X_scaler

    @property
    def y_scaler(self) -> Scaler:
        """Get a scaler for target.

        Returns
        -------
        Scaler
            Scale for target
        """
        if "y_scaler" not in self.context_manager:
            y_scaler = build_scaler(self.cfg)
            y_scaler.fit_batch(self.y_train)
            self.context_manager["y_scaler"] = y_scaler
        y_scaler = self.context_manager["y_scaler"]
        return y_scaler

    @property
    def collator(self) -> Collator:
        """Get a collator.

        Returns
        -------
        Collator
            Collator
        """
        if "collator" not in self.context_manager:
            collator = self.build_collator()
            self.context_manager["collator"] = collator
        collator = self.context_manager["collator"]
        return collator

    @property
    def train_dataloader(self) -> DataLoader:
        """Get a training dataloader.

        Returns
        -------
        DataLoader
            Training dataloader
        """
        if "train_dataloader" not in self.context_manager:
            train_dataloader = self.build_train_dataloader(
                self.train_dataset,
                self.collator,
            )
            self.context_manager["train_dataloader"] = train_dataloader
        train_dataloader = self.context_manager["train_dataloader"]
        return train_dataloader

    @property
    def valid_dataloader(self) -> DataLoader:
        """Get a validation dataloader.

        Returns
        -------
        DataLoader
            Validation dataloader
        """
        if "valid_dataloader" not in self.context_manager:
            valid_dataloader = self.build_valid_dataloader(
                self.valid_dataset,
                self.collator,
            )
            self.context_manager["valid_dataloader"] = valid_dataloader
        valid_dataloader = self.context_manager["valid_dataloader"]
        return valid_dataloader

    @property
    def logger(self) -> Logger:
        """Get a logger

        Returns
        -------
        Logger
            Logger
        """
        if "logger" not in self.context_manager:
            logger = self.build_logger(
                self.model,
                self.local_scaler,
                self.losses,
                self.metrics,
                self.context_manager,
            )
            self.context_manager["logger"] = logger
        logger = self.context_manager["logger"]
        return logger

    @property
    def trainer(self) -> Trainer:
        """Get a trainer

        Returns
        -------
        Trainer
            Trainer
        """
        if "trainer" not in self.context_manager:
            trainer = self.build_trainer(
                self.model,
                self.local_scaler,
                self.losses,
                self.metrics,
                self.optimizer,
                self.scheduler,
                self.train_dataloader,
                self.valid_dataloader,
            )
            self.context_manager["trainer"] = trainer
        trainer = self.context_manager["trainer"]
        return trainer

    def _register_raw_datasets(
        self,
        X: RawDataset,
        y: Optional[RawDataset],
        time_stamps: Optional[RawDataset],
    ) -> None:
        self.X = X
        self.y = y or X
        self.time_stamps = time_stamps

    def _apply_collator_to_test_data(
        self,
        X: Tensor,
        time_stamps: Optional[Tensor] = None,
    ) -> _TestData:
        raw_batch = (
            (
                X,
                torch.zeros_like(X),  # Dummy (y)
                X,
                time_stamps,
                lambda x: x,  # Dummy (X_inv_transform)
                lambda x: x,  # Dummy (y_inv_transform)
            ),
        )
        # Unpack a batch
        batch = self.collator(raw_batch)
        (X, _, bias, X_mask, _, time_stamps, _, _) = batch
        return (X, bias, X_mask, time_stamps)

    def predict(
        self,
        X: Tensor,
        time_stamps: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict y for X.

        Notes
        -----
        Unlike fit method, X and time_stamps are not lists.

        Parameters
        ----------
        X : Tensor
            Input
        """
        self.model.eval()
        self.local_scaler.eval()
        (X, bias, X_mask, time_stamps) = self._apply_collator_to_test_data(
            X, time_stamps
        )
        device = self.cfg.DEVICE
        X = X.to(device)
        bias = bias.to(device)
        X_mask = X_mask.to(device)
        if time_stamps is not None:
            time_stamps = time_stamps.to(device)
        Z = self.model(X, X_mask, time_stamps)
        Z = Z + self.local_scaler(bias)
        Z = Z[0].detach().cpu()
        return Z

    def fit(
        self,
        X: RawDataset,
        y: Optional[RawDataset] = None,
        time_stamps: Optional[RawDataset] = None,
    ) -> None:
        """Train model on given datasets.

        Notes
        -----
        If y is None, X is used for y.

        Parameters
        ----------
        X : RawDataset, (L, N, M)
            List of input datasets

        y : Optional[RawDataset], optional
            List of target datasets, by default None
        """
        num_epochs = self.cfg.TRAINING.NUM_EPOCHS
        self._register_raw_datasets(X, y, time_stamps)
        for i in range(num_epochs):
            (ave_loss_vals, ave_scores) = self.trainer.step()
            self.logger.log(i, ave_loss_vals, ave_scores)
