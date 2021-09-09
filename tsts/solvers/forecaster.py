from typing import Any, Dict, Optional

import torch
from torch import Tensor
from tsts.types import RawDataset

from .solver import Solver

__all__ = ["TimeSeriesForecaster"]


class TimeSeriesForecaster(Solver):
    """Tool to solve time series forecasting."""

    def predict(
        self,
        X: Tensor,
        bias: Optional[Tensor] = None,
        time_stamps: Optional[RawDataset] = None,
    ) -> Tensor:
        self.model.eval()
        if bias is None:
            bias = X
        src_device = X.device
        device = self.cfg.DEVICE
        X = X.to(device)
        bias = bias.to(device)
        lookback = self.cfg.IO.LOOKBACK
        horizon = self.cfg.IO.HORIZON
        num_in_feats = self.meta_info["num_in_feats"]
        num_out_feats = self.meta_info["num_out_feats"]
        X_new = torch.zeros((1, lookback, num_in_feats))
        bias_new = torch.zeros((1, lookback, num_out_feats))
        X_new = X_new.to(device)
        bias_new = bias_new.to(device)
        if time_stamps is not None:
            time_stamps = time_stamps.to(device)  # type: ignore
            time_stamps_new = torch.zeros((1, lookback + horizon, time_stamps.size(1)))
            time_stamps_new = time_stamps_new.type(torch.long)
            time_stamps_new = time_stamps_new.to(device)
            y_size = time_stamps.size(0) - X.size(0)
            start = -horizon - X.size(0)
            end = -horizon
            time_stamps_new[:, start:end] = time_stamps[: X.size(0)]
            start = -horizon
            end = -horizon + y_size
            if end >= 0:
                time_stamps_new[:, start:] = time_stamps[X.size(0) :]
            else:
                time_stamps_new[:, start:end] = time_stamps[X.size(0) :]
        else:
            time_stamps_new = None  # type: ignore
        X_new[:, -X.size(0) :] = X[-lookback:]
        bias_new[:, -X.size(0) :] = bias[-lookback:]
        X_new = self.X_scaler.transform([X_new])[0]
        bias_new = self.y_scaler.transform([bias_new])[0]
        X_mask = torch.zeros((1, lookback, num_in_feats))
        X_mask = X_mask.to(device)
        X_mask[:, -X.size(0) :] += 1.0
        with torch.no_grad():
            Z = self.model(X_new, bias_new, X_mask, time_stamps_new)
            Z = Z.squeeze(0)
        Z = self.y_scaler.inv_transform([Z])[0]
        return Z.to(src_device)

    def fit(
        self,
        X: RawDataset,
        y: Optional[RawDataset] = None,
        time_stamps: Optional[RawDataset] = None,
    ) -> None:
        """Train the target model on given dataset.

        Parameters
        ----------
        X : RawDataset, (L, N, M)
            Input on witch the model is trained

        y : Optional[RawDataset], optional
            Target, by default None
        """
        num_in_feats = self.infer_num_in_feats(X)
        if y is not None:
            num_out_feats = self.infer_num_out_feats(y)
        else:
            num_out_feats = self.infer_num_out_feats(X)
        meta_info: Dict[str, Any] = {
            "num_in_feats": num_in_feats,
            "num_out_feats": num_out_feats,
        }
        model = self.build_model(num_in_feats, num_out_feats)
        losses = self.build_losses()
        metrics = self.build_metrics()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)
        (
            X_train,
            X_valid,
            y_train,
            y_valid,
            time_stamps_train,
            time_stamps_valid,
        ) = self.split_train_and_valid_data(X, y, time_stamps)
        X_scaler = self.build_scaler(X_train)
        if y_train[0] is not None:
            y_scaler = self.build_scaler(y_train)  # type: ignore
        else:
            y_scaler = self.build_scaler(X_train)
        X_train = X_scaler.transform(X_train)
        X_valid = X_scaler.transform(X_valid)
        if y_train[0] is not None and y_valid[0] is not None:
            y_train = y_scaler.transform(y_train)  # type: ignore
            y_valid = y_scaler.transform(y_valid)  # type: ignore
        meta_info["X_scaler"] = X_scaler.meta_info
        meta_info["y_scaler"] = y_scaler.meta_info
        train_dataset = self.build_train_dataset(X_train, y_train, time_stamps_train)
        valid_dataset = self.build_valid_dataset(X_valid, y_valid, time_stamps_valid)
        collator = self.build_collator()
        train_dataloader = self.build_train_dataloader(train_dataset, collator)
        valid_dataloader = self.build_valid_dataloader(valid_dataset, collator)
        logger = self.build_logger(model, losses, metrics, meta_info)
        trainer = self.build_trainer(
            model,
            losses,
            metrics,
            optimizer,
            scheduler,
            train_dataloader,
            valid_dataloader,
            y_scaler,
        )
        num_epochs = self.cfg.TRAINING.NUM_EPOCHS
        for i in range(num_epochs):
            (ave_loss_vals, ave_scores) = trainer.step()
            logger.log(i, ave_loss_vals, ave_scores)
