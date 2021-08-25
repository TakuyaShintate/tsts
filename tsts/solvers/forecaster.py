from typing import Any, Dict, Optional

import torch
from torch import Tensor
from tsts.types import RawDataset

from .solver import Solver

__all__ = ["TimeSeriesForecaster"]


class TimeSeriesForecaster(Solver):
    """Tool to solve time series forecasting."""

    def predict(self, X: Tensor) -> Tensor:
        src_device = X.device
        device = self.cfg.DEVICE
        X = X.to(device)
        lookback = self.cfg.IO.LOOKBACK
        num_in_feats = self.meta_info["num_in_feats"]
        X_new = torch.zeros((1, lookback, num_in_feats))
        X_new = X_new.to(device)
        X_new[:, -X.size(0) :] = X[-lookback:]
        X_new = self.scaler.transform([X_new])[0]
        X_mask = torch.zeros((1, lookback, num_in_feats))
        X_mask = X_mask.to(device)
        X_mask[:, -X.size(0) :] += 1.0
        with torch.no_grad():
            Z = self.model(X_new, X_mask)
            Z = Z.squeeze(0)
        Z = self.scaler.inv_transform([Z])[0]
        return Z.to(src_device)

    def fit(
        self,
        X: RawDataset,
        y: Optional[RawDataset] = None,
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
        (X_train, X_valid, y_train, y_valid) = self.split_train_and_valid_data(X, y)
        if y_train[0] is not None:
            scaler = self.build_scaler(y_train)  # type: ignore
        else:
            scaler = self.build_scaler(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        if y_train[0] is not None and y_valid[0] is not None:
            y_train = scaler.transform(y_train)  # type: ignore
            y_valid = scaler.transform(y_valid)  # type: ignore
        meta_info["scaler"] = scaler.meta_info
        train_dataset = self.build_train_dataset(X_train, y_train)
        valid_dataset = self.build_valid_dataset(X_valid, y_valid)
        collator = self.build_collator()
        train_dataloader = self.build_train_dataloader(train_dataset, collator)
        valid_dataloader = self.build_valid_dataloader(valid_dataset, collator)
        logger = self.build_logger(model, losses, metrics, meta_info)
        trainer = self.build_trainer(
            model,
            losses,
            metrics,
            optimizer,
            train_dataloader,
            valid_dataloader,
            scaler,
        )
        num_epochs = self.cfg.TRAINING.NUM_EPOCHS
        for i in range(num_epochs):
            (ave_loss_vals, ave_scores) = trainer.step()
            logger.log(i, ave_loss_vals, ave_scores)
