import json
import os
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from tsts.core import SCALERS
from tsts.types import RawDataset

from .solver import Solver

__all__ = ["Forecaster"]


class Forecaster(Solver):
    """Tool to solve time series forecasting."""

    def predict(self, X: RawDataset) -> Tensor:
        log_dir = self.cfg.LOGGER.LOG_DIR
        if os.path.exists(log_dir) is False:
            FileNotFoundError(f"{log_dir} not found")
        meta_info_path = os.path.join(log_dir, "meta.json")
        with open(meta_info_path, "r") as f:
            meta_info = json.load(f)
        num_in_feats = meta_info["num_in_feats"]
        num_out_feats = meta_info["num_out_feats"]
        model = self.build_model(num_in_feats, num_out_feats)
        model.eval()
        scaler_name = self.cfg.SCALER.NAME
        scaler = SCALERS[scaler_name](cfg=self.cfg, **meta_info["scaler"])
        X_new = scaler.transform(X)
        test_dataset = self.build_test_dataset(X_new)
        collator = self.build_collator()
        horizon = self.cfg.IO.HORIZON
        num_instances = len(test_dataset) + horizon - 1
        Z_total = torch.zeros((num_instances, num_out_feats))
        device = self.cfg.DEVICE
        Z_total = Z_total.to(device)
        for i in range(len(test_dataset)):
            (X_new, _, X_mask, _) = collator((test_dataset[i],))
            X_new = X_new.to(device)
            X_mask = X_mask.to(device)
            with torch.no_grad():
                Z = model(X_new, X_mask)
                Z = Z.squeeze(0)
            if i > 0:
                Z_total[i + horizon - 1] += Z[horizon - 1]
                Z_total[i : i + horizon - 1] += Z[: horizon - 1]
                Z_total[i : i + horizon - 1] /= 2.0
            else:
                Z_total[i : i + horizon] += Z
        Z_total = scaler.inv_transform(Z_total)
        return Z_total

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
        num_in_feats = self.infer_num_in_feats(X)
        num_out_feats = self.infer_num_out_feats(y if y is not None else X)
        meta_info: Dict[str, Any] = {
            "num_in_feats": num_in_feats,
            "num_out_feats": num_out_feats,
        }
        model = self.build_model(num_in_feats, num_out_feats)
        losses = self.build_losses()
        metrics = self.build_metrics()
        optimizer = self.build_optimizer(model)
        (X_train, X_val, y_train, y_val) = self.split_train_and_val_data(X, y)
        # Scale with y_train
        scaler = self.build_scaler(y_train if y_train is not None else X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        if y_train is not None and y_val is not None:
            y_train = scaler.transform(y_train)
            y_val = scaler.transform(y_val)
        meta_info["scaler"] = scaler.meta_info
        train_dataset = self.build_train_dataset(X_train, y_train)
        val_dataset = self.build_val_dataset(X_val, y_val)
        collator = self.build_collator()
        train_dataloader = self.build_train_dataloader(train_dataset, collator)
        val_dataloader = self.build_val_dataloader(val_dataset, collator)
        logger = self.build_logger(model, losses, metrics, meta_info)
        trainer = self.build_trainer(
            model,
            losses,
            metrics,
            optimizer,
            train_dataloader,
            val_dataloader,
            scaler,
        )
        num_epochs = self.cfg.TRAINING.NUM_EPOCHS
        for i in range(num_epochs):
            (ave_loss_vals, ave_scores) = trainer.step()
            logger.log(i, ave_loss_vals, ave_scores)
