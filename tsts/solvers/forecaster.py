import json
import os
from typing import Any, Dict, Optional

from tsts.core import SCALERS
from tsts.models import Module
from tsts.scalers import Scaler
from tsts.types import RawDataset

from .solver import Solver

__all__ = ["Forecaster"]


class Forecaster(Solver):
    """Tool to solve time series forecasting."""

    def _load_meta_info(self) -> Dict[str, Any]:
        """Load meta info collected during training.

        Returns
        -------
        Dict[str, Any]
            Meta info
        """
        log_dir = self.cfg.LOGGER.LOG_DIR
        if os.path.exists(log_dir) is False:
            FileNotFoundError(f"{log_dir} not found")
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
        train_datasets = self.build_train_datasets(X_train, y_train)
        valid_datasets = self.build_valid_datasets(X_valid, y_valid)
        collator = self.build_collator()
        train_dataloaders = self.build_train_dataloaders(train_datasets, collator)
        valid_dataloaders = self.build_valid_dataloaders(valid_datasets, collator)
        logger = self.build_logger(model, losses, metrics, meta_info)
        trainer = self.build_trainer(
            model,
            losses,
            metrics,
            optimizer,
            train_dataloaders,
            valid_dataloaders,
            scaler,
        )
        num_epochs = self.cfg.TRAINING.NUM_EPOCHS
        for i in range(num_epochs):
            (ave_loss_vals, ave_scores) = trainer.step()
            logger.log(i, ave_loss_vals, ave_scores)
