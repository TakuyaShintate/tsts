import copy
from typing import Any, Dict, Optional, Tuple

from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from tsts.tools.datasets import ForecasterMNDataset
from tsts.tools.models import build_model
from tsts.tools.tool import _Tool
from tsts.tools.trainers import SupervisedTrainer

__all__ = ["Forecaster"]


DEFAULT_MODEL = "Seq2Seq"
MN = 0
LMN = 1
TRAIN_INDEX = 0
VAL_INDEX = 1
INVALID_INDEX = -1


class Forecaster(_Tool):
    """Tool to solve time series forecasting.

    Parameters
    ----------
    model_name : str, optional
        Target model name, by default "seq2seq"

    lookback : int, optional
        Number of previous timesteps used to predict the subsequent timesteps, by default 100

    horizon : int, optional
        Number of the subsequent timesteps predicted, by default 1
    """

    def __init__(
        self,
        lookback: int = 100,
        horizon: int = 1,
        batch_size: int = 32,
        train_ratio: float = 0.75,
        num_epochs: int = 100,
        model_cfg: Dict[str, Any] = {},
        dataset_cfg: Dict[str, Any] = {},
    ) -> None:
        super(Forecaster, self).__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_epochs = num_epochs
        self.model_cfg = model_cfg

    def _infer_num_in_feats(self, X: Tensor) -> int:
        assert X.dim() == 2, "X has invalid shape"
        num_in_feats = X.size(1)
        return num_in_feats

    def _infer_num_out_feats(self, y: Tensor) -> int:
        num_out_feats = self._infer_num_in_feats(y)
        return num_out_feats

    def _build_model(self, num_in_feats: int, num_out_feats: int) -> Module:
        cfg = copy.copy(self.model_cfg)
        cfg.setdefault("name", DEFAULT_MODEL)
        cfg.setdefault("args", {})
        cfg["args"].setdefault("num_in_feats", num_in_feats)
        cfg["args"].setdefault("num_out_feats", num_out_feats)
        cfg["args"].setdefault("horizon", self.horizon)
        return build_model(cfg)

    def _build_optimizer(self, model: Module) -> Adam:
        optimizer = Adam(model.parameters(), 0.002)
        return optimizer

    def _infer_dataset_type(self, X: Tensor) -> int:
        num_dims = X.dim()
        if num_dims == 2:
            return MN
        elif num_dims == 3:
            return LMN
        else:
            raise ValueError("Invalid dataset")

    def _split_train_and_val_data(
        self,
        X: Tensor,
        y: Optional[Tensor],
        dataset_type: int,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if dataset_type == MN:
            num_samples = X.size(0)
            num_train_samples = int(self.train_ratio * num_samples)
            indices = X.new_zeros(num_samples)
            offset = num_train_samples
            indices[offset + self.lookback :] += VAL_INDEX
            indices[offset : offset + self.lookback] += INVALID_INDEX
        else:
            raise NotImplementedError
        train_X = X[indices == TRAIN_INDEX]
        val_X = X[indices == VAL_INDEX]
        if y is not None:
            train_y: Optional[Tensor] = y[indices == TRAIN_INDEX]
            val_y: Optional[Tensor] = y[indices == VAL_INDEX]
        else:
            train_y = None
            val_y = None
        return (train_X, val_X, train_y, val_y)

    def _build_dls(
        self,
        X: Tensor,
        y: Optional[Tensor],
    ) -> Tuple[DataLoader, DataLoader]:
        dataset_type = self._infer_dataset_type(X)
        (train_X, val_X, train_y, val_y) = self._split_train_and_val_data(
            X,
            y,
            dataset_type,
        )
        if dataset_type == MN:
            dataset = ForecasterMNDataset
        else:
            raise NotImplementedError
        train_dataset = dataset(train_X, train_y, self.lookback, self.horizon)
        val_dataset = dataset(val_X, val_y, self.lookback, self.horizon)
        train_dl = DataLoader(train_dataset, self.batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, self.batch_size)
        return (train_dl, val_dl)

    def _build_trainer(
        self,
        model: Module,
        optimizer: Adam,
        train_dl: DataLoader,
        val_dl: DataLoader,
    ) -> SupervisedTrainer:
        trainer = SupervisedTrainer(model, optimizer, train_dl, val_dl)
        return trainer

    def fit(
        self,
        X: Tensor,
        y: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> None:
        """Train the target model on given dataset.

        Parameters
        ----------
        X : Tensor (L, N, M) or (N, M)
            Input on witch the model is trained

        y : Optional[Tensor], optional
            Target, by default None
        """
        num_in_feats = self._infer_num_in_feats(X)
        num_out_feats = self._infer_num_out_feats(y if y is not None else X)
        model = self._build_model(num_in_feats, num_out_feats)
        optimizer = self._build_optimizer(model)
        (train_dl, val_dl) = self._build_dls(X, y)
        trainer = self._build_trainer(model, optimizer, train_dl, val_dl)
        for _ in range(self.num_epochs):
            trainer.step()
