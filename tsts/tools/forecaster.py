from typing import Optional, Tuple

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tsts.tools.models import seq2seq
from tsts.tools.tool import _Tool

__all__ = ["Forecaster"]


MODELS = {
    "seq2seq": seq2seq,
}

# Dataset types
LMN = 0
MN = 1


class _ForecasterNMDataset(Dataset):
    def __init__(
        self, X: Tensor, y: Optional[Tensor], lookback: int, horizon: int
    ) -> None:
        self.X = X
        self.y = y
        self.lookback = lookback
        self.horizon = horizon

    def _verify_time_length(self, X: Tensor, lookback: int, horizon: int) -> None:
        pass

    def __len__(self) -> int:
        return len(self.X) - self.lookback - self.horizon

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        X = self.X[i : i + self.lookback]
        if self.y is not None:
            y = self.y[i + self.lookback : i + self.lookback + self.horizon]
        else:
            y = self.X[i + self.lookback : i + self.lookback + self.horizon]
        return (X, y)


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
        model_name: str = "seq2seq",
        lookback: int = 100,
        horizon: int = 1,
        batch_size: int = 8,
    ) -> None:
        super(Forecaster, self).__init__()
        self.model_name = model_name
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size

    def _infer_num_in_feats(self, X: Tensor) -> int:
        assert X.dim() == 2, "X has invalid shape"
        num_in_feats = X.size(1)
        return num_in_feats

    def _infer_num_out_feats(self, y: Tensor) -> int:
        num_out_feats = self._infer_num_in_feats(y)
        return num_out_feats

    def _build_forecasting_model(self, num_in_feats: int, num_out_feats: int) -> Module:
        assert self.model_name in MODELS, "model_name is invalid"
        cls = MODELS[self.model_name]
        model = cls(num_in_feats, num_out_feats, self.lookback, self.horizon)
        return model

    def _infer_dataset_type(self, X: Tensor) -> str:
        num_dims = X.dim()
        if num_dims == 2:
            return "nm"
        elif num_dims == 3:
            return "lnm"
        else:
            raise ValueError("Invalid dataset")

    def _build_dls(
        self, X: Tensor, y: Optional[Tensor]
    ) -> Tuple[DataLoader, DataLoader]:
        dataset_type = self._infer_dataset_type(X)
        if dataset_type == "nm":
            dataset = _ForecasterNMDataset(X, y, self.lookback, self.horizon)
        dl = DataLoader(dataset, self.batch_size)
        # Dummy for test
        return (dl, dl)

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
        num_out_feats = self._infer_num_out_feats(X)
        model = self._build_forecasting_model(num_in_feats, num_out_feats)
        (train_dl, val_dl) = self._build_dls(X, y)
        print(next(iter(train_dl)))
