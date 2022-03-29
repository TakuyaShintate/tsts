from typing import Callable, Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset as _Dataset
from tsts.cfg import CfgNode as CN
from tsts.core import DATASETS
from tsts.scalers import Scaler, build_X_scaler, build_y_scaler
from tsts.transforms.pipeline import Pipeline

__all__ = ["Dataset"]

_DataFrame = Tuple[
    Tensor,
    Tensor,
    Tensor,
    Optional[Tensor],
    Callable,
    Callable,
]


@DATASETS.register()
class Dataset(_Dataset):
    """Basic dataset.

    Parameters
    ----------
    X : Tensor (M, N)
        Time series

    y : Tensor
        Target for the given time series

    pipeline : Pipeline
        Data augmentation pipeline

    lookback : int, optional
        Number of input time steps, by default 100

    horizon : int, optional
        Number of output time steps, by default 1
    """

    def __init__(
        self,
        X: Tensor,
        y: Tensor,
        X_scaler: Scaler,
        y_scaler: Scaler,
        pipeline: Pipeline,
        time_stamps: Optional[Tensor] = None,
        lookback: int = 100,
        horizon: int = 1,
        base_start_index: int = 0,
        base_end_index: int = -1,
    ) -> None:
        self.X = X
        self.y = y
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.pipeline = pipeline
        self.time_stamps = time_stamps
        self.lookback = lookback
        self.horizon = horizon
        self.base_start_index = base_start_index
        self.base_end_index = base_end_index

    @classmethod
    def from_cfg(
        cls,
        X: Tensor,
        y: Tensor,
        time_stamps: Optional[Tensor],
        image_set: str,
        X_scaler: Scaler,
        y_scaler: Scaler,
        pipeline: Pipeline,
        cfg: CN,
    ) -> "Dataset":
        lookback = cfg.IO.LOOKBACK
        horizon = cfg.IO.HORIZON
        base_start_index = cfg.DATASET.BASE_START_INDEX
        base_end_index = cfg.DATASET.BASE_END_INDEX
        norm_per_dataset = cfg.DATASET.NORM_PER_DATASET
        if norm_per_dataset is True:
            X_scaler = build_X_scaler(cfg)
            y_scaler = build_y_scaler(cfg)
            X_scaler.fit(X)
            y_scaler.fit(y)
        dataset = cls(
            X,
            y,
            X_scaler,
            y_scaler,
            pipeline,
            time_stamps,
            lookback,
            horizon,
            base_start_index,
            base_end_index,
        )
        return dataset

    def __len__(self) -> int:
        # For -1, every instance has target which horizon is larger than 0
        num_instances = len(self.X) - self.base_start_index
        if self.base_end_index > 0:
            num_instances -= self.base_end_index
        return num_instances

    def __getitem__(self, i: int) -> _DataFrame:
        """Return input and target corresponding to index.

        [input]

               <----  X   ---->
        ----------------------------------

        [output]

               <---- bias ----><--- y --->
        ----------------------------------

        Parameters
        ----------
        i : int
            Data index
        """
        i += self.base_start_index
        start = max(0, i - self.lookback)
        mid = i
        end = i + self.horizon
        X = self.X[start:mid]
        y = self.y[mid:end]
        bias = self.y[start:mid]
        if self.time_stamps is not None:
            time_stamps: Optional[Tensor] = self.time_stamps[start:end]
        else:
            time_stamps = None
        X = self.X_scaler.transform(X)
        y = self.y_scaler.transform(y)
        bias = self.y_scaler.transform(bias)
        # Apply data augmentation (pipeline could be empty)
        # NOTE: y and bias are not None here
        (X, y, bias, time_stamps) = self.pipeline.apply(  # type: ignore
            X,
            y,
            bias,
            time_stamps,
        )
        return (
            X,
            y,
            bias,
            time_stamps,
            self.X_scaler.inv_transform,
            self.y_scaler.inv_transform,
        )
