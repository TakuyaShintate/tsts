from typing import Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset as _Dataset
from tsts.cfg import CfgNode as CN
from tsts.core import DATASETS

__all__ = ["Dataset"]

_DataFrame = Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]


@DATASETS.register()
class Dataset(_Dataset):
    """Basic dataset.

    TODO: Add transform

    Parameters
    ----------
    X : Tensor (M, N)
        Time series

    y : Tensor, optional
        Target for the given time series, by default None

    lookback : int, optional
        Number of input time steps, by default 100

    horizon : int, optional
        Number of output time steps, by default 1
    """

    def __init__(
        self,
        X: Tensor,
        y: Optional[Tensor] = None,
        time_stamps: Optional[Tensor] = None,
        lookback: int = 100,
        horizon: int = 1,
        base_start_index: int = 0,
    ) -> None:
        self.X = X
        self.y = y
        self.time_stamps = time_stamps
        self.lookback = lookback
        self.horizon = horizon
        self.base_start_index = base_start_index

    @classmethod
    def from_cfg(
        cls,
        X: Tensor,
        y: Optional[Tensor],
        time_stamps: Optional[Tensor],
        image_set: str,
        cfg: CN,
    ) -> "Dataset":
        lookback = cfg.IO.LOOKBACK
        horizon = cfg.IO.HORIZON
        base_start_index = cfg.DATASET.BASE_START_INDEX
        dataset = cls(
            X,
            y,
            time_stamps,
            lookback,
            horizon,
            base_start_index,
        )
        return dataset

    def __len__(self) -> int:
        # For -1, every instance has target which horizon is larger than 0
        num_instances = len(self.X) - 1 - self.base_start_index
        return num_instances

    def __getitem__(self, i: int) -> _DataFrame:
        i += self.base_start_index
        start = max(0, i - self.lookback + 1)
        mid = i + 1
        end = i + 1 + self.horizon
        X = self.X[start:mid]
        if self.y is not None:
            y = self.y[mid:end]
            bias = self.y[start:mid]
        else:
            y = self.X[mid:end]
            bias = self.X[start:mid]
        if self.time_stamps is not None:
            time_stamps: Optional[Tensor] = self.time_stamps[start:end]
        else:
            time_stamps = None
        return (X, y, bias, time_stamps)
