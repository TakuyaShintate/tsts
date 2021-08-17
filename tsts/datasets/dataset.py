from typing import Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset as _Dataset
from tsts.cfg import CfgNode as CN
from tsts.core import DATASETS
from tsts.utils import infer_dataset_type

__all__ = ["Dataset"]


@DATASETS.register()
class Dataset(_Dataset):
    """Basic dataset.

    TODO: Add transform

    Parameters
    ----------
    X : Tensor (L, M, N) or (M, N)
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
        lookback: int = 100,
        horizon: int = 1,
    ) -> None:
        self.X = X
        self.y = y
        self.lookback = lookback
        self.horizon = horizon

    @classmethod
    def from_cfg(
        cls,
        X: Tensor,
        y: Optional[Tensor],
        image_set: str,
        cfg: CN,
    ) -> "Dataset":
        lookback = cfg.IO.LOOKBACK
        horizon = cfg.IO.HORIZON
        dataset = cls(
            X,
            y,
            lookback,
            horizon,
        )
        return dataset

    def __len__(self) -> int:
        num_instances = len(self.X)
        # No zero padding at the beginning
        num_instances -= self.lookback
        # No zero padding at the end
        num_instances -= self.horizon
        return num_instances

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        dataset_type = infer_dataset_type(self.X)
        if dataset_type == "mn":
            start = i
            mid = i + self.lookback
            end = i + self.lookback + self.horizon
            X = self.X[start:mid]
            if self.y is not None:

                y = self.y[mid:end]
            else:
                y = self.X[mid:end]
        else:
            raise NotImplementedError
        return (X, y)
