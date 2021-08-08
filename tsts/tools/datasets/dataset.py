from typing import Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset

# from tsts.core import DATASETS

__all__ = ["ForecasterMNDataset"]


# @DATASETS.register()
class ForecasterMNDataset(Dataset):
    def __init__(
        self,
        X: Tensor,
        y: Optional[Tensor],
        lookback: int,
        horizon: int,
    ) -> None:
        self.X = X
        self.y = y
        self.lookback = lookback
        self.horizon = horizon

    def _verify_time_length(
        self,
        X: Tensor,
        lookback: int,
        horizon: int,
    ) -> None:
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
