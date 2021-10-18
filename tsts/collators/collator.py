from typing import List, Optional

import torch
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import COLLATORS
from tsts.types import Batch, RawBatch

__all__ = ["Collator"]


@COLLATORS.register()
class Collator(object):
    """Basic collator.

    Padding works as the following example:

    If lookback is 4 but the length of input time series is 2 e.g. [1, 2] (we remove feature
    dimension for simplicity), it adds 0 paddings to the left i.e. [0, 0, 1, 2]. Instead of it, if
    horizon is 4 but the length of output time series is 2 e.g. [1, 2], it adds 0 paddings to the
    right i.e. [1, 2, 0, 0]. masks are used to specify which time steps are valid (not padded).
    For the examples above, masks for padded input time series and output time series are [0, 0, 1,
    1] and [1, 1, 0, 0].

    Parameters
    ----------
    lookback : int, optional
        Number of input time steps, by default 100

    horizon : int, optional
        Number of output time steps, by default 1
    """

    def __init__(
        self,
        lookback: int = 100,
        horizon: int = 1,
    ) -> None:
        self.lookback = lookback
        self.horizon = horizon

    @classmethod
    def from_cfg(cls, cfg: CN) -> "Collator":
        """Build Collator from config.

        Returns
        -------
        Collator
            Built collator
        """
        lookback = cfg.IO.LOOKBACK
        horizon = cfg.IO.HORIZON
        collator = cls(lookback, horizon)
        return collator

    def __call__(self, batch: RawBatch) -> Batch:
        """Return structured batch.

        It masks time series which is shorter than lookback or horizon by 0.

        Parameters
        ----------
        batch : RawBatch
            Non-padded time series

        Returns
        -------
        Batch
            Tuple of 0-padded input time series, 0-padded output time series and these masks.
        """
        X_new = []
        y_new = []
        bias_new = []
        X_mask = []
        y_mask = []
        X_inv_transforms = []
        y_inv_transforms = []
        time_stamps_new: List[Optional[Tensor]] = []  # type: ignore
        for (X, y, bias, time_stamps, X_inv_transform, y_inv_transform) in batch:
            (X_num_steps, X_num_feats) = X.size()
            X_num_steps = min(X_num_steps, self.lookback)
            (y_num_steps, y_num_feats) = y.size()
            y_num_steps = min(y_num_steps, self.horizon)
            _X_new = X.new_zeros((self.lookback, X_num_feats))
            _bias_new = bias.new_zeros((self.lookback, y_num_feats))
            # NOTE: X could be longer than lookback
            _X_new[-X_num_steps:] = X[-X_num_steps:]
            _bias_new[-X_num_steps:] = bias[-X_num_steps:]
            _y_new = y.new_zeros((self.horizon, y_num_feats))
            # NOTE: y could be longer than horizon
            _y_new[:y_num_steps] = y[:y_num_steps]
            _X_mask = X.new_zeros((self.lookback, X_num_feats))
            _X_mask[-X_num_steps:] += 1.0
            _y_mask = y.new_zeros((self.horizon, y_num_feats))
            _y_mask[:y_num_steps] += 1.0
            if time_stamps is not None:
                time_stamps_size = time_stamps.size(-1)
                _time_stamps_new = y.new_zeros(
                    (self.lookback + self.horizon, time_stamps_size),
                    dtype=torch.long,
                )
                _time_stamps_new[
                    self.lookback - X_num_steps : self.lookback
                ] = time_stamps[:X_num_steps]
                _time_stamps_new[
                    self.lookback : self.lookback + y_num_steps
                ] = time_stamps[X_num_steps:]
                time_stamps_new.append(_time_stamps_new)
            else:
                time_stamps_new.append(None)
            X_new.append(_X_new)
            y_new.append(_y_new)
            bias_new.append(_bias_new)
            X_mask.append(_X_mask)
            y_mask.append(_y_mask)
            X_inv_transforms.append(X_inv_transform)
            y_inv_transforms.append(y_inv_transform)
        return (
            torch.stack(X_new),
            torch.stack(y_new),
            torch.stack(bias_new),
            torch.stack(X_mask),
            torch.stack(y_mask),
            torch.stack(time_stamps_new)  # type: ignore
            if time_stamps_new[0] is not None
            else time_stamps,
            X_inv_transforms,
            y_inv_transforms,
        )
