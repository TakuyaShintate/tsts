import torch
from tsts.cfg import CfgNode as CN
from tsts.core import COLLATORS
from tsts.types import Batch, RawBatch

__all__ = ["Collator"]


@COLLATORS.register()
class Collator(object):
    def __init__(self, lookback: int, horizon: int) -> None:
        self.lookback = lookback
        self.horizon = horizon

    @classmethod
    def from_cfg(cls, cfg: CN) -> "Collator":
        lookback = cfg.IO.LOOKBACK
        horizon = cfg.IO.HORIZON
        collator = cls(lookback, horizon)
        return collator

    def __call__(self, batch: RawBatch) -> Batch:
        X_new = []
        y_new = []
        X_mask = []
        y_mask = []
        for (X, y) in batch:
            (X_num_steps, X_num_feats) = X.size()
            X_num_steps = min(X_num_steps, self.lookback)
            (y_num_steps, y_num_feats) = y.size()
            y_num_steps = min(y_num_steps, self.horizon)
            _X_new = X.new_zeros((self.lookback, X_num_feats))
            # NOTE: X could be longer than lookback
            _X_new[-X_num_steps:] = X[-X_num_steps:]
            _y_new = y.new_zeros((self.horizon, y_num_feats))
            # NOTE: y could be longer than horizon
            _y_new[:y_num_steps] = y[:y_num_steps]
            _X_mask = X.new_zeros((self.lookback, X_num_feats))
            _X_mask[-X_num_steps:] += 1.0
            _y_mask = y.new_zeros((self.horizon, y_num_feats))
            _y_mask[:y_num_steps] += 1.0
            X_new.append(_X_new)
            y_new.append(_y_new)
            X_mask.append(_X_mask)
            y_mask.append(_y_mask)
        return (
            torch.stack(X_new),
            torch.stack(y_new),
            torch.stack(X_mask),
            torch.stack(y_mask),
        )
