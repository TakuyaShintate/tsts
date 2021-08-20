from tsts.cfg import CfgNode as CN
from tsts.core import SCALERS
from tsts.types import RawDataset

from .scaler import Scaler

__all__ = ["build_scaler"]


def build_scaler(X_or_y: RawDataset, cfg: CN) -> Scaler:
    scaler_name = cfg.SCALER.NAME
    cls = SCALERS[scaler_name]
    scaler = cls.from_cfg(X_or_y, cfg)
    return scaler
