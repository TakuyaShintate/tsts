from tsts.cfg import CfgNode as CN
from tsts.core import SCALERS

from .scaler import Scaler

__all__ = ["build_scaler"]


def build_scaler(cfg: CN) -> Scaler:
    scaler_name = cfg.SCALER.NAME
    cls = SCALERS[scaler_name]
    scaler = cls.from_cfg(cfg)
    return scaler
