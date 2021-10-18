from tsts.cfg import CfgNode as CN
from tsts.core import SCALERS

from .scaler import Scaler

__all__ = ["build_X_scaler", "build_y_scaler"]


def build_X_scaler(cfg: CN) -> Scaler:
    scaler_name = cfg.X_SCALER.NAME
    cls = SCALERS[scaler_name]
    scaler = cls.from_cfg(cfg)
    return scaler


def build_y_scaler(cfg: CN) -> Scaler:
    scaler_name = cfg.Y_SCALER.NAME
    cls = SCALERS[scaler_name]
    scaler = cls.from_cfg(cfg)
    return scaler
