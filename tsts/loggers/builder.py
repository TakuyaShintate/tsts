from typing import Any, Dict, List

from tsts.cfg import CfgNode as CN
from tsts.core import LOGGERS
from tsts.losses import Loss
from tsts.metrics import Metric
from tsts.models import Module

from .logger import Logger

__all__ = ["build_logger"]


def build_logger(
    model: Module,
    losses: List[Loss],
    metrics: List[Metric],
    meta_info: Dict[str, Any],
    cfg: CN,
) -> Logger:
    logger_name = cfg.LOGGER.NAME
    cls = LOGGERS[logger_name]
    logger = cls.from_cfg(
        model,
        losses,
        metrics,
        meta_info,
        cfg,
    )
    return logger
