from typing import List

from tsts.cfg import CfgNode as CN
from tsts.core import LOGGERS, ContextManager
from tsts.losses import Loss
from tsts.metrics import Metric
from tsts.models import Module

from .logger import Logger

__all__ = ["build_logger"]


def build_logger(
    model: Module,
    local_scaler: Module,
    losses: List[Loss],
    metrics: List[Metric],
    context_manager: ContextManager,
    cfg: CN,
) -> Logger:
    logger_name = cfg.LOGGER.NAME
    cls = LOGGERS[logger_name]
    logger = cls.from_cfg(
        model,
        local_scaler,
        losses,
        metrics,
        context_manager,
        cfg,
    )
    return logger
