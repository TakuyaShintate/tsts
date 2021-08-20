import json
import os
import uuid
from typing import Any, Dict, List

import torch
from tsts.cfg import CfgNode as CN
from tsts.core import LOGGERS
from tsts.losses import Loss
from tsts.metrics import Metric
from tsts.models import Module

__all__ = ["Logger"]


@LOGGERS.register()
class Logger(object):
    def __init__(
        self,
        log_dir: str,
        model: Module,
        losses: List[Loss],
        metrics: List[Metric],
        meta_info: Dict[str, Any],
    ) -> None:
        self.log_dir = log_dir
        self.model = model
        self.losses = losses
        self.metrics = metrics
        self.meta_info = meta_info
        self._init_internal_state()
        self._init_log_dir()

    @classmethod
    def from_cfg(
        cls,
        model: Module,
        losses: List[Loss],
        metrics: List[Metric],
        meta_info: Dict[str, Any],
        cfg: CN,
    ) -> "Logger":
        log_dir = cfg.LOGGER.LOG_DIR
        if log_dir == "auto":
            log_dir = str(uuid.uuid4())
        logger = cls(
            log_dir,
            model,
            losses,
            metrics,
            meta_info,
        )
        return logger

    def _init_internal_state(self) -> None:
        self.best_ave_score = float("inf")

    def _init_log_dir(self) -> None:
        if os.path.exists(self.log_dir) is False:
            os.mkdir(self.log_dir)
        meta_info_file = os.path.join(self.log_dir, "meta.json")
        with open(meta_info_file, "w") as f:
            json.dump(self.meta_info, f)

    def log(
        self,
        epoch: int,
        ave_loss_vs: List[float],
        ave_scores: List[float],
    ) -> None:
        # Update model params
        current_ave_score = sum(ave_scores) / len(ave_scores)
        if current_ave_score < self.best_ave_score:
            self.best_ave_score = current_ave_score
            root = os.path.join(self.log_dir, "model.pth")
            torch.save(self.model.state_dict(), root)
        # Add new record to log file
        record: Dict[str, Any] = {
            "epoch": epoch,
            "loss": {},
            "metric": {},
        }
        for (i, loss) in enumerate(self.losses):
            loss_name = loss.__class__.__name__
            ave_loss_v = ave_loss_vs[i]
            record["loss"][loss_name] = ave_loss_v
        for (i, metric) in enumerate(self.metrics):
            metric_name = metric.__class__.__name__
            ave_score = ave_scores[i]
            record["metric"][metric_name] = ave_score
        log_file = os.path.join(self.log_dir, "log.txt")
        with open(log_file, "a") as f:
            f.write(str(record) + "\n")
