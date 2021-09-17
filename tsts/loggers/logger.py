import json
import os
import uuid
from typing import Any, Dict, List

import torch
from tsts.cfg import CfgNode as CN
from tsts.core import LOGGERS, ContextManager
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
        local_scaler: Module,
        losses: List[Loss],
        metrics: List[Metric],
        context_manager: ContextManager,
    ) -> None:
        self.log_dir = log_dir
        self.model = model
        self.local_scaler = local_scaler
        self.losses = losses
        self.metrics = metrics
        self.context_manager = context_manager
        self._init_internal_state()
        self._init_log_dir()
        self._dump_meta_info()

    @classmethod
    def from_cfg(
        cls,
        model: Module,
        local_scaler: Module,
        losses: List[Loss],
        metrics: List[Metric],
        context_manager: ContextManager,
        cfg: CN,
    ) -> "Logger":
        log_dir = cfg.LOGGER.LOG_DIR
        if log_dir == "auto":
            log_dir = str(uuid.uuid4())
        logger = cls(
            log_dir,
            model,
            local_scaler,
            losses,
            metrics,
            context_manager,
        )
        return logger

    def _init_internal_state(self) -> None:
        self.best_ave_score = float("inf")

    def _init_log_dir(self) -> None:
        if os.path.exists(self.log_dir) is False:
            os.mkdir(self.log_dir)

    def _dump_meta_info(self) -> None:
        meta_info_file = os.path.join(self.log_dir, "meta.json")
        meta_info = {
            "num_in_feats": self.context_manager["num_in_feats"],
            "num_out_feats": self.context_manager["num_out_feats"],
        }
        with open(meta_info_file, "w") as f:
            json.dump(meta_info, f)

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
            model_path = os.path.join(self.log_dir, "model.pth")
            torch.save(self.model.state_dict(), model_path)
            local_scaler_path = os.path.join(self.log_dir, "local_scaler.pth")
            torch.save(self.local_scaler.state_dict(), local_scaler_path)
        # Add new record to log file
        record: Dict[str, Any] = {
            "epoch": epoch,
            "loss": {},
            "best": self.best_ave_score,
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
