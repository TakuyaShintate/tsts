from typing import List

from tsts.cfg import CfgNode as CN
from tsts.core import METRICS

from .metric import Metric

__all__ = ["build_metrics"]


def build_metrics(cfg: CN) -> List[Metric]:
    metrics = []
    num_metrics = len(cfg.METRICS.NAMES)
    for i in range(num_metrics):
        metric_name = cfg.METRICS.NAMES[i]
        cls = METRICS[metric_name]
        metric_args = cfg.METRICS.ARGS[i]
        metric = cls(**metric_args)
        metrics.append(metric)
    return metrics
