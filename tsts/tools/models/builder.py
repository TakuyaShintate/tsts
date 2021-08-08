import typing
from typing import Any, Dict

from torch.nn import Module
from tsts.core import MODELS

__all__ = ["build_model"]


def build_model(cfg: Dict[str, Any]) -> Module:
    cls_name = typing.cast(str, cfg.get("name"))
    args = cfg.get("args")
    cls = MODELS[cls_name]
    return cls(**args)
