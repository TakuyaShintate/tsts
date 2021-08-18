from tsts.cfg import CfgNode as CN
from tsts.core import COLLATORS

from .collator import Collator

__all__ = ["build_collator"]


def build_collator(cfg: CN) -> Collator:
    collator_name = cfg.COLLATOR.NAME
    cls = COLLATORS[collator_name]
    collator = cls.from_cfg(cfg)
    return collator
