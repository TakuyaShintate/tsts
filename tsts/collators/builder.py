from tsts.cfg import CfgNode as CN
from tsts.core import COLLATORS

from .collator import Collator

__all__ = ["build_collator"]


def build_collator(cfg: CN) -> Collator:
    """Build collator.

    To configure collator, change lines which start with "_CN.COLLATOR". For other settings not
    written in global configuration, refer to `from_cfg method` of each class.

    Parameters
    ----------
    cfg : CN
        Global configuration

    Returns
    -------
    Collator
        Built collator
    """
    collator_name = cfg.COLLATOR.NAME
    cls = COLLATORS[collator_name]
    collator = cls.from_cfg(cfg)
    return collator
