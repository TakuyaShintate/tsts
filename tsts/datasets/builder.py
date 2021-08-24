from typing import Optional

from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import DATASETS

from .dataset import Dataset

__all__ = ["build_dataset"]


def build_dataset(
    X: Tensor,
    y: Optional[Tensor],
    image_set: str,
    cfg: CN,
) -> Dataset:
    if image_set == "train":
        dataset_name = cfg.DATASET.NAME_TRAIN
    else:
        dataset_name = cfg.DATASET.NAME_VALID
    cls = DATASETS[dataset_name]
    dataset = cls.from_cfg(
        X,
        y,
        image_set,
        cfg,
    )
    return dataset
