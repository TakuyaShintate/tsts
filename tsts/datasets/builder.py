from typing import Optional, Type

from tsts.cfg import CfgNode as CN
from tsts.core import DATASETS
from tsts.types import RawDataset

from .dataset import Dataset

__all__ = ["build_dataset"]


def build_dataset(
    X: RawDataset,
    y: Optional[RawDataset],
    image_set: str,
    cfg: CN,
) -> Type[Dataset]:
    if image_set == "train":
        dataset_name = cfg.DATASET.NAME_TRAIN
    else:
        dataset_name = cfg.DATASET.NAME_VAL
    cls = DATASETS[dataset_name]
    dataset = cls.from_cfg(
        X,
        y,
        image_set,
        cfg,
    )
    return dataset
