from tsts.cfg import CfgNode as CN
from tsts.collators import Collator
from tsts.core import DATALOADERS
from tsts.datasets import Dataset

from .dataloader import DataLoader

__all__ = ["build_dataloader"]


def build_dataloader(
    dataset: Dataset,
    image_set: str,
    collator: Collator,
    cfg: CN,
) -> DataLoader:
    if image_set == "train":
        dataloader_name = cfg.DATALOADER.NAME_TRAIN
    else:
        dataloader_name = cfg.DATALOADER.NAME_VALID
    cls = DATALOADERS[dataloader_name]
    dataloader = cls.from_cfg(
        dataset,
        image_set,
        collator,
        cfg,
    )
    return dataloader
