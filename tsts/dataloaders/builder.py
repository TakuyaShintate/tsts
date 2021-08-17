from tsts.cfg import CfgNode as CN
from tsts.core import DATALOADERS
from tsts.datasets import Dataset

from .dataloader import DataLoader

__all__ = ["build_dataloader"]


def build_dataloader(
    dataset: Dataset,
    image_set: str,
    cfg: CN,
) -> DataLoader:
    if image_set == "train":
        dataloader_name = cfg.DATALOADER.NAME_TRAIN
    else:
        dataloader_name = cfg.DATALOADER.NAME_VAL
    cls = DATALOADERS[dataloader_name]
    dataloader = cls.from_cfg(dataset, image_set, cfg)
    return dataloader
