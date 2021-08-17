from typing import Type

from torch.utils.data import DataLoader as _DataLoader
from tsts.cfg import CfgNode as CN
from tsts.core import DATALOADERS
from tsts.datasets import Dataset

__all__ = ["DataLoader"]


@DATALOADERS.register()
class DataLoader(_DataLoader):
    @classmethod
    def from_cfg(
        cls,
        dataset: Type[Dataset],
        image_set: str,
        cfg: CN,
    ) -> "DataLoader":
        """Build dataloader from config.

        Parameters
        ----------
        dataset : Dataset
            Target dataset

        image_set : str
            Dataset type

        cfg : CN
            Global configuration
        """
        if image_set == "train":
            batch_size = cfg.DATALOADER.BATCH_SIZE_TRAIN
            shuffle = cfg.DATALOADER.SHUFFLE_TRAIN
        elif image_set == "val":
            batch_size = cfg.DATALOADER.BATCH_SIZE_VAL
            shuffle = cfg.DATALOADER.SHUFFLE_VAL
        else:
            raise ValueError
        dataloader = cls(
            dataset,  # type:ignore
            batch_size,
            shuffle=shuffle,
        )
        return dataloader
