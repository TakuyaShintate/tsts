from typing import Type

from torch.utils.data import DataLoader as _DataLoader
from tsts.cfg import CfgNode as CN
from tsts.collators import Collator
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
        collator: Collator,
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
            drop_last = cfg.DATALOADER.DROP_LAST_TRAIN
            num_workers = cfg.DATALOADER.NUM_WORKERS_TRAIN
        elif image_set == "valid":
            batch_size = cfg.DATALOADER.BATCH_SIZE_VALID
            shuffle = cfg.DATALOADER.SHUFFLE_VALID
            drop_last = cfg.DATALOADER.DROP_LAST_VALID
            num_workers = cfg.DATALOADER.NUM_WORKERS_VALID
        else:
            raise ValueError
        dataloader = cls(
            dataset,  # type:ignore
            batch_size,
            shuffle=shuffle,
            collate_fn=collator,  # type:ignore
            drop_last=drop_last,
            num_workers=num_workers,
        )
        return dataloader
