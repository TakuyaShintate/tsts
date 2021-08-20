from typing import Any, Dict

from tsts.cfg import CfgNode as CN
from tsts.types import RawDataset

__all__ = ["Scaler"]


class Scaler(object):
    @classmethod
    def from_cfg(cls, X_or_y: RawDataset, cfg: CN) -> "Scaler":
        raise NotImplementedError

    @property
    def meta_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    def transform(self, X_or_y: RawDataset) -> RawDataset:
        raise NotImplementedError

    def inv_transform(self, X_or_y: RawDataset) -> RawDataset:
        raise NotImplementedError
