import unittest
from unittest import TestCase

import torch
from tsts.cfg import get_cfg_defaults
from tsts.models import Informer, build_model

MODEL_CFG = [
    "IO.LOOKBACK",
    8,
    "IO.HORIZON",
    1,
    "MODEL.NAME",
    "Informer",
    "MODEL.NUM_H_FEATS",
    512,
]


class TestInformer(TestCase):
    """Test Informer."""

    def test_informer(self) -> None:
        model = Informer(2, 2, 8, 1)
        X = torch.randn(1, 8, 2)
        X_mask = torch.ones_like(X)
        Z = model(X, X_mask)
        self.assertEqual(Z.size(), torch.Size([1, 1, 2]))

    def test_informer_from_cfg(self) -> None:
        cfg = get_cfg_defaults()
        cfg.merge_from_list(MODEL_CFG)
        model = build_model(2, 2, cfg)
        X = torch.randn(1, 8, 2)
        X_mask = torch.ones_like(X)
        Z = model(X, X_mask)
        self.assertEqual(Z.size(), torch.Size([1, 1, 2]))


if __name__ == "__main__":
    unittest.main()
