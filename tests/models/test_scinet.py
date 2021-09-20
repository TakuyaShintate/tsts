import unittest
from unittest import TestCase

import torch
from tsts.cfg import get_cfg_defaults
from tsts.models import SCINet, build_model

MODEL_CFG = [
    "IO.LOOKBACK",
    8,
    "IO.HORIZON",
    1,
    "MODEL.NAME",
    "SCINet",
]


class TestSCINet(TestCase):
    """Test SCINet."""

    def test_scinet(self) -> None:
        model = SCINet(2, 2, 8, 1)
        X = torch.randn(1, 8, 2)
        X_mask = torch.ones_like(X)
        Z = model(X, X_mask)
        self.assertEqual(Z.size(), torch.Size([1, 1, 2]))

    def test_scinet_from_cfg(self) -> None:
        cfg = get_cfg_defaults()
        cfg.merge_from_list(MODEL_CFG)
        model = build_model(2, 2, cfg)
        X = torch.randn(1, 8, 2)
        X_mask = torch.ones_like(X)
        Z = model(X, X_mask)
        self.assertEqual(Z.size(), torch.Size([1, 1, 2]))


if __name__ == "__main__":
    unittest.main()
