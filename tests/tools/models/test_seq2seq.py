import unittest
from unittest import TestCase

import torch
from tsts.tools.models import Seq2Seq


class TestSeq2Seq(TestCase):
    """Test Seq2Seq."""

    def test_resnet18(self) -> None:
        model = Seq2Seq(2, 2)
        mb_series = torch.randn(1, 8, 2)
        mb_preds = model(mb_series)
        self.assertEqual(mb_preds.size(), torch.Size([1, 1, 2]))


if __name__ == "__main__":
    unittest.main()
