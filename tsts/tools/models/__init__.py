from typing import List

from tsts.tools.models.seq2seq import Seq2Seq

__all__ = ["Seq2Seq"]


def get_classifier_names() -> List[str]:
    return []


def get_forecaster_names() -> List[str]:
    return ["Seq2Seq"]
