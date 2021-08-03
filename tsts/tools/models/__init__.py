from typing import List

from tsts.tools.models.seq2seq import seq2seq

__all__ = ["seq2seq"]


def get_classifier_names() -> List[str]:
    return []


def get_forecaster_names() -> List[str]:
    return ["seq2seq"]
