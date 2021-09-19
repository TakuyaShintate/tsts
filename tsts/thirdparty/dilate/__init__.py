"""
@incollection{leguen19dilate,
title = {Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models},
author = {Le Guen, Vincent and Thome, Nicolas},
booktitle = {Advances in Neural Information Processing Systems},
pages = {4191--4203},
year = {2019}
}

This module comes from https://github.com/vincent-leguen/DILATE
"""


from .path_soft_dtw import PathDTWBatch
from .soft_dtw import SoftDTWBatch, pairwise_distances

__all__ = ["PathDTWBatch", "SoftDTWBatch", "pairwise_distances"]
