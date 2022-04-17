import math
import os
import random
import warnings
from collections import OrderedDict
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

__all__ = [
    "plot",
    "set_random_seed",
    "merge_state_dict",
]

sns.set_theme(style="darkgrid")


def plot(
    Z: Tensor,
    y: Optional[Tensor] = None,
    num_max_cols: int = 3,
    feat_names: Optional[List[str]] = None,
    ylabels: Optional[List[str]] = None,
    xlabels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (32, 8),
    fontsize: int = 16,
    linewidth: int = 4,
    xticks: Optional[Tensor] = None,
    plot_diff: bool = True,
    Z_label: str = "Z",
    y_label: str = "y",
    diff_label: str = "diff",
) -> Tuple[Figure, List[Axes]]:
    """Plot time series.

    This utility function takes 2 time series and plots them in comparable way. These time series c
    an be multivariate ones. In that case, it decomposes multivariate time series and lists them in
    dividually. `num_max_cols` determines how many time series are plotted in a row.

    Parameters
    ----------
    Z : Tensor
        First time series. Model prediction is often given

    y : Optional[Tensor], optional
        Second time series. Ground truth is often given. If None is given, it does not show second
        time series, by default None

    num_max_cols : int, optional
        Maximum number of time series shown in a row, by default 3

    feat_names : Optional[List[str]], optional
        List of feature names. They will be the titles of plots, by default None

    ylabels : Optional[List[str]], optional
        List of the labels of y axis for each plot. If None is given, no labels are shown in y axis
        , by default None

    xlabels : Optional[List[str]], optional
        List of the labels of y axis for each plot, If None is given, no labels are shown in x axis
        , by default None
    """
    num_steps = Z.size(0)
    num_out_feats = Z.size(-1)
    num_rows = max(2, math.ceil(num_out_feats / num_max_cols))
    num_cols = num_max_cols
    (fig, axes) = plt.subplots(
        num_rows,
        num_cols,
        figsize=figsize,
    )
    if xticks is None:
        xticks = torch.arange(0.0, float(num_steps), 1.0)
    for i in range(num_out_feats):
        row = i // num_max_cols
        col = i % num_max_cols
        axes[row][col].plot(
            xticks,
            Z[:, i],
            alpha=0.8,
            label=Z_label,
            color="#2ca02c",
            linewidth=linewidth,
        )
        if y is not None:
            axes[row][col].plot(
                xticks,
                y[:, i],
                alpha=0.8,
                label=y_label,
                color="#d62728",
                linewidth=linewidth,
            )
        # Plot difference between prediction and ground truth
        if y is not None and plot_diff is True:
            diff = (Z[:, i] - y[:, i]).abs()  # type: ignore
            axes[row][col].fill_between(
                xticks,
                diff,
                color="#1f77b4",
                alpha=0.8,
            )
            axes[row][col].plot(
                xticks,
                diff,
                alpha=0.8,
                label=diff_label,
                color="#1f77b4",
                linewidth=linewidth,
            )
        if feat_names is not None:
            axes[row][col].set_title(
                feat_names[i],
                fontsize=fontsize,
            )
        if ylabels is not None:
            axes[row][col].set_ylabel(
                ylabels[i],
                fontsize=fontsize,
            )
        if xlabels is not None:
            axes[row][col].set_xlabel(
                xlabels[i],
                fontsize=fontsize,
            )
        axes[row][col].legend(fontsize=fontsize)
    for i in range(num_out_feats, num_rows * num_cols):
        row = i // num_max_cols
        col = i % num_max_cols
        fig.delaxes(axes[row][col])
    fig.tight_layout()
    return (fig, axes)


def set_random_seed(seed: int = 42) -> None:
    """Enforce deterministic behavior.

    Parameters
    ----------
    seed : int, optional
        Random seed, by default 42
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge_state_dict(src: OrderedDict, tgt: OrderedDict) -> OrderedDict:
    new_state_dict = OrderedDict()
    for (k, v) in zip(src.keys(), tgt.values()):
        if v.size() == src[k].size():
            new_state_dict[k] = v
        # NOTE: This usually happens if the number of input/output vars is different from the one
        # during pretraining. In this case, randomly initialized params are used.
        else:
            warnings.warn(f"{k} does not match between src and tgt")
            new_state_dict[k] = src[k]
    return new_state_dict
