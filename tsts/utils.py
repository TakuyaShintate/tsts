import math
import os
import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

__all__ = ["plot", "set_random_seed"]

sns.set_theme(style="darkgrid")


def plot(
    Z: Tensor,
    y: Tensor,
    num_max_cols: int = 3,
    feat_names: Optional[List[str]] = None,
    ylabels: Optional[List[str]] = None,
    xlabels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (32, 8),
    fontsize: int = 16,
    linewidth: int = 4,
    xticks: Optional[Tensor] = None,
) -> Tuple[Figure, List[Axes]]:
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
        diff = (Z[:, i] - y[:, i]).abs()
        axes[row][col].plot(
            xticks,
            Z[:, i],
            alpha=0.8,
            label="Z",
            color="#2ca02c",
            linewidth=linewidth,
        )
        axes[row][col].plot(
            xticks,
            y[:, i],
            alpha=0.8,
            label="y",
            color="#d62728",
            linewidth=linewidth,
        )
        # Plot difference between prediction and ground truth
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
            label="diff",
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
