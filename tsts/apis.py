import glob
import os
from typing import List, Tuple, Union

import pandas as pd
import torch
from torch import Tensor
from torch.autograd import Variable

from .cfg import CfgNode as CN
from .cfg import get_cfg_defaults
from .scalers import Scaler, build_X_scaler, build_y_scaler
from .solvers import TimeSeriesForecaster

__all__ = [
    "load_cfg",
    "load_sample",
    "build_scalers",
    "init_forecaster",
    "run_forecaster",
]


def load_cfg(cfg_name: str) -> CN:
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_name)
    return cfg


def load_sample(
    cfg: CN,
    filename: str,
    in_feats: List[str] = [],
    out_feats: List[str] = [],
) -> Tuple[Union[Tensor, None], ...]:
    """Load time series sample.

    Notes
    -----
    When in_feats or out_feats, it tries to load all features. If the sample length is too short,
    return None.

    Parameters
    ----------
    cfg : CN
        Config

    filename : str
        Sample file name. Must be a CSV file.

    in_feats : List[str], optional
        List of input features, by default []

    out_feats : List[str], optional
        List of output features, by default []

    Returns
    -------
    Tuple[Union[Tensor, None], ...]
        Input and target pair.
    """
    df = pd.read_csv(filename)
    df = df.fillna(0.0)
    # Take only the values of input & output variables
    if len(in_feats) > 0:
        X = df[in_feats].values
    if len(out_feats) > 0:
        y = df[out_feats].values
    if len(X) < cfg.DATASET.BASE_START_INDEX + cfg.DATASET.BASE_END_INDEX + 1:
        print(f"{filename} is smaller than the minimum size, so skipped")
        return (None, None)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def build_scalers(
    cfg: CN,
    train_dir: str,
    in_feats: List[str] = [],
    out_feats: List[str] = [],
) -> Tuple[Scaler, Scaler]:
    """Return input and output scalers.

    Example
    -------
    The following example shows how to initialize scalers by loading the config named "
    my_baseline_model.yml" and using the samples in "data/train". In this example, the
    columns named "x" and "y" are used as input and output variables.

    >>> from tsts.apis import build_scalers, load_cfg
    >>> cfg = load_cfg("my_baseline_model.yml")
    >>> (X_scalers, y_scalers) = build_scalers(cfg, "data/train", ["x"], ["y"])

    Notes
    -----
    Samples that are too short are removed.

    Parameters
    ----------
    cfg : CN
        Config

    train_dir : str
        Directory where samples used for the training are stored.

    in_feats : List[str], optional
        List of input features, by default []

    out_feats : List[str], optional
        List of output features, by default []

    Returns
    -------
    Tuple[Scaler, Scaler]
        Input and output scalers
    """
    X_scaler = build_X_scaler(cfg)
    y_scaler = build_y_scaler(cfg)
    X = []
    y = []
    for filename in glob.glob(os.path.join(train_dir, "*.csv")):
        # Initialize input & output values
        (_X, _y) = load_sample(cfg, filename, in_feats, out_feats)
        if _X is None or _y is None:
            continue
        X.append(_X)
        y.append(_y)
    X_scaler.fit_batch(X)
    y_scaler.fit_batch(y)
    return (X_scaler, y_scaler)


def init_forecaster(
    cfg_name: str,
    train_dir: str,
    in_feats: List[str],
    out_feats: List[str],
) -> Tuple[TimeSeriesForecaster, Scaler, Scaler]:
    cfg = load_cfg(cfg_name)
    (X_scaler, y_scaler) = build_scalers(
        cfg,
        train_dir,
        in_feats,
        out_feats,
    )
    solver = TimeSeriesForecaster(cfg_name)
    return (solver, X_scaler, y_scaler)


def run_forecaster(
    solver: TimeSeriesForecaster,
    X_scaler: Scaler,
    y_scaler: Scaler,
    X: Tensor,
) -> Tensor:
    X = X_scaler.transform(X)
    Z = solver.predict(X)
    Z = y_scaler.inv_transform(Z)
    return Z


def get_activation_map(
    solver: TimeSeriesForecaster,
    X_scaler: Scaler,
    X: Tensor,
    tgt_time_step: int,
    tgt_var: int,
) -> Tensor:
    X = X_scaler.transform(X)
    X = Variable(X, requires_grad=True)
    Z = solver.predict(X)
    Z[tgt_time_step, tgt_var].backward()
    return X.grad
