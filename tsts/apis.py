import glob
import os
from typing import List, Tuple

import pandas as pd
import torch
from torch import Tensor

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
    filename: str,
    in_feats: List[str] = [],
    out_feats: List[str] = [],
) -> Tuple[Tensor, Tensor]:
    df = pd.read_csv(filename)
    df = df.fillna(0.0)
    # Take only the values of input & output variables
    if len(in_feats) > 0:
        df = df[in_feats]
    if len(out_feats) > 0:
        df = df[out_feats]
    X = torch.tensor(df.values, dtype=torch.float32)
    y = torch.tensor(df.values, dtype=torch.float32)
    return (X, y)


def build_scalers(
    cfg: CN,
    train_dir: str,
    in_feats: List[str] = [],
    out_feats: List[str] = [],
) -> Tuple[Scaler, Scaler]:
    X_scaler = build_X_scaler(cfg)
    y_scaler = build_y_scaler(cfg)
    X = []
    y = []
    for filename in glob.glob(os.path.join(train_dir, "*.csv")):
        # Initialize input & output values
        (_X, _y) = load_sample(filename, in_feats, out_feats)
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
