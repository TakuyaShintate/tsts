"""Train a forecaster on a given train data.

Example
-------
The following example shows how to launch a new training.

python tools/train.py \
    --cfg-name config \
    --train-dir train-dir \
    --valid-dir valid-dir \
    --in-feats a b \
    --out-feats c d \
"""

import glob
from argparse import ArgumentParser, Namespace
from pathlib import Path
from textwrap import wrap
from typing import List, Tuple

import pandas as pd
import torch
from terminaltables import SingleTable
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.cfg import get_cfg_defaults
from tsts.solvers import TimeSeriesForecaster


class ColorText(object):
    RED = "\033[31m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    END = "\033[0m"


LOG = "[" + ColorText.GREEN + "log" + ColorText.END + "]"
ERROR = "[" + ColorText.RED + "error" + ColorText.END + "]"
WARNING = "[" + ColorText.BLUE + "warning" + ColorText.END + "]"


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train forecasting model")
    parser.add_argument("--cfg-name", help="config file path")
    parser.add_argument("--train-dir", help="directory containing train samples")
    parser.add_argument("--valid-dir", help="directory containing extra valid samples")
    parser.add_argument("--in-feats", type=str, nargs="+", help="input features")
    parser.add_argument("--out-feats", type=str, nargs="+", help="output features")
    parser.add_argument("--lagging", action="store_true")
    args = parser.parse_args()
    return args


def show_info(args: Namespace) -> None:
    """Print given command line arguments.

    Parameters
    ----------
    args : Namespace
        Command line arguments
    """
    color_text = ColorText()
    try:
        table_data: List[List[str]] = []
        title = color_text.RED + "exp info" + color_text.END
        table = SingleTable(table_data, title)
        # Config
        row_name = color_text.BLUE + "config" + color_text.END
        table_data.append([row_name, " "])
        max_width = table.column_max_width(1) - 5
        text = str(Path(args.cfg_name).resolve())
        text = "\n".join(wrap(text, max_width))
        table.table_data[0][1] = text
        # Input features
        row_name = color_text.BLUE + "in feats" + color_text.END
        table_data.append([row_name, " "])
        max_width = table.column_max_width(1) - 5
        text = " ".join(args.in_feats)
        text = "\n".join(wrap(text, max_width))
        table.table_data[1][1] = text
        # Output features
        row_name = color_text.BLUE + "out feats" + color_text.END
        table_data.append([row_name, " "])
        max_width = table.column_max_width(1) - 5
        text = " ".join(args.out_feats)
        text = "\n".join(wrap(text, max_width))
        table.table_data[2][1] = text
        # Other settings
        table.inner_heading_row_border = False
        table.inner_row_border = True
        table.justify_columns = {0: "center", 1: "center"}
        print(table.table)
    except ValueError:
        print("Unexpected error happens")


def load_cfg(cfg_name: str) -> CN:
    print(f"{LOG} loading config...")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_name)
    print(f"{LOG} finished loading config!")
    return cfg


def load_timeseries_data(
    cfg: CN,
    args: Namespace,
    target_dir: str,
    lookback: int,
    horizon: int,
) -> Tuple[List[Tensor], ...]:
    X = []
    Y = []
    if len(glob.glob(f"{target_dir}/*.csv")) == 0:
        raise ValueError(f"{ERROR} found no files in {target_dir}")
    for path in glob.glob(f"{target_dir}/*.csv"):
        df = pd.read_csv(path)
        df = df.fillna(0.0)
        x = df[args.in_feats]
        y = df[args.out_feats]
        x = x.values
        y = y.values
        if len(x) < cfg.IO.HORIZON + 1:
            print(f"{WARNING} {path} is smaller than the minimum size, so skipped")
            continue
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        if args.lagging is True:
            x_pad = torch.zeros((horizon, x.shape[-1]))
            y_pad = torch.zeros((lookback, len(args.out_feats)))
            X.append(torch.cat([x, x_pad]))
            Y.append(torch.cat([y_pad, y]))
        else:
            X.append(x)
            Y.append(y)
    # No valid files
    if len(X) == 0:
        raise ValueError(f"{ERROR} found no valid files in {target_dir}")
    return (X, Y)


def main() -> None:
    args = parse_args()
    show_info(args)
    cfg = load_cfg(args.cfg_name)
    lookback = cfg.IO.LOOKBACK
    horizon = cfg.IO.HORIZON
    # Load train data
    print(f"{LOG} loading train data...")
    (X_train, Y_train) = load_timeseries_data(
        cfg,
        args,
        args.train_dir,
        lookback,
        horizon,
    )
    print(f"{LOG} finished loading train data!")
    # Load valid data
    print(f"{LOG} loading validation data...")
    (X_valid, Y_valid) = load_timeseries_data(
        cfg,
        args,
        args.valid_dir,
        lookback,
        horizon,
    )
    print(f"{LOG} finished loading validation data!")
    print(f"{LOG} loading solver...")
    solver = TimeSeriesForecaster(args.cfg_name, override=True)
    print(f"{LOG} finished loading solver!")
    print(f"{LOG} finished initialization!")
    solver.fit(
        X_train,
        Y_train,
        X_valid=X_valid,
        y_valid=Y_valid,
    )


if __name__ == "__main__":
    main()
