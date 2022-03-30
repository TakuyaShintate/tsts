import glob
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from textwrap import wrap
from typing import List, Tuple

import pandas as pd
import torch
from terminaltables import SingleTable
from torch import Tensor
from torch.nn import Module
from tsts.cfg import CfgNode as CN
from tsts.cfg import get_cfg_defaults
from tsts.scalers import Scaler, build_X_scaler, build_y_scaler
from tsts.solvers import TimeSeriesForecaster
from tsts.types import MaybeTensor


class ColorText(object):
    RED = "\033[31m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    END = "\033[0m"


LOG = "[" + ColorText.GREEN + "log" + ColorText.END + "]"


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Test forecasting model")
    parser.add_argument("--cfg-name", help="config file path")
    parser.add_argument("--train-dir", help="directory containing train samples")
    parser.add_argument("--valid-dir", help="directory containing extra valid samples")
    parser.add_argument("--in-feats", type=str, nargs="+", help="input features")
    parser.add_argument("--out-feats", type=str, nargs="+", help="output features")
    parser.add_argument("--out-dir", type=str, help="output directory")
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
        # Output directory
        row_name = color_text.BLUE + "out dir" + color_text.END
        table_data.append([row_name, " "])
        max_width = table.column_max_width(1) - 5
        text = str(Path(args.out_dir).resolve())
        text = "\n".join(wrap(text, max_width))
        table.table_data[3][1] = text
        # Other settings
        table.inner_heading_row_border = False
        table.inner_row_border = True
        table.justify_columns = {0: "center", 1: "center"}
        print(table.table)
    except ValueError:
        print("Unexpected error happens")


def load_cfg(cfg_name: str) -> CN:
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_name)
    return cfg


def load_sample(args: Namespace, filename: str, cfg: CN) -> Tuple[Tensor, Tensor]:
    """Load a csv file and remove input/output features which are not used.

    Parameters
    ----------
    args : Namespace
        Command line arguments

    filename : str
        CSV filename

    Returns
    -------
    Tuple[Tensor, Tensor]
        Input/output time series
    """
    df = pd.read_csv(filename)
    df = df.fillna(0.0)
    x = torch.tensor(df[args.in_feats].values, dtype=torch.float32)
    y = torch.tensor(df[args.out_feats].values, dtype=torch.float32)
    if args.lagging is True:
        y = torch.cat([torch.zeros((cfg.IO.LOOKBACK, len(args.out_feats))), y])
    return (x, y)


def build_scalers(cfg: CN, args: Namespace) -> Tuple[Scaler, Scaler]:
    """Build input/output time series scalers

    Notes
    -----
    Scalers are built with train data. Train data needs to be saved in `train_dir`.

    Parameters
    ----------
    cfg : CN
        Config

    args : Namespace
        Command line arguments

    Returns
    -------
    Tuple[Scaler, Scaler]
        Input/output scalers
    """
    print(f"{LOG} initializing scalers...")
    X_scaler = build_X_scaler(cfg)
    y_scaler = build_y_scaler(cfg)
    X = []
    Y = []
    for filename in glob.glob(os.path.join(args.train_dir, "*.csv")):
        (x, y) = load_sample(args, filename, cfg)
        X.append(x)
        Y.append(y)
    X_scaler.fit_batch(X)
    y_scaler.fit_batch(Y)
    print(f"{LOG} finished initializing scalers!")
    return (X_scaler, y_scaler)


class ONNXModelWrapper(Module):
    def __init__(
        self,
        cfg: CN,
        model: Module,
        X_scaler: Scaler,
        y_scaler: Scaler,
    ) -> None:
        super(ONNXModelWrapper, self).__init__()
        self.lookback = cfg.IO.LOOKBACK
        self.model = model.cpu().eval()
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler

    def forward(
        self,
        X: Tensor,
        bias: MaybeTensor = None,
    ) -> Tensor:
        X_mask = torch.ones_like(X)
        X = self.X_scaler.transform(X)
        Z = self.model(X, X_mask, bias)
        Z = self.y_scaler.inv_transform(Z)
        return Z


def main() -> None:
    args = parse_args()
    show_info(args)
    cfg = load_cfg(args.cfg_name)
    (X_scaler, y_scaler) = build_scalers(cfg, args)
    solver = TimeSeriesForecaster(args.cfg_name)
    onnx_model_wrapper = ONNXModelWrapper(cfg, solver.model, X_scaler, y_scaler)
    X = torch.randn(1, cfg.IO.LOOKBACK, len(args.in_feats), requires_grad=True)
    bias = torch.randn(1, cfg.IO.HORIZON, len(args.out_feats), requires_grad=True)
    torch.onnx.export(
        onnx_model_wrapper,
        (X, bias),
        "model.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["X", "bias"],
        output_names=["Z"],
        dynamic_axes={
            "X": {0: "batch_size", 1: "time_steps", 2: "in_feats"},
            "bias": {0: "batch_size", 1: "time_steps", 2: "out_feats"},
            "Z": {0: "batch_size", 1: "time_steps", 2: "out_feats"},
        },
    )


if __name__ == "__main__":
    main()
