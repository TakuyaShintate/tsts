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
from tqdm import tqdm
from tsts.cfg import CfgNode as CN
from tsts.cfg import get_cfg_defaults
from tsts.scalers import Scaler, build_X_scaler, build_y_scaler
from tsts.solvers import TimeSeriesForecaster
from tsts.utils import plot


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
    parser.add_argument("--test-dir", help="directory containing test samples")
    parser.add_argument("--in-feats", type=str, nargs="+", help="input features")
    parser.add_argument("--out-feats", type=str, nargs="+", help="output features")
    parser.add_argument("--out-dir", type=str, help="output directory")
    parser.add_argument("--lagging", action="store_true")
    parser.add_argument("--zero-padding", action="store_true")
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
    Y_scaler = build_y_scaler(cfg)
    X = []
    Y = []
    for filename in glob.glob(os.path.join(args.train_dir, "*.csv")):
        (x, y) = load_sample(args, filename, cfg)
        X.append(x)
        Y.append(y)
    X_scaler.fit_batch(X)
    Y_scaler.fit_batch(Y)
    print(f"{LOG} finished initializing scalers!")
    return (X_scaler, Y_scaler)


def infer_step(
    args: Namespace,
    x: Tensor,
    y: Tensor,
    solver: TimeSeriesForecaster,
    X_scaler: Scaler,
    Y_scaler: Scaler,
    lookback: int,
    horizon: int,
) -> Tensor:
    """Return inference result on given input time series.

    Single input time series length is `horizon`. To do inference on longer time series, it uses `r
    olling mean` of a set of predicted time series. `rolling mean` takes average over moving window
    s. Each window corresponds to a single predicted time series. For understanding how `rolling me
    an` works, see the following example.

    1)  0.0  0.2  0.1  0.2
    2)       0.2  0.1  0.2  0.1
    3)            0.1  0.2  0.0  0.1
        ----------------------------
        0.0  0.2  0.1  0.2  0.5  0.1 <=== final prediction

    Parameters
    ----------
    args : Namespace
        Command line arguments

    x : Tensor
        Input time series (N x M: M is the number of input features)

    solver : TimeSeriesForecaster
        Pretrained model

    X_scaler : Scaler
        Input time series scaler

    Y_scaler : Scaler
        Output time series scaler (used to inverse transform predicted time series)

    lookback : int
        Lookback number

    horizon : int
        Horizon number

    Returns
    -------
    Tensor
        Output time series (N - H, M': H is the horizon and M' is the number of output features)
    """
    num_out_feats = len(args.out_feats)
    mae = 0.0
    mse = 0.0
    with torch.no_grad():
        num_steps = len(x)
        # If lagging is True, target is left shifted by lookback when it is loaded
        if args.lagging is True:
            num_target_steps = num_steps + lookback
            end_steps = num_steps - lookback
        else:
            num_target_steps = num_steps
            end_steps = num_steps - horizon
        z = torch.zeros((num_target_steps, num_out_feats), dtype=torch.float32)
        c = torch.zeros((num_target_steps,), dtype=torch.float32)
        for i in tqdm(range(lookback, end_steps)):
            # Transform input time series and reverse transform predicted time series
            x_scale = X_scaler.transform(x[i - lookback : i])
            y_scale = Y_scaler.transform(y[i : i + horizon])
            b_scale = Y_scaler.transform(y[i - lookback : i])
            pred = solver.predict(x_scale, b_scale)
            pred = pred.detach().cpu()
            z_scale = Y_scaler.inv_transform(pred)
            mae += (pred - y_scale).abs().mean().item()
            mse += ((pred - y_scale) ** 2).mean().item()
            z[i : i + horizon] += z_scale
            c[i : i + horizon] += 1.0
    print(f"{LOG} mae: {mae / (end_steps - lookback)}")
    print(f"{LOG} mse: {mse / (end_steps - lookback)}")
    z = z / c.unsqueeze(-1).clamp(min=1.0)
    # Remove last `horizon` steps
    return z


def save_result(
    args: Namespace,
    z: Tensor,
    y: Tensor,
    filename: str,
    cfg: CN,
) -> None:
    if args.zero_padding is False:
        lookback = cfg.IO.LOOKBACK
        horizon = cfg.IO.HORIZON
        z = z[lookback:-horizon]
        y = y[lookback:-horizon]
    (fig, _) = plot(
        z,
        y,
        feat_names=args.out_feats,
        linewidth=1,
        Z_label="Pred",
        y_label="GT",
        diff_label="Error",
    )
    filepath = os.path.join(args.out_dir, filename)
    fig.savefig(filepath + ".png")
    data = torch.cat([z, y, torch.abs(z - y)], dim=1)
    # Update column names
    columns = []
    for i in args.out_feats:
        columns.append(i + " " + "(pred)")
    columns.extend(args.out_feats)
    for i in args.out_feats:
        columns.append(i + " " + "(diff)")
    df = pd.DataFrame(data.detach().numpy(), columns=columns)
    df.to_csv(filepath, index=False)


def predict(
    args: Namespace,
    solver: TimeSeriesForecaster,
    X_scaler: Scaler,
    Y_scaler: Scaler,
    lookback: int,
    horizon: int,
    cfg: CN,
) -> None:
    filenames = glob.glob(os.path.join(args.test_dir, "*.csv"))
    for (i, filename) in enumerate(filenames):
        print(f"{LOG} processing {filename.split('/')[-1]} {i + 1}/{len(filenames)}...")
        (x, y) = load_sample(args, filename, cfg)
        z = infer_step(
            args,
            x,
            y,
            solver,
            X_scaler,
            Y_scaler,
            lookback,
            horizon,
        )
        save_result(
            args,
            z,
            y,
            filename.split("/")[-1],
            cfg,
        )


def main() -> None:
    args = parse_args()
    show_info(args)
    cfg = load_cfg(args.cfg_name)
    (X_scaler, Y_scaler) = build_scalers(cfg, args)
    solver = TimeSeriesForecaster(args.cfg_name)
    lookback = cfg.IO.LOOKBACK
    horizon = cfg.IO.HORIZON
    predict(
        args,
        solver,
        X_scaler,
        Y_scaler,
        lookback,
        horizon,
        cfg,
    )


if __name__ == "__main__":
    main()
