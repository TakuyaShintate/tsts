import os
from argparse import ArgumentParser, Namespace

import pandas as pd
import torch
from tsts.utils import plot


class ColorText(object):
    RED = "\033[31m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    END = "\033[0m"


LOG = "[" + ColorText.GREEN + "log" + ColorText.END + "]"


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Plot csv file contents")
    parser.add_argument("--file", type=str, help="path to csv file")
    parser.add_argument("--out-dir", type=str, help="output directory")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.file)
    data = torch.tensor(df.values, dtype=torch.float32)
    (fig, _) = plot(data, feat_names=df.columns)
    fig.savefig(os.path.join(args.out_dir, "result.png"))


if __name__ == "__main__":
    main()
