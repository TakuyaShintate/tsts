<div align="center">
  <img src="img/tsts-logo.png" width="600"/>
</div>

---

[![pypi](https://img.shields.io/pypi/v/tsts?style=flat)](https://pypi.org/project/tsts/1.0.0/)[![license](https://img.shields.io/github/license/TakuyaShintate/tsts?style=flat)](https://github.com/TakuyaShintate/tsts/blob/main/LICENSE)

[English](README.md)/日本語

([ドキュメンテーション](https://takuyashintate.github.io/tsts/))

## ❓ このプロジェクトについて

最新の時系列予測手法を提供するオープンソースプロジェクトです。

自己回帰（AR）モデルなどと組み合わせてモデルを構築するなど、より柔軟なモデルの構築が可能です。また、モデルの他にData Augmentationや損失関数、オプティマイザーに関しても最新のモジュールを提供しています。

## ⛏ インストール方法

```
pip install tsts
```

## ⚡️ コンフィグサンプル

各モデルの使用例に関しては、[サンプル](cfg)を参照してください。

## 🚀 使用方法

あるベンチマークでモデルの性能を測定したい場合など、決められたテストデータに対して予測を行いたい場合は"`tools/train.py` & `tools/test.py`を使用する場合"を、オンラインで予測を行いたい場合は"APIを使用する場合"を参照してください。

### `tools/train.py` & `tools/test.py`を使用する場合

✅ 少ないコードで学習 & 推論を行うことができます

#### 1. 学習に使用するデータの準備

訓練データ（CSVファイル）とバリデーションデータ、テストデータを各々のディレクトリに保存してください。ディレクトリの名前は任意です。訓練・バリデーション・テストデータが複数存在する場合、それらを全て各々のディレクトリに保存してください。

##### CSVファイルの例

> 実行時に使用する入力・出力変数を選択できます

| feat0 | feat1 | feat2 |
| ----- | ----- | ----- |
| xxxxx | yyyyy | zzzzz |

#### 2. コンフィグファイルを作成

学習時の設定を記述したコンフィグファイルを作成します。モデル、Data Augmentation、オプティマイザー、学習率スケジューラーなどを指定することができます。設定可能な項目についての詳細は[ドキュメンテーション](https://takuyashintate.github.io/tsts/projects/config.html)を参照してください。

今回は簡単のため、最小限のコンフィグファイルを使用します。下の内容を`my_first_model.yml`として保存します。[ドキュメンテーション](https://takuyashintate.github.io/tsts/)から対象のセクションをコピーすることで別のモデルや手法を使用できます。

```yaml
# `my_first_model.yml`として保存してください
LOGGER:
  LOG_DIR: "my-first-model"
```

#### 3. 学習

下のコマンドを実行して学習を始めます。いったん学習が始まると、2で指定したディレクトリ（`my-first-model`）にモデルのパラメータやログファイルが作成されます。ログファイルには損失値やメトリックの値が書き込まれます。

> モデルなどと同様、損失関数やメトリックも変更可能です

`--train-dir`と`--valid-dir`には訓練データとバリデーションデータを保存したディレクトリを指定してください。`--in-feats`と`--out-feats`には入力変数の名前のリストと出力変数の名前のリストを指定してください。

ある出力変数に対し、入力変数と同じ時刻の値を予測したい場合があります（時刻t-n:tの入力変数の値に対し、時刻t-n:tの出力変数の値を予測したい場合）。そうした場合には`--lagging`オプションを追加してください。

```
python tools/train.py \
    --cfg-name my_first_model.yml \
    --train-dir <dir to contain training data> \
    --valid-dir <dir to contain validation data> \
    --in-feats <list of input feature names> \
    --out-feats <list of output feature names>
```

#### 4. 学習後のモデルのテスト

学習が完了した後に下のコマンドを実行することで、テストデータに対する予測結果を得ることができます。予測結果と正解ラベル、それらの誤差が記録されたCSVファイルとそれらがプロットされた画像が`--out-dir`で指定されたディレクトリに保存されます。各テストデータに対して結果が保存されます。

```
python tools/test.py \
    --cfg-name my_first_model.yml \
    --train-dir <dir to contain training data> \
    --valid-dir <dir to contain validation data> \
    --test-dir <dir to contain test data> \
    --in-feats <list of input feature names> \
    --out-feats <list of output feature names> \
    --out-dir <result is saved in this directory>
```

### APIを使用する場合

✅ 未来の値を予測するために使用できます

#### 1~3. データの準備 ~ 学習

`tools/train.py` & `tools/test.py`を使用する場合 と同様の手順です。

#### 4. 学習後のモデルのテスト

学習済みのモデルを使用して任意の値で予測を行います。

> 入力値は（タイムステップ数, 変数の数）の形をしている必要があります

```python
import glob
import os

import pandas as pd
import torch
from tsts.cfg import get_cfg_defaults
from tsts.scalers import build_X_scaler, build_y_scaler
from tsts.solvers import TimeSeriesForecaster
from tsts.utils import plot


IN_FEATS = "<list of input feature names>"
OUT_FEATS = "<list of output feature names>"


def load_cfg(cfg_name):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_name)
    return cfg


def load_sample(cfg, filename):
    df = pd.read_csv(filename)
    df = df.fillna(0.0)
    # Take only the values of input & output variables
    x = torch.tensor(df[IN_FEATS].values, dtype=torch.float32)
    y = torch.tensor(df[OUT_FEATS].values, dtype=torch.float32)
    return (x, y)
  

def build_scalers(cfg):
    X_scaler = build_X_scaler(cfg)
    y_scaler = build_y_scaler(cfg)
    X = []
    Y = []
    for filename in glob.glob(os.path.join("<dir to contain training data>", "*.csv")):
        # Initialize input & output values
        (x, y) = load_sample(cfg, filename)
        X.append(x)
        Y.append(y)
    X_scaler.fit_batch(X)
    y_scaler.fit_batch(Y)
    return (X_scaler, y_scaler)


# Load merged config
cfg = load_cfg("my_first_model.yml")
# Build input & output value scalers
(X_scaler, y_scaler) = build_scalers(cfg)
solver = TimeSeriesForecaster("my_first_model.yml")
# Initialize inputs to the model (`torch.Tensor`)
# NOTE: This input value does not have to be on the device specified in training
test_input = "..."
# Before passing input values to the model, scale the values in the same way as during training
test_input = X_scaler.transform(test_input)
test_output = solver.predict(test_input)
# Restore the result to the original scale
test_output = y_scaler.inv_transform(test_output)

# Plot the result
# NOTE: One figure is assigned to one variable
plot(test_output)
```
