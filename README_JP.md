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

## 🚀 使用方法

あるベンチマークでモデルの性能を測定したい場合など、決められたテストデータに対して予測を行いたい場合は"`tools/train.py`/`tools/test.py`を使用する場合"を、オンラインで予測を行いたい場合は"APIを使用する場合"を参照してください。

### `tools/train.py`/`tools/test.py`を使用する場合

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

作業中です 🔧
