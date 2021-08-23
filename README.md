![tsts-logo](img/tsts-logo.png)

🐛 [Docs](https://takuyashintate.github.io/tsts/)

## Introduction

tsts is an opensource easy-to-use toolset for time series forecasting.

## Installation

```
git clone https://github.com/TakuyaShintate/tsts.git
cd tsts
pip install .
```

## Quick Start

For training, use `fit` method as [sklearn](https://scikit-learn.org/stable/).

```python
import torch
from tsts.solvers import Forecaster

# Use cosine curve as dataset
X = torch.cos(torch.arange(0, 100, 0.1)).unsqueeze(-1)
forecaster = Forecaster()
forecaster.fit(X)
```

Use `predict` for inference.

> Batch inference is no supported yet (we are working on it)

`fit` method makes random named directory (it can be configured) which contains training log and model parameters. Refer to the directory to load pretrained model by adding LOGGER section to a custom config.

```yaml
# cos.yml
LOGGER:
  LOG_DIR: "b5a20cc3-6d8e-43a0-8d08-4357de81d8d7"
```

Update default config by the custom config (`cos.yml`) and use it to load the pretrained model for inference.

```python
import torch
from tsts.cfg import get_cfg_defaults
from tsts.solvers import Forecaster

cfg = get_cfg_defaults()
cfg.merge_from_file("cos.yml")
forecaster = Forecaster(cfg)
X = torch.cos(torch.arange(0, 100, 0.1)).unsqueeze(-1)
Z = forecaster.predict(X)
```

## Models

Add `MODEL` section shown below to your config to use.

* Seq2Seq

```yaml
MODEL:
  NAME: "Seq2Seq"
```

* [NBeats](https://arxiv.org/abs/1905.10437)

```yaml
MODEL:
  NAME: "NBeats"
  NUM_H_UNITS: 512
```

### Example

Create a custom config to use the custom model.

```yaml
# custom_model.yml
MODEL:
  NAME: "Seq2Seq"
```

Then, update default config with the custom config and pass it to `Forecaster`.

```python
from tsts.cfg import get_cfg_defaults
from tsts.solvers import Forecaster

cfg = get_cfg_defaults()
cfg.merge_from_file("custom_model.yml")
forecaster = Forecaster(cfg)
```

## Losses

Add `LOSSES` section shown below to your config to use.

* [DILATE](https://arxiv.org/abs/1909.09020)

```yaml
LOSSES:
  NAMES: ["DILATE"]
```

### Example

Create a custom config to use the custom loss. Note `NAMES` takes a list of loss functions, which means multiple loss functions can be used. In that case, configure `WEIGHT_PER_LOSS` to reweight loss functions.

```yaml
# custom_loss.yml
LOSSES:
  NAMES: ["DILATE"]
  WEIGHT_PER_LOSS: [1.0]  # Optional if only single loss function
```

Then, update default config with the custom config and pass it to `Forecaster`.

```python
from tsts.cfg import get_cfg_defaults
from tsts.solvers import Forecaster

cfg = get_cfg_defaults()
cfg.merge_from_file("custom_loss.yml")
forecaster = Forecaster(cfg)
```
