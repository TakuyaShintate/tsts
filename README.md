![tsts-logo](img/tsts-logo.png)

## Introduction

tsts is an opensource easy-to-use toolset for time series forecasting.

## Installation

TODO

## Quick Start

For training, use `fit` method as [sklearn].

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

`fit` method makes random named directory (it can be configured) which contains training log and model parameters. Refer to the directory to load pretrained model in your config.

```yaml
# cos.yml
LOGGER:
  LOG_DIR: "b5a20cc3-6d8e-43a0-8d08-4357de81d8d7"
```

Update default config by your config (`cos.yml`) and use it to use pretrained model for inference.

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

## Losses

Add `LOSSES` section shown below to your config to use.

* [DILATE](https://arxiv.org/abs/1909.09020)

```yaml
LOSSES:
  NAMES: ["DILATE"]
```
