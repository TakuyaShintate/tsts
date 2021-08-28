<div align="center">
  <img src="img/tsts-logo.png" width="600"/>
</div>

[Docs](https://takuyashintate.github.io/tsts/)・[Benchmark](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/)

## Introduction

tsts is an opensource easy-to-use toolset for time series forecasting.

## Installation

```
pip install tsts
```

## Getting Started

Following example shows how to train a model on sine curve dataset. See [Docs](https://takuyashintate.github.io/tsts/) for the details.

```python
import torch
from tsts.solvers import TimeSeriesForecaster

sin_dataset = torch.sin(torch.arange(0, 100, 0.1))
sin_dataset = sin_dataset.unsqueeze(-1)
forecaster = TimeSeriesForecaster()
forecaster.fit([sin_dataset])
```

## Available Modules

### Models

* [Seq2Seq](https://arxiv.org/abs/1409.3215)
* [NBeats](https://arxiv.org/abs/1905.10437) 

### Losses

* [DILATE](https://arxiv.org/abs/1909.09020)
* MAPE
* MSE
