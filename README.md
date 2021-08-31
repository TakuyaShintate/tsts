<div align="center">
  <img src="img/tsts-logo.png" width="600"/>
</div>

[![pypi](https://img.shields.io/pypi/v/tsts?style=flat)](https://pypi.org/project/tsts/0.2.2/)
[![license](https://img.shields.io/github/license/TakuyaShintate/tsts?style=flat)](https://github.com/TakuyaShintate/tsts/blob/main/LICENSE)

([Docs](https://takuyashintate.github.io/tsts/))([Benchmark](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/))

## Introduction

tsts is an open-source easy-to-use toolset for time series forecasting.

## Installation

```
pip install tsts
```

## What's that?

<div align="center">
  <img src="img/whats-forecasting.png" width="850"/>
</div>

Time series forecasting is the task to predict the values of the time series on **Horizon** given the values of the time series on **Lookback Period**. Note that data can be multivariate.

## Getting started

Following example shows how to train a model on sine curve dataset. See [Docs](https://takuyashintate.github.io/tsts/) for the details.

```python
import torch
from tsts.solvers import TimeSeriesForecaster

sin_dataset = torch.sin(torch.arange(0, 100, 0.1))
sin_dataset = sin_dataset.unsqueeze(-1)
forecaster = TimeSeriesForecaster()
forecaster.fit([sin_dataset])
``` 
