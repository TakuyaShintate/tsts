<div align="center">
  <img src="img/tsts-logo.png" width="600"/>
</div>

[![pypi](https://img.shields.io/pypi/v/tsts?style=flat)](https://pypi.org/project/tsts/0.4.0/)
[![license](https://img.shields.io/github/license/TakuyaShintate/tsts?style=flat)](https://github.com/TakuyaShintate/tsts/blob/main/LICENSE)

([Docs](https://takuyashintate.github.io/tsts/))([Benchmark](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/))

## Introduction

tsts is an open-source easy-to-use toolset for time series forecasting.

## What's New

### Sep, 10

✨ New Model  

Informer and related modules were added. See [results](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/informer) for how to use it.

## Installation

```
pip install tsts
```

## Task Details

<div align="center">
  <img src="img/whats-forecasting.png" width="850"/>
</div>

Time series forecasting is the task to predict the values of the time series on **Horizon** given the values of the time series on **Lookback Period**. Note that data can be multivariate.

## Available Modules

Following modules are supported.

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Architectures</b>
      </td>
      <td>
        <b>Losses</b>
      </td>
      <td>
        <b>Metrics</b>
      </td>
      <td>
        <b>Optimizers</b>
      </td>
      <td>
        <b>Scalers</b>
      </td>
      <td>
        <b>Schedulers</b>
      </td>
    </tr>
    <tr>
      <td>
        <li>Informer</li>
        <li>NBeats</li>
        <li>Seq2Seq</li>
      </td>
      <td>
        <li>DILATE</li>
        <li>MAPE</li>
        <li>MSE</li>
      </td>
      <td>
        <li>MAE</li>
        <li>MSE</li>
        <li>RMSE</li>
        <li>MAPE</li>
      </td>
      <td>
        <li>Adam</li>
        <li>SGD</li>
      </td>
      <td>
        <li>MinMaxScaler</li>
        <li>StandardScaler</li>
      </td>
      <td>
        <li>CosineAnnealing</li>
        <li>StepScheduler</li>
      </td>
    </td>
    </tr>
  </tbody>
</table>

## Getting Started

Following example shows how to train a model on sine curve dataset. See [Docs](https://takuyashintate.github.io/tsts/) for the details.

### Training

Define config and start training with it.

```yaml
# cfg.yml
LOGGER:
  # Log file and parameters are saved here
  LOG_DIR: "my-first-tsts-model"
```

```python
import torch
from tsts.solvers import TimeSeriesForecaster

# Define trainining + validation datasets (they are divided inside)
sin_dataset = torch.sin(torch.arange(0.0, 100.0, 0.1))
sin_dataset = sin_dataset.unsqueeze(-1)

# Run training
forecaster = TimeSeriesForecaster("cfg.yml")
forecaster.fit([sin_dataset])
``` 

See results in `my-first-tsts-model` directory.

### Inference

For inference, it needs to load parameters from a log directory generated in training.

```python
import torch
from tsts.scalers import StandardScaler
from tsts.solvers import TimeSeriesForecaster

# Initialize scaler with training dataset
sin_dataset = torch.sin(torch.arange(0, 100, 0.1))
sin_dataset = sin_dataset.unsqueeze(-1)
X_scaler = StandardScaler()
# NOTE: 0.75 is default training validation dataset ratio (training: 0.75, validation: 0.25)
num_train_samples = int(0.75 * len(sin_dataset))
X_scaler.fit(sin_dataset)

# Define test dataset
X = torch.sin(torch.arange(100.0, 110.0, 0.1))
X = X.unsqueeze(-1)
X = X_scaler.transform(X)

# Run inference
forecaster = TimeSeriesForecaster("cfg.yml")
Z = forecaster.predict(X)

# Initialize target to compare prediction with it
y = torch.sin(torch.arange(110.0, 110.8, 0.1)).unsqueeze(1)
y = X_scaler.transform(y)

# Result
print(Z)
print(y)
```

## Examples

See [Benchmark](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/) for advanced usage.
