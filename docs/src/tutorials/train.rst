========
Training
========

In this tutorial, we will learn how to start training on a custom dataset.

Workflow
========

To start training, we have 3 steps:

1. (Optional) Make a config file
2. Prepare training datasets (validation dataset should be included in training dataset)
3. Run **fit** method
  
Let's go through step by step.

(Step 1) Configuration
======================

Training process can be configured by a custom config file. In the following config file, model and the number of hidden units are indicated.

.. code-block:: yaml

    # first-config.yml
    MODEL:
      NAME: "NBeats"
      NUM_H_UNITS: 512

To update global configuration with the custom config file, pass the custom config file path to **TimeSeriesForecaster**.

.. code-block:: python

    import torch
    from tsts.solvers import TimeSeriesForecaster

    sin_dataset = torch.sin(torch.arange(0, 100, 0.1))
    sin_dataset = sin_dataset.unsqueeze(-1)
    forecaster = TimeSeriesForecaster(cfg_path="first-config.yml")
    forecaster.fit([sin_dataset])

You can see how it was changed by **cfg** property.

.. code-block:: python

    print(forecaster.cfg)

.. code-block:: yaml
    :emphasize-lines: 3, 4

    MODEL:
      DEPTH: 2
      NAME: NBeats
      NUM_H_UNITS: 512

See **tsts.cfg.defaults.py** for more details.

(Step 2) Dataset Preparation
============================

**fit** takes a list of dataset. Inside the method, each dataset is split into training dataset and validation dataset. You can specify the training dataset ratio by TRAINING.TRAIN_DATA_RATIO.

.. code-block:: yaml

    TRAINING:
      TRAIN_DATA_RATIO: 0.8

Each dataset in the list must have the shape (number of instances, number of features).

.. code-block:: python

    import torch
    from tsts.solvers import TimeSeriesForecaster

    sin_dataset = torch.sin(torch.arange(0, 100, 0.1))
    sin_dataset = sin_dataset.unsqueeze(-1)
    print(sin_dataset.size())  # (1000, 1)

If you want to use multiple datasets, add a new dataset to the list.

.. code-block:: python

    import torch

    sin_dataset = torch.sin(torch.arange(0, 100, 0.1))
    sin_dataset = sin_dataset.unsqueeze(-1)
    cos_dataset = torch.cos(torch.arange(0, 100, 0.1))
    cos_dataset = cos_dataset.unsqueeze(-1)
    dataset = [sin_dataset, cos_dataset]

(Step 3) Start Training
=======================

Training can be started just by running **fit**.

.. code-block:: python

    from tsts.solvers import TimeSeriesForecaster

    forecaster = TimeSeriesForecaster()
    forecaster.fit(dataset)

If you have specific target time series, you can pass it by **y**. Then model is trained to predict **y**.

.. code-block:: python

    import torch
    from tsts.solvers import TimeSeriesForecaster

    sin_dataset = torch.sin(torch.arange(0, 100, 0.1))
    sin_dataset = sin_dataset.unsqueeze(-1)
    cos_dataset = torch.cos(torch.arange(0, 100, 0.1))
    cos_dataset = cos_dataset.unsqueeze(-1)
    forecaster = TimeSeriesForecaster()
    forecaster.fit(X=[sin_dataset], y=[cos_dataset])
