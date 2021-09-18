=====================
(Tutorial 1) Training
=====================

In this tutorial, we will learn how to start training on a custom dataset.

Workflow
========

To start training, we have 3 steps:

1. (Optional) Make a config file
2. Prepare training datasets (validation dataset should be included in training dataset)
3. Run **fit** method
  
Let's go through step by step.

(Step 1) Config
===============

Training can be configured by a custom config. In the following config file, model and the number of hidden units are specified.

.. code-block:: yaml

    # cfg.yml
    LOGGER:
      # Log file and parameters are saved here
      LOG_DIR: "my-first-tsts-model"
    MODEL:
      NAME: "NBeats"
      # Number of hidden units
      NUM_H_FEATS: 512

To update default config, pass the custom config file path to **TimeSeriesForecaster**.

.. code-block:: python

    import torch
    from tsts.solvers import TimeSeriesForecaster

    # Define training + validation datasets (they are divided inside)
    sin_dataset = torch.sin(torch.arange(0.0, 100.0, 0.1))
    sin_dataset = sin_dataset.unsqueeze(-1)

    # Pass config here
    forecaster = TimeSeriesForecaster("cfg.yml")

You can see how it was changed by **cfg** property.

.. code-block:: python

    print(forecaster.cfg)

See **tsts.cfg.defaults.py** for more details.

(Step 2) Dataset Preparation
============================

**fit** takes a list of dataset. Inside this method, each dataset is split into training dataset and validation dataset. You can specify the training dataset ratio by TRAINING.TRAIN_DATA_RATIO.

.. code-block:: yaml

    # cfg.yml
    LOGGER:
      # Log file and parameters are saved here
      LOG_DIR: "my-first-tsts-model"
    MODEL:
      NAME: "NBeats"
      # Number of hidden units
      NUM_H_FEATS: 512
    TRAINING:
      TRAIN_DATA_RATIO: 0.8

Each dataset in the list must have the shape (number of instances, number of features).

.. code-block:: python

    import torch
    from tsts.solvers import TimeSeriesForecaster

    sin_dataset = torch.sin(torch.arange(0.0, 100.0, 0.1))
    sin_dataset = sin_dataset.unsqueeze(-1)
    print(sin_dataset.size())  # (1000, 1)

If you want to use multiple datasets, add a new dataset to the list.

.. code-block:: python

    import torch

    sin_dataset = torch.sin(torch.arange(0.0, 100.0, 0.1))
    sin_dataset = sin_dataset.unsqueeze(-1)

    # Now define cosine dataset
    cos_dataset = torch.cos(torch.arange(0.0, 100.0, 0.1))
    cos_dataset = cos_dataset.unsqueeze(-1)

    dataset = [sin_dataset, cos_dataset]

(Step 3) Start Training
=======================

Training can be started just by running **fit**.

.. code-block:: python

    ...

    # Pass config here
    forecaster = TimeSeriesForecaster("cfg.yml")

    # Run training
    forecaster.fit(dataset)

If you have specific target time series, you can pass it by **y**. Then model is trained to predict **y**.

.. code-block:: python

    ...

    forecaster.fit(X=[sin_dataset], y=[cos_dataset])