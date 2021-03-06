.. _config:

===============================
How a Config File is Structured
===============================

Indicates items that can be changed in the configuration. See `Catalog` for the modules available in each section. If you want to try out & test, copy and use `Putting It Together` at the bottom. Modules that can be changed in the config include models, optimizers, loss functions, and data augmentation, etc. Other modules that can be changed include the device to be used, how training and test data are split, etc.

Basic Usage
===========

Config can be specified and used during training. In this case, the value of the config is reflected.

.. code-block:: bash 

    python tools/train.py --cfg-name "" ...

General Settings
================

.. code-block:: yaml
    :emphasize-lines: 1,2,3,4

    # Which device to use
    DEVICE: "cuda:0"
    # Random seed
    SEED: 42

    IO:
      ...

Input & Output
==============

.. code-block:: yaml
    :emphasize-lines: 3,4,5,6,7

    SEED: ...

    IO:
      # Number of input time steps
      LOOKBACK: 48
      # Number of output time steps
      HORIZON: 16

Training
========

.. code-block:: yaml
    :emphasize-lines: 4,5,6,7,8,9,10,11,12,13,14

    IO:
      ...

    TRAINING:
      # How to split train dataset {"col" or "row"}
      TRAIN_DATA_SPLIT: "col"
      # Ratio of train dataset size over val dataset size
      TRAIN_DATA_RATIO: 0.75
      # Number of epochs
      NUM_EPOCHS: 100
      # If True, datasets are split randomly (valid only if TRAIN_DATA_SPLIT = "col")
      RANDOM_SPLIT: False
      # Try to load pretrained model & local scaler in this directory
      PRETRAIN: None

Optimizer
=========

See a full list of :ref:`optimizers <optimizers>`.

.. code-block:: yaml
    :emphasize-lines: 4,5,6,7,8

    TRAINING:
      ...

    OPTIMIZER:
      # Optimizer name
      NAME: "Adam"
      # Learning rate
      LR: 0.001

Learning Rate Scheduler
=======================

See a full list of :ref:`learning rate schedulers <schedulers>`.

.. code-block:: yaml
    :emphasize-lines: 4,5,6

    OPTIMIZER:
      ...

    SCHEDULER:
      # Scheduler name
      NAME: "CosineAnnealing"

Trainer
=======

.. code-block:: yaml
    :emphasize-lines: 4,5,6,7,8,9,10

    SCHEDULER:
      ...

    TRAINER:
      # Trainer name
      NAME: "SupervisedTrainer"
      # Maximum gradient norm
      MAX_GRAD_NORM: 1.0
      # Denormalize before computing metric values
      DENORM: False

Model
=====

See a full list of :ref:`models <models>`.

.. code-block:: yaml
    :emphasize-lines: 4,5,6

    TRAINER:
      ...

    MODEL:
      # Model name
      NAME: "Seq2Seq"

Local Scaler 
============

.. code-block:: yaml
    :emphasize-lines: 4,5,6

    MODEL:
      ...

    LOCAL_SCALER:
      # Local scaler name
      NAME: "NOOP"

Loss Function
=============

See a full list of :ref:`loss functions <losses>`. Multiple loss functions can be passed.

.. code-block:: yaml
    :emphasize-lines: 4,5,6,7,8,9,10

    LOCAL_SCALER:
      ...

    LOSSES:
      # Loss function names
      NAMES: ["MSE"]
      # Loss function arguments
      ARGS: [{}]
      # Loss function weights
      WEIGHT_PER_LOSS: [1.0]

Metric
======

.. code-block:: yaml
    :emphasize-lines: 4,5,6,7,8

    LOSSES:
      ...

    METRICS:
      # Metric names
      NAMES: ["RMSE"]
      # Metric arguments
      ARGS: [{}]

Dataset
=======

If i-th sequence is smaller than **IO.LOOKBACK**, it is zero padded to match **IO.LOOKBACK**.

.. note:: Zero padding may reduce the accuracy. To avoid zero padding, set **DATASET.BASE_START_INDEX** to the same value to 2 * **IO.LOOKBACK**.

.. code-block:: yaml
    :emphasize-lines: 4,5,6,7,8

    METRICS:
      ...

    DATASET:
      # Dataset index starts with this value
      BASE_START_INDEX: 0
      # Last BASE_END_INDEX samples are not used for training
      BASE_END_INDEX: -1

Pipeline
========

See a full list of :ref:`transforms <transforms>`.

.. code-block:: yaml
    :emphasize-lines: 4,5,6,7,8,9

    DATASET:
      ...

    PIPELINE:
      # List of transforms
      # Each dictionary must contain `name` and `args` pairs
      # Ex: [{"name": "GaussianNoise", "args": {"mean": 0.0, "std": 0.001}}]
      TRANSFORMS_TRAIN: []
      TRANSFORMS_VALID: []

Scaler 
======

.. code-block:: yaml
    :emphasize-lines: 4,5,6,7,8,9,10

    PIPELINE:
      ...

    X_SCALER:
      # Scaler for input time series
      NAME: "StandardScaler"

    Y_SCALER:
      # Scaler for output time series
      NAME: "StandardScaler"


Dataloader
===========

.. code-block:: yaml
    :emphasize-lines: 7,8,9,10,11,12,13,14,15

    X_SCALER:
      ...

    Y_SCALE:
      ...

    DATALOADER:
      # Train dataloader name
      NAME_TRAIN: "DataLoader"
      # Validation dataloader name
      NAME_VALID: "DataLoader"
      # Batch size of train dataset
      BATCH_SIZE_TRAIN: 100
      # Batch size of validation dataset
      BATCH_SIZE_VALID: 100


Logger 
======

.. code-block:: yaml

    DATALOADER:
      ...

    LOGGER:
      # Log directory name (if "auto", it is randomly generated)
      LOG_DIR: "auto"

Putting It Together
===================

.. code-block:: yaml

    # Which device to use
    DEVICE: "cuda:0"
    # Random seed
    SEED: 42

    IO:
      # Number of input time steps
      LOOKBACK: 48
      # Number of output time steps
      HORIZON: 16

    TRAINING:
      # How to split train dataset {"col" or "row"}
      TRAIN_DATA_SPLIT: "col"
      # Ratio of train dataset size over val dataset size
      TRAIN_DATA_RATIO: 0.75
      # Number of epochs
      NUM_EPOCHS: 100
      # If True, datasets are split randomly (valid only if TRAIN_DATA_SPLIT = "col")
      RANDOM_SPLIT: False
      # Try to load pretrained model & local scaler in this directory
      PRETRAIN: None

    OPTIMIZER:
      # Optimizer name
      NAME: "Adam"
      # Learning rate
      LR: 0.001

    SCHEDULER:
      # Scheduler name
      NAME: "CosineAnnealing"

    TRAINER:
      # Trainer name
      NAME: "SupervisedTrainer"
      # Maximum gradient norm
      MAX_GRAD_NORM: 1.0
      # Denormalize before computing metric values
      DENORM: False

    MODEL:
      # Model name
      NAME: "Seq2Seq"

    LOCAL_SCALER:
      # Local scaler name
      NAME: "NOOP"

    LOSSES:
      # Loss function names
      NAMES: ["MSE"]
      # Loss function arguments
      ARGS: [{}]
      # Loss function weights
      WEIGHT_PER_LOSS: [1.0]

    METRICS:
      # Metric names
      NAMES: ["RMSE"]
      # Metric arguments
      ARGS: [{}]

    DATASET:
      # Dataset index starts with this value
      BASE_START_INDEX: 0
      # Last BASE_END_INDEX samples are not used for training
      BASE_END_INDEX: -1

    PIPELINE:
      # List of transforms
      # Each dictionary must contain `name` and `args` pairs
      # Ex: [{"name": "GaussianNoise", "args": {"mean": 0.0, "std": 0.001}}]
      TRANSFORMS_TRAIN: []
      TRANSFORMS_VALID: []

    X_SCALER:
      # Scaler for input time series
      NAME: "StandardScaler"

    Y_SCALER:
      # Scaler for output time series
      NAME: "StandardScaler"

    DATALOADER:
      # Train dataloader name
      NAME_TRAIN: "DataLoader"
      # Validation dataloader name
      NAME_VALID: "DataLoader"
      # Batch size of train dataset
      BATCH_SIZE_TRAIN: 100
      # Batch size of validation dataset
      BATCH_SIZE_VALID: 100

    LOGGER:
      # Log directory name (if "auto", it is randomly generated)
      LOG_DIR: "auto"
