===
API
===

Getting Started 
===============

Following example shows how to train a model on sine curve dataset.

.. code-block:: python

    import torch
    from tsts.solvers import Forecaster

    sin_dataset = torch.sin(torch.arange(0, 100, 0.1))
    sin_dataset = sin_dataset.unsqueeze(-1)
    forecaster = Forecaster()
    forecaster.fit([sin_dataset])

.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    tutorials/cfg
    tutorials/train
    tutorials/infer
    tutorials/gpu

.. toctree::
    :maxdepth: 1
    :caption: Modules

    collators/collator
    losses/loss
    metrics/metric
    models/model
    optimizers/optimizer
    schedulers/scheduler
