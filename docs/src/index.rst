===
API
===

Getting Started 
===============

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

    tutorials/train

.. toctree::
    :maxdepth: 1
    :caption: Modules

    collators/collator
