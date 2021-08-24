=============================
Inference (Sine/Cosine Curve)
=============================

In this tutorial, we will learn how to run inference on a test data.

Workflow
========

For inference, we have 2 steps:

1. Specify **LOG_DIR** (**LOG_DIR** is a directory where parameters are saved)
2. Run **predict** method

(Step 1) Pre-trained Model Restoration
======================================

When training starts, a log directory where the results are saved will be made. To restore the results, specify the log directory path in a config file and pass it when initialization. 

.. code-block:: yaml

    # infer.yml
    LOGGER:
      # For default, log_dir name is randomly generated
      LOG_DIR: "bfb5118b-7687-453d-a8d8-6100df7d36d4"

.. code-block:: python

    from tsts.solvers import Forecaster

    forecaster = Forecaster("infer.yml")

To specify the name of **log_dir**, pass a config file when starting training.

.. code-block:: yaml

    # custom-log-dir.yml
    LOGGER:
      LOG_DIR: "mymodel"

.. code-block:: python

    import torch
    from tsts.solvers import Forecaster

    sin_dataset = torch.sin(torch.arange(0, 100, 0.1))
    sin_dataset = sin_dataset.unsqueeze(-1)
    forecaster = Forecaster("custom-log-dir.yml")
    forecaster.fit([sin_dataset])

(Step 2) Running Inference
==========================

Run **predict** method to infer on test data.

.. code-block:: python

    import torch
    from tsts.solvers import Forecaster

    test_data = torch.arange(0, 10, 0.1)
    test_data = test_data.unsqueeze(-1)
    forecaster = Forecaster("custom-log-dir.yml")
    print(forecaster.predict(test_data))

    """
    Output:
    tensor([[0.1068],
        [0.2669],
        [0.3835],
        [0.4387],
        [0.4649],
        [0.4782],
        [0.4856],
        [0.4902]])
    """


