============
GPU Training
============

In this tutorial, we will learn how to use GPU for training.

Using GPU for Training
======================

To use GPU for training, specify **DEVICE** in a config file.

.. code-block:: yaml

    # gpu-training.yml
    DEVICE: "cuda:0"

Initialize **Forecaster** with the config file.

.. code-block:: python

    from tsts.solvers import Forecaster

    forecaster = Forecaster("gpu-training.yml")
