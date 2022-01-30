=================
Data Augmentation
=================

.. contents:: Catalog
    :depth: 1
    :local:

--------------
Gaussian Noise
--------------

Adds gaussian noise to input time series

.. code-block:: yaml

    PIPELINE:
      TRANSFORMS_TRAIN:
        [{"name": "GaussianNoise", "args": {"mean": 0.0, "std": 0.1}}]
