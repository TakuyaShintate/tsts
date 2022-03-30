.. _transforms:

=================
Data Augmentation
=================

Data augmentation can be applied during training by adding a `PIPELINE` section to the configuration. Multiple data augmentations can be applied in combination by adding multiple data augmentations to the list.

.. contents:: Catalog
    :depth: 1
    :local:

--------------
Gaussian Noise
--------------

Add gaussian noise to input values.

.. code-block:: yaml

    PIPELINE:
      TRANSFORMS_TRAIN:
        [{"name": "GaussianNoise", "args": {"mean": 0.0, "std": 0.1}}]
