.. _losses:

==============
Loss Functions
==============

Adding a `LOSSES` section to the config allows the user to select the loss function to be used during training. Multiple loss functions can be used by adding multiple elements to the list. You can set the weight of each loss function in `WEIGHT_PER_LOSS`. The default loss function is `MSE`.

.. contents:: Catalog
    :depth: 1
    :local:

------
DILATE
------

.. code-block:: yaml

    LOSSES:
      NAMES: ["DILATE"]
      ARGS: [{"alpha": 0.5, "gamma": 0.001}]

Reference
------------

`Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models <https://arxiv.org/abs/1909.09020>`_

-------------------------
Mean Absolute Error (MAE)
-------------------------

.. code-block:: yaml

    LOSSES:
      NAMES: ["MAE"]
      ARGS: [{}]

-------------------------------------
Mean Absolute Percentage Error (MAPE)
-------------------------------------

.. code-block:: yaml

    LOSSES:
      NAMES: ["MAPE"]
      ARGS: [{}]

------------------------
Mean Squared Error (MSE)
------------------------

.. code-block:: yaml

    LOSSES:
      NAMES: ["MSE"]
      ARGS: [{}]

----------
Smooth MAE
----------

.. code-block:: yaml

    LOSSES:
      NAMES: ["SmoothMAE"]
      ARGS: [{"beta": 0.11}]

Reference
------------

`Fast R-CNN <https://arxiv.org/abs/1504.08083>`_
