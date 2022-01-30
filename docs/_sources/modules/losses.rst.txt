==============
Loss Functions
==============

.. contents:: Catalog
    :depth: 1
    :local:

------
DILATE
------

.. code-block:: yaml

    LOSS:
      NAME: ["DILATE"]
      ARGS: [{"alpha": 0.5, "gamma": 0.001}]

🔍 Reference
------------

`Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models <https://arxiv.org/abs/1909.09020>`_

-------------------------
Mean Absolute Error (MAE)
-------------------------

.. code-block:: yaml

    LOSS:
      NAME: ["MAE"]
      ARGS: []

-------------------------------------
Mean Absolute Percentage Error (MAPE)
-------------------------------------

.. code-block:: yaml

    LOSS:
      NAME: ["MAPE"]
      ARGS: []

------------------------
Mean Squared Error (MSE)
------------------------

.. code-block:: yaml

    LOSS:
      NAME: ["MSE"]
      ARGS: []

----------
Smooth MAE
----------

.. code-block:: yaml

    LOSSES:
      NAMES: ["SmoothMAE"]
      ARGS: [{"beta": 1.0 / 9.0}]

🔍 Reference
------------

`Fast R-CNN <https://arxiv.org/abs/1504.08083>`_
