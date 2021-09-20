======
SCINet
======

How to Use
==========

Add following lines to config to use *SCINet*.

.. code-block:: yaml

    MODEL:
      NAME: "SCINet"
      # Number of levels
      DEPTH: 3
      # Kernel size of conv modules
      KERNEL_SIZE: 5
      # Expansion rate of conv modules
      EXPANSION_RATE: 4.0
      # Dropout rate
      DROPOUT_RATE: 0.5

Reference
=========

`Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction <https://arxiv.org/abs/2106.09305?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29>`_
