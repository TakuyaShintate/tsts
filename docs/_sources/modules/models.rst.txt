======
Models
======

.. contents:: Catalog
    :depth: 1
    :local:

--------
Informer
--------

.. code-block:: yaml

    MODEL:
      NAME: "Informer"
      # Number of hidden units
      NUM_H_FEATS: 512
      # Number of encoders
      NUM_ENCODERS: 2
      # Number of decoders 
      NUM_DECODERS: 1
      # Number of heads of self attention
      NUM_HEADS: 8
      # Smaller value leads to higher memory efficiency
      CONTRACTION_FACTOR: 5
      # int(NUM_H_FEATS * FF_EXPANSION_RATE) is channel size of conv block after self attention
      EXPANSION_RATE: 4.0
      # Decoder input series length (last DECODER_IN_LENGTH values are used)
      DECODER_IN_LENGTH: 168
      # Dropout rate
      DROPOUT_RATE: 0.05

🔍 Reference
------------

`Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting <https://arxiv.org/abs/2012.07436>`_

-------
N-BEATS
-------

.. code-block:: yaml

    MODEL:
      NAME: "NBeats"
      # Number of hidden units
      NUM_H_FEATS: 512
      # Depth of each block (set small value if dataset has high mean and variance)
      DEPTH: 4
      # Number of blocks
      STACK_SIZE: 30
      # Block type (option: {"identity", "trend"})
      BLOCK_TYPE: "identity"
      # Polynomial degree (used only if BLOCK_TYPE == "trend")
      DEGREE: 2

🔍 Reference
------------

`N-BEATS: Neural basis expansion analysis for interpretable time series forecasting <https://arxiv.org/abs/1905.10437>`_

------
SCINet
------

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

🔍 Reference
------------

`Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction <https://arxiv.org/abs/2106.09305?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29>`_

-------
Seq2Seq
-------

.. code-block:: yaml

    MODEL:
      NAME: "Seq2Seq"
      # Number of hidden units
      NUM_H_FEATS: 64
      # Number of encoders
      NUM_ENCODERS: 2
      # Number of decoders 
      NUM_DECODERS: 1

🔍 Reference
------------

`Sequence to Sequence Learning with Neural Networks <https://arxiv.org/abs/1409.3215>`_
