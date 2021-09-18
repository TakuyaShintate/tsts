======
NBeats
======

How to Use
==========

Add following lines to config to use *NBeats*.

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

Reference
=========

`N-BEATS: Neural basis expansion analysis for interpretable time series forecasting <https://arxiv.org/abs/1905.10437>`_
