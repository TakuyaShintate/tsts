========
Informer
========

How to Use
==========

Add following lines to config to use *Informer*.

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

Reference
=========

`Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting <https://arxiv.org/abs/2012.07436>`_
