=======
Seq2Seq
=======

How to Use
==========

Add following lines to config to use *Seq2Seq*.

.. code-block:: yaml

    MODEL:
      NAME: "Seq2Seq"
      # Number of hidden units
      NUM_H_FEATS: 64
      # Number of encoders
      NUM_ENCODERS: 2
      # Number of decoders 
      NUM_DECODERS: 1

Reference
=========

`Sequence to Sequence Learning with Neural Networks <https://arxiv.org/abs/1409.3215>`_
