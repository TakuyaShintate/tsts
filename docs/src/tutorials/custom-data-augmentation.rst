===================================
(Tutorial) Custom Data Augmentation
===================================

In this tutorial, we will learn how to define and use a new custom data augmentation method.

(Step 1) Define a new custom data augmentation
==============================================

First, we need to define a new custom data augmentation and register it to `tsts` by using `TRANSFORMS` registry. Let's make flip data augmentation which flips positive/negative sign of input time series.


.. code-block:: python

    import random

    from tsts.core import TRANSFORMS
    from tsts.transforms.transform import Transform


    @TRANSFORMS.register()
    class RandomFlip(Transform):
        def __init__(self, p=0.5):
            super().__init__()
            self.p = p

        def apply(self, X, y, bias, time_stamps):
            # X is input time series which shape is (length, features)
            # y is the target
            if random.random() <= self.p:
                X = -X
                y = -y
            return (X, y, bias, time_stamps)


(Step 2) Use the data augmentation
==================================

We can use the data augmentation defined above by adding the following block to our config.

.. code-block:: yaml

    PIPELINE:
        TRANSFORMS_TRAIN: [{"name": "RandomFlip", "args": {"p": 0.5}}]
