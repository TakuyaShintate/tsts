.. _optimizers:

==========
Optimizers
==========

.. contents:: Catalog
    :depth: 1
    :local:

----
Adam
----

.. code-block:: yaml

    OPTIMIZER:
      NAME: "Adam"
      LR: 0.001
      WEIGHT_DECAY: 1.0E-5
      EPS: 1.0E-8

🔍 Reference
------------

`Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_

-----
AdamW
-----

.. code-block:: yaml

    OPTIMIZER:
        NAME: "AdamW"
        LR: 0.001
        WEIGHT_DECAY: 1.0E-5
        EPS: 1.0E-8

🔍 Reference
------------

`Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_

---
SAM
---

.. code-block:: yaml

    OPTIMIZER:
      NAME: "SAM"
      LR: 0.001
      RHO: 0.05
      BASE_OPTIMIZER_NAME: "Adam"
      WEIGHT_DECAY: 1.0E-5
      EPS: 1.0E-8

🔍 Reference
------------

`Sharpness-Aware Minimization for Efficiently Improving Generalization <https://arxiv.org/abs/2010.01412>`_

---
SGD
---

.. code-block:: yaml

    OPTIMIZER:
      NAME: "SGD"
      LR: 0.001
      MOMENTUM: 0.9
