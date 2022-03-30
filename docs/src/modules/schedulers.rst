.. _schedulers:

=============
LR Schedulers
=============

The learning rate scheduler can be changed by adding a `SCHEDULER` section to the config. The default learning rate scheduler is `CosineAnnealing`.

.. contents:: Catalog
    :depth: 1
    :local:

----------------
Cosine Annealing
----------------

.. code-block:: yaml

    SCHEDULER:
        NAME: "CosineAnnealing"
        T_MAX: 10

------------------------------
Cosine Annealing Warm Restarts
------------------------------

.. code-block:: yaml

    SCHEDULER:
        NAME: "CosineAnnealingWithRestarts"
        T_MAX: 10

-----------------
Exponential Decay
-----------------

.. code-block:: yaml

    SCHEDULER:
        NAME: "ExponentialDecay"
        DECAY_RATE: 0.96

--------
Identity
--------

.. code-block:: yaml

    SCHEDULER:
      NAME: "IdentityScheduler"

----
Step
----

.. code-block:: yaml

    SCHEDULER:
      NAME: "StepScheduler"
      STEP_SIZE: 10
      GAMMA: 0.1
