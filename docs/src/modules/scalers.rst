=======
Scalers
=======

Scalers for input and output variables can be changed by adding a `X_SCALER`/`Y_SCALER` section to the config. The `X_SCALER` section represents the input variable scaler and the `Y_SCALER` section represents the output variable scaler.

.. contents:: Catalog
    :depth: 1
    :local:

---------------
Identity Scaler
---------------

Apply identity scaling (values are not changed at all).

.. code-block:: yaml

    X_SCALER:
      NAME: "IdentityScaler"

    Y_SCALER:
      NAME: "IdentityScaler"

--------------
Min-Max Scaler
--------------

Scale values to a certain range (0-1).

.. code-block:: yaml

    X_SCALER:
      NAME: "MinMaxScaler"

    Y_SCALER:
      NAME: "MinMaxScaler"

---------------
Standard Scaler
---------------

Standardize values.

.. code-block:: yaml

    X_SCALER:
      NAME: "StandardScaler"

    Y_SCALER:
      NAME: "StandardScaler"
