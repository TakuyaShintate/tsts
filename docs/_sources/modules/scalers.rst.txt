=======
Scalers
=======

Scales input/output time series to match them to model's input/output range

.. contents:: Catalog
    :depth: 1
    :local:

---------------
Identity Scaler
---------------

Apply identity scaling (values are not changed at all)

.. code-block:: yaml

    X_SCALER:
      NAME: "IdentityScaler"

    Y_SCALER:
      NAME: "IdentityScaler"

--------------
Min-Max Scaler
--------------

Scales values to a certain range (0-1)

.. code-block:: yaml

    X_SCALER:
      NAME: "MinMaxScaler"

    Y_SCALER:
      NAME: "MinMaxScaler"

---------------
Standard Scaler
---------------

Standardizes values

.. code-block:: yaml

    X_SCALER:
      NAME: "StandardScaler"

    Y_SCALER:
      NAME: "StandardScaler"
