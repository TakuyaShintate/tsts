=========================
Train on a Custom Dataset
=========================

**tools/train.py** can be used to train forecasting models on a custom dataset.

--------------------------
1. Create a Custom Dataset
--------------------------

Put **.csv** files to train directory (used for training) and valid directory (used for validation).

.. note:: Any directory names can be used.

🗂 Directory Structure
----------------------

.. image:: ../../../img/train-tool.jpg
   :scale: 100%
   :align: center

----------------
2. Create Config
----------------

To specify training settings, create config file.

.. note:: See :ref:`config structure <config>` for the details.

📝 Config
---------

.. code-block::

   LOGGER:
     LOG_DIR: "./log"

   MODEL:
     NAME: "SCINet"
     DEPTH: 3

--------------------------
3. Train Forecasting Model
--------------------------

Run **tools/train.py** with the custom dataset and config. Input/output feature names have to be specified too.

.. code-block:: bash

   python tools/train.py \
    --cfg-name config.yml \
    --train-dir train \
    --valid-dir valid \
    --in-feats input_feat1 input_feat2 \
    --out-feats output_feat1 output_feat2 \