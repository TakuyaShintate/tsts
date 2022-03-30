=================================
Train & Test on a Custom Dataset
=================================

Using `tools/train.py` & `tools/test.py`
==========================================

----------------------------------------------------
1. Preparation of data to be used for training
----------------------------------------------------

Save the training data (CSV files), validation data, and test data in their respective directories. The name of the directory is arbitrary. If there are multiple training, validation, and test data, please save them all in their respective directories.

----------------------------------------------------
2. Create a config file
----------------------------------------------------

Create a config file describing the settings during training. You can specify the model, Data Augmentation, Optimizer, Learning Rate Scheduler, etc.

For simplicity, we will use a minimal config file here. Save the following as `my_first_model.yml`.

.. code-block:: yaml

   # Save as `my_first_model.yml`
   LOGGER:
     LOG_DIR: "my-first-model"

----------------------------------------------------
3. Training
----------------------------------------------------

Execute the command below to begin training. Once training begins, model parameters and a log file will be created in the directory specified in 2 (`my-first-model` here). The log file will contain loss and metric values for each epoch.

.. note:: Loss functions and metrics can be changed as well as models

Specify the directory where the training and validation data are stored in `--train-dir` and `--valid-dir`. Specify a list of input variable names and a list of output variable names in `--in-feats` and `--out-feats`.

Sometimes you may want to predict the value of an output variable at the same time as the input variable (i.e., you want to predict the value of an output variable at time t-n to t for the value of an input variable at time t-n to t). In such cases, add the `--lagging` option.

.. code-block:: bash

   python tools/train.py \
      --cfg-name my_first_model.yml \
      --train-dir <dir to contain training data> \
      --valid-dir <dir to contain validation data> \
      --in-feats <list of input feature names> \
      --out-feats <list of output feature names>

----------------------------------------------------
4. Testing a trained model
----------------------------------------------------

After training is complete, the command below can be executed to obtain the prediction results for the test data. CSV files containing the prediction results, the correct labels, and their errors will be saved in the directory specified by `--out-dir`, and images of them plotted. Results are saved for each test data.

.. code-block:: bash

   python tools/test.py \
      --cfg-name my_first_model.yml \
      --train-dir <dir to contain training data> \
      --valid-dir <dir to contain validation data> \
      --test-dir <dir to contain test data> \
      --in-feats <list of input feature names> \
      --out-feats <list of output feature names> \
      --out-dir <result is saved in this directory>