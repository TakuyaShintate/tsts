from .cfg import CfgNode as CN

__all__ = ["get_cfg_defaults"]

_C = CN()
_C.DEVICE = "cpu"

_C.IO = CN()
# Number of input time steps
_C.IO.LOOKBACK = 100
# Number of output time steps
_C.IO.HORIZON = 1

_C.TRAINING = CN()
# Ratio of training dataset over validation dataset
_C.TRAINING.TRAIN_DATA_RATIO = 0.75
# Number of epochs (epoch means a single iteration over whole training + validation dataset)
_C.TRAINING.NUM_EPOCHS = 100

_C.OPTIMIZER = CN()
# OPTIM name
_C.OPTIMIZER.NAME = "SGD"
# Learning rate
_C.OPTIMIZER.LR = 0.01
# Momentum factor
_C.OPTIMIZER.MOMENTUM = 0.9
# L2 penalty factor
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4

_C.TRAINER = CN()
# Trainer name
_C.TRAINER.NAME = "SupervisedTrainer"

_C.SOLVER = CN()
# Solver name
_C.SOLVER.NAME = "Forecaster"

_C.MODEL = CN()
# Model name
_C.MODEL.NAME = "Seq2Seq"
# Number of hidden units in encoder and decoder
_C.MODEL.NUM_H_UNITS = 64
# Number of hidden layers in encoder and decoder
_C.MODEL.DEPTH = 2

_C.LOSSES = CN()
# Loss function names
_C.LOSSES.NAMES = ["MSE"]
# Loss function arguments
_C.LOSSES.ARGS = [{}]
# Loss function weights
_C.LOSSES.WEIGHT_PER_LOSS = [1.0]

_C.DATASET = CN()
# Train dataset name
_C.DATASET.NAME_TRAIN = "Dataset"
# Validation dataset name
_C.DATASET.NAME_VAL = "Dataset"

_C.COLLATOR = CN()
_C.COLLATOR.TRAIN_NAME = "Collator"

_C.DATALOADER = CN()
# Train dataloader name
_C.DATALOADER.NAME_TRAIN = "DataLoader"
# Validation dataloader name
_C.DATALOADER.NAME_VAL = "DataLoader"
# Batch size for train dataset
_C.DATALOADER.BATCH_SIZE_TRAIN = 100
# Batch size for validation dataset
_C.DATALOADER.BATCH_SIZE_VAL = 100
# If True, shuffle train dataset for every epoch
_C.DATALOADER.SHUFFLE_TRAIN = True
# If True, shuffle validation dataset for every epoch
_C.DATALOADER.SHUFFLE_VAL = False


def get_cfg_defaults() -> CN:
    """Return global configuration.

    Returns
    -------
    CN
        Global configuration
    """
    return _C.clone()
