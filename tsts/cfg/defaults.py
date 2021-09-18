from .cfg import CfgNode as CN

__all__ = ["get_cfg_defaults"]

_C = CN()
# Main device
_C.DEVICE = "cpu"
# Random seed
_C.SEED = 42

_C.IO = CN()
# Number of input time steps
_C.IO.LOOKBACK = 100
# Number of output time steps
_C.IO.HORIZON = 8

_C.TRAINING = CN()
# How to split train dataset {"col" or "row"}
_C.TRAINING.TRAIN_DATA_SPLIT = "col"
# Ratio of training dataset over VALIDidation dataset
_C.TRAINING.TRAIN_DATA_RATIO = 0.75
# Number of epochs (epoch means a single iteration over whole training + VALIDidation dataset)
_C.TRAINING.NUM_EPOCHS = 100

_C.OPTIMIZER = CN()
# Optimizer name
_C.OPTIMIZER.NAME = "Adam"
# Learning rate
_C.OPTIMIZER.LR = 0.001
# L2 penalty factor
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4

_C.SCHEDULER = CN()
# Scheduler name
_C.SCHEDULER.NAME = "CosineAnnealing"
# Maximum number of iterations (it comes from torch)
_C.SCHEDULER.T_MAX = 10
# Minimum learning rate (it comes from torch)
_C.SCHEDULER.ETA_MIN = 0.0
# Every STEP_SIZE steps, GAMMA is multiplied to learning rate used by StepScheduler
_C.SCHEDULER.STEP_SIZE = 1
# Scaling factor used by StepScheduler
_C.SCHEDULER.GAMMA = 0.1

_C.TRAINER = CN()
# Trainer name
_C.TRAINER.NAME = "SupervisedTrainer"
# Maximum gradient norm
_C.TRAINER.MAX_GRAD_NORM = 1.0
# Denormalize before computing metric values
_C.TRAINER.DENORM = False

_C.SOLVER = CN()
# Solver name
_C.SOLVER.NAME = "Forecaster"

_C.MODEL = CN()
# Model name
_C.MODEL.NAME = "Seq2Seq"
# Number of hidden units in encoder and decoder
_C.MODEL.NUM_H_FEATS = 64
# Number of hidden layers in encoder and decoder
_C.MODEL.DEPTH = 2
# Number of encoders (Seq2Seq, Informer)
_C.MODEL.NUM_ENCODERS = 2
# Number of decoder (Seq2Seq, Informer)
_C.MODEL.NUM_DECODERS = 1
# Number of heads for multi-head self attention
_C.MODEL.NUM_HEADS = 8
# Scale factor of how many queries and keys are sampled (Informer)
_C.MODEL.CONTRACTION_FACTOR = 5
# Dropout rate
_C.MODEL.DROPOUT_RATE = 0.1
# Feed forward expansion rate (Transformer based models)
_C.MODEL.FF_EXPANSION_RATE = 4.0
# Input length for decoder (Informer)
_C.MODEL.DECODER_IN_LENGTH = 24
# Stack size of NBeats
_C.MODEL.STACK_SIZE = 30
# Block type of NBeats {"identity", "trend", "seasonal"}
_C.MODEL.BLOCK_TYPE = "identity"
# Polynomial degree
_C.MODEL.DEGREE = 2

_C.LOCALSCALER = CN()
# Local scaler name
_C.LOCALSCALER.NAME = "NOOP"
# Order p i.e. AR(p)
_C.LOCALSCALER.NUM_STEPS = 100

_C.LOSSES = CN()
# Loss function names
_C.LOSSES.NAMES = ["MSE"]
# Loss function arguments
_C.LOSSES.ARGS = [{}]
# Loss function weights
_C.LOSSES.WEIGHT_PER_LOSS = [1.0]

_C.METRICS = CN()
# Metric names
_C.METRICS.NAMES = ["RMSE"]
# Metric arguments
_C.METRICS.ARGS = [{}]

_C.DATASET = CN()
# Train dataset name
_C.DATASET.NAME_TRAIN = "Dataset"
# Validation dataset name
_C.DATASET.NAME_VALID = "Dataset"
# Dataset index starts with this value
_C.DATASET.BASE_START_INDEX = 0
# Normalize per dataset differently
_C.DATASET.NORM_PER_DATASET = False

_C.SCALER = CN()
# Scaler name
_C.SCALER.NAME = "StandardScaler"

_C.COLLATOR = CN()
# Collator name
_C.COLLATOR.NAME = "Collator"

_C.DATALOADER = CN()
# Train dataloader name
_C.DATALOADER.NAME_TRAIN = "DataLoader"
# Validation dataloader name
_C.DATALOADER.NAME_VALID = "DataLoader"
# Batch size for train dataset
_C.DATALOADER.BATCH_SIZE_TRAIN = 100
# Batch size for Validation dataset
_C.DATALOADER.BATCH_SIZE_VALID = 100
# If True, shuffle train dataset for every epoch
_C.DATALOADER.SHUFFLE_TRAIN = True
# If True, shuffle Validation dataset for every epoch
_C.DATALOADER.SHUFFLE_VALID = False

_C.LOGGER = CN()
_C.LOGGER.NAME = "Logger"
# Log directory name (if "auto", it is randomly generated)
_C.LOGGER.LOG_DIR = "auto"


def get_cfg_defaults() -> CN:
    """Return global configuration.

    Returns
    -------
    CN
        Global configuration
    """
    return _C.clone()
