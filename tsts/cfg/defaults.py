from .cfg import CfgNode as CN

__all__ = ["get_cfg_defaults"]

_C = CN()
# Main device
_C.DEVICE = "cpu"
# Random seed
_C.SEED = 42

_C.IO = CN()
# Number of input time steps
_C.IO.LOOKBACK = 96
# Number of output time steps
_C.IO.HORIZON = 48

_C.TRAINING = CN()
# How to split train dataset {"col" or "row"}
_C.TRAINING.TRAIN_DATA_SPLIT = "col"
# Ratio of train dataset size over val dataset size
_C.TRAINING.TRAIN_DATA_RATIO = 0.75
# Number of epochs
_C.TRAINING.NUM_EPOCHS = 100
# If True, datasets are split randomly (valid only if TRAIN_DATA_SPLIT = "col")
_C.TRAINING.RANDOM_SPLIT = True
# Try to load pretrained model & local scaler in this directory
_C.TRAINING.PRETRAIN = ""

_C.OPTIMIZER = CN()
# Optimizer name
_C.OPTIMIZER.NAME = "Adam"
# Learning rate
_C.OPTIMIZER.LR = 0.001
# L2 penalty factor
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4
# Base optimizer for second step optimizers like SAM
_C.OPTIMIZER.BASE_OPTIMIZER_NAME = "Adam"
# Hyper parameter for SAM
_C.OPTIMIZER.RHO = 0.05
# Value for numerical stability
_C.OPTIMIZER.EPS = 1e-8

_C.SCHEDULER = CN()
# Scheduler name
_C.SCHEDULER.NAME = "CosineAnnealing"
# Maximum number of iterations (it comes from torch)
_C.SCHEDULER.T_MAX = 100
# Minimum learning rate (it comes from torch)
_C.SCHEDULER.ETA_MIN = 0.0
# Every STEP_SIZE steps, GAMMA is multiplied to learning rate used by StepScheduler
_C.SCHEDULER.STEP_SIZE = 1
# Scaling factor used by StepScheduler
_C.SCHEDULER.GAMMA = 0.1
# Decaying parameter 1 (used by ExponentialDecay scheduler)
_C.SCHEDULER.DECAY_RATE = 0.96
# Decaying parameter 2 (used by ExponentialDecay scheduler)
_C.SCHEDULER.DECAY_STEPS = 10000.0
# Number of iterations for the first restart
_C.SCHEDULER.T_0 = 100
# A factor increases T_{i} after a restart
_C.SCHEDULER.T_MULT = 1
# Decreasing factor by cycle (used by CosineAnnealingWarmRestarts)
_C.SCHEDULER.M_MULT = 1.0
# Number of warmup steps
_C.SCHEDULER.WARMUP_STEPS = 0

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
# Expansion rate (Transformer based models and SCINet)
_C.MODEL.EXPANSION_RATE = 4.0
# Input length for decoder (Informer)
_C.MODEL.DECODER_IN_LENGTH = 24
# Stack size of NBeats
_C.MODEL.STACK_SIZE = 30
# Block type of NBeats {"identity", "trend", "seasonal"}
_C.MODEL.BLOCK_TYPE = "identity"
# Polynomial degree
_C.MODEL.DEGREE = 2
# Kernel size (used by SCINet)
_C.MODEL.KERNEL_SIZE = 5
# Use another Linear layer to adjust the number of output feats (used by SCINet)
_C.MODEL.USE_REGRESSOR_ACROSS_TIME = False

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
# Last BASE_END_INDEX samples are not used for training
_C.DATASET.BASE_END_INDEX = -1
# Normalize per dataset differently
_C.DATASET.NORM_PER_DATASET = False

_C.PIPELINE = CN()
# List of transforms
# Each dictionary must contain `name` and `args` pairs
# Ex: [{"name": "GaussianNoise", "args": {"mean": 0.0, "std": 0.001}}]
_C.PIPELINE.TRANSFORMS_TRAIN = []
_C.PIPELINE.TRANSFORMS_VALID = []

_C.X_SCALER = CN()
# Scaler for input time series
_C.X_SCALER.NAME = "StandardScaler"

_C.Y_SCALER = CN()
# Scaler for output time series
_C.Y_SCALER.NAME = "StandardScaler"

_C.COLLATOR = CN()
# Collator name
_C.COLLATOR.NAME = "Collator"

_C.DATALOADER = CN()
# Train dataloader name
_C.DATALOADER.NAME_TRAIN = "DataLoader"
# Validation dataloader name
_C.DATALOADER.NAME_VALID = "DataLoader"
# Batch size of train dataset
_C.DATALOADER.BATCH_SIZE_TRAIN = 100
# Batch size of validation dataset
_C.DATALOADER.BATCH_SIZE_VALID = 100
# Number of workers for train dataset
_C.DATALOADER.NUM_WORKERS_TRAIN = 4
# Number of workers for validation dataset
_C.DATALOADER.NUM_WORKERS_VALID = 4
# If True, shuffle train dataset for every epoch
_C.DATALOADER.SHUFFLE_TRAIN = True
# If True, shuffle Validation dataset for every epoch
_C.DATALOADER.SHUFFLE_VALID = False
# If True, last training batch is dropped
_C.DATALOADER.DROP_LAST_TRAIN = True
# If True, last validation batch is dropped
_C.DATALOADER.DROP_LAST_VALID = False

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
