import os
import random

import numpy as np
import torch

__all__ = ["set_random_seed"]


def set_random_seed(seed: int = 42) -> None:
    """Enforce deterministic behavior.

    Parameters
    ----------
    seed : int, optional
        Random seed, by default 42
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
