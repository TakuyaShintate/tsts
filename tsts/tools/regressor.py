import numpy as np
from tsts.tools.tool import Tool

__all__ = ["Forecaster"]


class Forecaster(Tool):
    """Tool to solve time series forecasting.

    Parameters
    ----------
    lookback : int, optional
        Number of previous timesteps used to predict the subsequent timesteps, by default 100

    horizon : int, optional
        Number of the subsequent timesteps predicted, by default 1
    """

    def __init__(self, lookback: int = 100, horizon: int = 1) -> None:
        super(Forecaster, self).__init__()
        self.lookback = lookback
        self.horizon = horizon

    def fit(self, dataset: np.ndarray) -> None:
        """Train the target model on given dataset.

        Parameters
        ----------
        dataset : np.ndarray
            Dataset on witch the target model is trained
        """
        pass
