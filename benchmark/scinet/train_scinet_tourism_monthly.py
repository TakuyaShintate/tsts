import numpy as np
import pandas as pd
import torch
from tsts.solvers import TimeSeriesForecaster

X = pd.read_csv("/path/to/monthly_in.csv")
X = X.values[3:].T
X = [x[~np.isnan(x)] for x in X]
X = [torch.tensor(x, dtype=torch.float32) for x in X]
X = [x.unsqueeze(-1) for x in X]

forecaster = TimeSeriesForecaster("./scinet_tourism_monthly.yml")
forecaster.fit(X)
