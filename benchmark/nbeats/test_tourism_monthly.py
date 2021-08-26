import numpy as np
import pandas as pd
import torch
from tsts.metrics import MAPE
from tsts.solvers import TimeSeriesForecaster

X = pd.read_csv("path/to/monthly_in.csv")
X = X.values[3:].T
X = [x[~np.isnan(x)] for x in X]
X = [torch.tensor(x, dtype=torch.float32) for x in X]
X = [x.unsqueeze(-1) for x in X]

Y = pd.read_csv("path/to/monthly_oos.csv")
Y = Y.values[3:].T
Y = [y[~np.isnan(y)] for y in Y]
Y = [torch.tensor(y, dtype=torch.float32) for y in Y]
Y = [y.unsqueeze(-1) for y in Y]

forecaster = TimeSeriesForecaster("./tourism_monthly.yml")
metric = MAPE()

for (x, y) in zip(X, Y):
    z = forecaster.predict(x)
    z = z.unsqueeze(0)
    y = y.unsqueeze(0)
    y_mask = torch.ones_like(y)
    metric.update(z, y, y_mask)

score = metric()
print(f"MAPE: {score}")
