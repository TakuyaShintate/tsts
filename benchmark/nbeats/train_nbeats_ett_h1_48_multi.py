import pandas as pd
import torch
from tsts.solvers import TimeSeriesForecaster

start = 0
end = 12 * 30 * 24 + 4 * 30 * 24

X = pd.read_csv("/path/to/ETTh1.csv")
X = X[X.columns[1:]]
X = X.values
X = X[start:end]
X = torch.tensor(X, dtype=torch.float32)

solver = TimeSeriesForecaster("./nbeats-ett-h1-48-multi.yml")
solver.fit([X])
