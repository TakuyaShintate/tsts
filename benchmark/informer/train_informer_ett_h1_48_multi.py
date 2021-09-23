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

time_stamps = pd.read_csv("/path/to/ETTh1.csv")
time_stamps = time_stamps[["date"]]
time_stamps["date"] = pd.to_datetime(time_stamps.date)
time_stamps["month"] = time_stamps.date.apply(lambda r: r.month, 1)
time_stamps["weekday"] = time_stamps.date.apply(lambda r: r.weekday(), 1)
time_stamps["day"] = time_stamps.date.apply(lambda r: r.day, 1)
time_stamps["hour"] = time_stamps.date.apply(lambda r: r.hour, 1)
time_stamps["minute"] = time_stamps.date.apply(lambda r: r.minute, 1)
time_stamps["minute"] = time_stamps.minute.map(lambda r: r // 15)
time_stamps = time_stamps[["month", "weekday", "day", "hour"]]
time_stamps = time_stamps.values
time_stamps = time_stamps[start:end]
time_stamps = torch.tensor(time_stamps, dtype=torch.long)

solver = TimeSeriesForecaster("./informer-ett-h1-48-multi.yml")
solver.fit([X], time_stamps=[time_stamps])
