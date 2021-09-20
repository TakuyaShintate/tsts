import pandas as pd
import torch
from tqdm import tqdm
from tsts.metrics import MAE, MSE
from tsts.scalers import StandardScaler
from tsts.solvers import TimeSeriesForecaster

lookback = 96
horizon = 48
start = 12 * 30 * 24 + 4 * 30 * 24 + lookback
end = 12 * 30 * 24 + 8 * 30 * 24 + lookback - horizon

X = pd.read_csv("/path/to/ETTh1.csv")
X = X[["OT"]]
X = X.values
X = torch.tensor(X, dtype=torch.float32)

solver = TimeSeriesForecaster("./scinet-ett-h1.yml")

# Initialize scalers with training dataset
num_train_samples = int(0.75 * (12 * 30 * 24 + 4 * 30 * 24))
X_scaler = StandardScaler()
X_scaler.fit(X[:num_train_samples])
y_scaler = StandardScaler()
y_scaler.fit(X[:num_train_samples])

metric1 = MSE()
metric2 = MAE()

for i in tqdm(range(start, end)):
    x = X[i - lookback : i]
    y = X[i : i + horizon]
    y_mask = torch.ones_like(y)
    x = X_scaler.transform(x)
    y = y_scaler.transform(y)
    Z = solver.predict(x)
    metric1.update(Z, y, y_mask)
    metric2.update(Z, y, y_mask)

score1 = metric1()
score2 = metric2()
print(f"MSE: {score1}")
print(f"MAE: {score2}")