# N-BEATS: Neural basis expansion analysis for interpretable time series forecasting

## Citation

```
@inproceedings{
  Oreshkin2020:N-BEATS,
  title={{N-BEATS}: Neural basis expansion analysis for interpretable time series forecasting},
  author={Boris N. Oreshkin and Dmitri Carpov and Nicolas Chapados and Yoshua Bengio},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=r1ecqn4YwB}
}
```

## ETT (Uni)

Download the datasets [here](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)

| Type | Input | Target | Lookback | Horizon | MSE | MAE | LocalScaler | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| h1 | OT | OT | 96 | 48 | 3.7264 | 1.4755 | LastStep | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/nbeats/train_nbeats_ett_h1_48_uni.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/nbeats/test_nbeats_ett_h1_48_uni.py) |

## ETT (Multi)

Download the datasets [here](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)

| Type | Input | Target | Lookback | Horizon | MSE | MAE | LocalScaler | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| h1 | HUFL HULL MUFL MULL <br> LUFL LULL OT | HUFL HULL MUFL MULL <br> LUFL LULL OT | 96 | 48 | 20.3007 | 2.5492 | LastStep | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/nbeats/train_nbeats_ett_h1_48_multi.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/nbeats/test_nbeats_ett_h1_48_multi.py) |

## Tourism

Download the datasets [here](https://robjhyndman.com/data/27-3-Athanasopoulos1.zip)

| Interval | Lookback | Horizon | MAPE | LocalScaler | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Monthly | 2H | 24 | 20.12 | LastStep | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/nbeats/train_tourism_monthly.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/nbeats/test_tourism_monthly.py) |

