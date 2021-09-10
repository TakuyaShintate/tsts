# Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models

## Citation

```
@incollection{leguen19dilate,
title = {Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models},
author = {Le Guen, Vincent and Thome, Nicolas},
booktitle = {Advances in Neural Information Processing Systems},
pages = {4191--4203},
year = {2019}
}
```

## ETT

Download the datasets [here](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)

> MSE and MAE are computed on unnormalized targets

| Model | Type | Lookback | Horizon | MSE | MAE | Ensemble | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Informer | h1 | 192 | 48 | 10.46 | 2.62 | No | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/dilate/train_informer_ett_h1.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/dilate/test_informer_ett_h1.py) |

