# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

## Citation

```
@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and
               Shanghang Zhang and
               Jieqi Peng and
               Shuai Zhang and
               Jianxin Li and
               Hui Xiong and
               Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference},
  volume    = {35},
  number    = {12},
  pages     = {11106--11115},
  publisher = {{AAAI} Press},
  year      = {2021},
}
```

## ETT

Download the datasets [here](https://robjhyndman.com/data/27-3-Athanasopoulos1.zip)

> MSE and MAE are computed on unnormalized targets

| Type | Lookback | Horizon | MSE | MAE | Ensemble | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|
| h1 | 192 | 48 | 11.48 | 2.82 | No | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/informer/train_ett_h1.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/informer/test_ett_h1.py) |