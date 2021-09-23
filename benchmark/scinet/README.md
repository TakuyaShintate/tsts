# Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction

## Citation

```
@article{DBLP:journals/corr/abs-2106-09305,
  author    = {Minhao Liu and
               Ailing Zeng and
               Qiuxia Lai and
               Qiang Xu},
  title     = {Time Series is a Special Sequence: Forecasting with Sample Convolution
               and Interaction},
  journal   = {CoRR},
  volume    = {abs/2106.09305},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.09305},
  eprinttype = {arXiv},
  eprint    = {2106.09305},
  timestamp = {Tue, 29 Jun 2021 16:55:04 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2106-09305.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

> () indicates normalized scores

## ETT (Uni)

Download the datasets [here](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)

| Type | Input | Target | Lookback | Horizon | MSE | MAE | LocalScaler | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| h1 | OT | OT | 96 | 48 | 5.4206 (0.0643) | 1.7550 (0.1912) | No | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/scinet/train_scinet_ett_h1_48_uni.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/scinet/test_scinet_ett_h1_48_uni.py) |

## ETT (Multi)

Download the datasets [here](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)

| Type | Input | Target | Lookback | Horizon | MSE | MAE | LocalScaler | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| h1 | HUFL HULL MUFL MULL <br> LUFL LULL OT | HUFL HULL MUFL MULL <br> LUFL LULL OT | 96 | 48 | 8.6568 (0.3850) | 1.6065 (0.4150) | No | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/scinet/train_scinet_ett_h1_48_multi.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/scinet/test_scinet_ett_h1_48_multi.py) |

## Tourism

Download the datasets [here](https://robjhyndman.com/data/27-3-Athanasopoulos1.zip)

| Interval | Lookback | Horizon | MAPE | LocalScaler | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Monthly | 2H | 24 | 22.13 | LastStep | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/nbeats/train_scinet_tourism_monthly.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/nbeats/test_scinet_tourism_monthly.py) |
