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

> () indicates normalized scores

## ETT (Uni)

Download the datasets [here](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)

| Type | Input | Target | Lookback | Horizon | MSE | MAE | Ensemble | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| h1 | OT | OT | 720 | 48 | 12.9825 (0.1538) | 2.9829 (0.3247) | No | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/informer/train_informer_ett_h1_48_uni.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/informer/test_informer_ett_h1_48_uni.py) |

## ETT (Multi)

Download the datasets [here](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)

| Type | Input | Target | Lookback | Horizon | MSE | MAE | Ensemble | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| h1 | HUFL HULL MUFL MULL <br> LUFL LULL OT | HUFL HULL MUFL MULL <br> LUFL LULL OT | 96 | 48 | 25.5204 (0.8541) | 2.9924 (0.7042) | No | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/informer/train_informer_ett_h1_48_multi.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/informer/test_informer_ett_h1_48_multi.py) |
