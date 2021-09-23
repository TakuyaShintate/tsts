# Sequence to Sequence Learning with Neural Networks

## Citation

```
@inproceedings{10.5555/2969033.2969173,
author = {Sutskever, Ilya and Vinyals, Oriol and Le, Quoc V.},
title = {Sequence to Sequence Learning with Neural Networks},
year = {2014},
publisher = {MIT Press},
address = {Cambridge, MA, USA},
booktitle = {Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2},
pages = {3104–3112},
numpages = {9},
location = {Montreal, Canada},
series = {NIPS'14}
}
```

> () indicates normalized scores

## ETT (Uni)

Download the datasets [here](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)

| Type | Input | Target | Lookback | Horizon | MSE | MAE | Ensemble | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| h1 | OT | OT | 96 | 48 | 18.2566 (0.2167) | 3.6065 (0.3929) | No | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/seq2seq/train_seq2seq_ett_h1_48_uni.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/seq2seq/test_seq2seq_ett_h1_48_uni.py) |

## ETT (Multi)

Download the datasets [here](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)

| Type | Input | Target | Lookback | Horizon | MSE | MAE | Ensemble | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| h1 | OT | OT | 96 | 48 | 15.7467 (0.8879) | 2.4629 (0.7020) | No | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/seq2seq/train_seq2seq_ett_h1_48_multi.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/seq2seq/test_seq2seq_ett_h1_48_multi.py) |
