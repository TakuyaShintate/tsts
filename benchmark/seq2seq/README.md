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

## ETT

Download the datasets [here](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)

| Type | Input | Target | Lookback | Horizon | MSE | MAE | Ensemble | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| h1 | OT | OT | 96 | 48 | 0.2200 | 0.3899 | No | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/seq2seq/train_seq2seq_ett_h1.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/seq2seq/test_seq2seq_ett_h1.py) |
| h1 | OT | OT | 96 | 48 | 0.1307 | 0.2806 | AR | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/seq2seq/train_seq2seq_ar_ett_h1.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/seq2seq/test_seq2seq_ar_ett_h1.py) |
