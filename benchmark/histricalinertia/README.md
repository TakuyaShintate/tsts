# Historical Inertia: A Neglected but Powerful Baseline for Long Sequence Time-series Forecasting

## Citation

```
@article{DBLP:journals/corr/abs-2103-16349,
  author    = {Yue Cui and
               Jiandong Xie and
               Kai Zheng},
  title     = {Historical Inertia: An Ignored but Powerful Baseline for Long Sequence
               Time-series Forecasting},
  journal   = {CoRR},
  volume    = {abs/2103.16349},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.16349},
  eprinttype = {arXiv},
  eprint    = {2103.16349},
  timestamp = {Wed, 07 Apr 2021 15:31:46 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-16349.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## ETT

Download the datasets [here](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)

| Type | Input | Target | Lookback | Horizon | MSE | MAE | Ensemble | Script |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| h1 | OT | OT | 720 | 48 | 0.0788 | 0.2194 | No | [train](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/histricalinertia/train_hi_ett_h1.py)/[test](https://github.com/TakuyaShintate/tsts/tree/main/benchmark/histricalinertia/test_hi_ett_h1.py) |
