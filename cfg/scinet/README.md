# [Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction](https://arxiv.org/abs/2106.09305)

## ETTh1

The first 8640 data points were used for training. Of the remaining data points, 2880-n were used for validation and the last 8640-n for testing, where n is the length of lookback. MAE and MSE values were calculated using standardized predictions. "HUFL HULL MUFL MULL LUFL LULL OT" are the input/output variables selected.

| lookback | horizon | val | test | config |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| 48 | 24 | mae=0.4109 <br> mse=0.3924 | mae=0.3590 <br> mse=0.3213 | [link](scinet_etth1_l48_h24.yml) |
| 96 | 48 | mae=0.4837 <br> mse=0.5216 | mae=0.4391 <br> mse=0.4134 | [link](scinet_etth1_l96_h48.yml) |
| 336 | 168 | mae=0.6229 <br> mse=0.8526 | mae=0.4874 <br> mse=0.4941 | [link](scinet_etth1_l336_h168.yml) |
| 336 | 336 | mae=0.7123 <br> mse=1.0756 | mae=0.4777 <br> mse=0.4947 | [link](scinet_etth1_l336_h336.yml) |
| 736 | 720 | mae=0.7897 <br> mse=1.2429 | mae=0.5525 <br> mse=0.5785 | [link](scinet_etth1_l736_h720.yml) |