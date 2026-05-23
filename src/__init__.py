"""Mobile Network Traffic Forecasting -- source package.

Modules
-------
config        : loads ``config.yaml`` and resolves project paths.
data_loading  : memory-efficient streaming/loading of the raw 5 GB dataset (Task 1).
preprocessing : supervised-window construction, scaling, train/val/test split (Task 3).
eda           : exploratory data analysis routines (Task 2).
metrics       : MAE / MAPE / RMSE forecasting metrics (Task 3).
utils         : timing, plotting and small shared helpers.
models        : SARIMA, LSTM and GRU forecasting models (Task 3).
"""

__version__ = "1.0.0"
