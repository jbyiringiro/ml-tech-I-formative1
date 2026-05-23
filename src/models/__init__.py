"""Forecasting models for Task 3.

* :class:`SarimaForecaster` -- the classical statistical model (Seasonal ARIMA).
* :class:`NeuralForecaster` -- a configurable recurrent network; instantiated
  as an LSTM or a GRU via the ``cell`` argument.

All models return a :class:`ForecastResult`, giving the run scripts a uniform
interface for evaluation and plotting.
"""
from .base import ForecastResult
from .sarima_model import SarimaForecaster
from .neural import NeuralForecaster, build_lstm, build_gru

__all__ = [
    "ForecastResult",
    "SarimaForecaster",
    "NeuralForecaster",
    "build_lstm",
    "build_gru",
]
