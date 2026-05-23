"""Shared result container for all forecasting models."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Uniform output of every forecasting model.

    Attributes
    ----------
    model_name : 'SARIMA' | 'LSTM' | 'GRU'.
    square_id : the area the model was trained and evaluated on.
    y_true / y_pred : test-week actual and predicted traffic, original units.
    test_index : timestamps of the test week (16-22 December).
    train_time_s / predict_time_s : wall-clock training and inference times.
    extra : free-form dict for model-specific diagnostics (e.g. Keras history).
    """

    model_name: str
    square_id: int
    y_true: np.ndarray
    y_pred: np.ndarray
    test_index: pd.DatetimeIndex
    train_time_s: float
    predict_time_s: float
    extra: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy (timestamp-indexed) DataFrame of actual vs predicted."""
        return pd.DataFrame(
            {"y_true": np.asarray(self.y_true).ravel(),
             "y_pred": np.asarray(self.y_pred).ravel()},
            index=self.test_index,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ForecastResult(model={self.model_name}, area={self.square_id}, "
            f"n_test={len(self.y_true)}, train={self.train_time_s:.2f}s, "
            f"predict={self.predict_time_s:.3f}s)"
        )
