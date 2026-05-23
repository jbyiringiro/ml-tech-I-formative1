"""Forecasting evaluation metrics (Task 3.III).

Three metrics are required by the assignment:

* **MAE  -- Mean Absolute Error**
      MAE = (1/n) * sum |y_i - yhat_i|
  Same unit as the traffic signal; robust to outliers; easy to interpret as
  the "average size of the error".

* **RMSE -- Root Mean Squared Error**
      RMSE = sqrt( (1/n) * sum (y_i - yhat_i)^2 )
  Also in the original unit, but squares the errors before averaging, so it
  penalises large mistakes more heavily than MAE. RMSE >= MAE always; a large
  gap between them indicates a few big errors.

* **MAPE -- Mean Absolute Percentage Error**
      MAPE = (100/n) * sum |y_i - yhat_i| / |y_i|
  Scale-free (a percentage), which makes it comparable across the three areas
  even though they have very different traffic volumes. Its weakness is
  instability when y_i -> 0 (night-time troughs); we therefore guard the
  denominator with a small epsilon and additionally report the number of
  points excluded, so the figure is not dominated by near-zero actuals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Denominator guard for percentage errors.
_EPS = 1e-8


def _align(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to flat float64 arrays of equal length."""
    yt = np.asarray(y_true, dtype="float64").ravel()
    yp = np.asarray(y_pred, dtype="float64").ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    return yt, yp


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    yt, yp = _align(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    yt, yp = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true, y_pred, eps: float = _EPS) -> float:
    """Mean Absolute Percentage Error (%), with an epsilon-guarded denominator."""
    yt, yp = _align(y_true, y_pred)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)


def smape(y_true, y_pred, eps: float = _EPS) -> float:
    """Symmetric MAPE (%) -- bounded in [0, 200], well-behaved near zero.

    Reported as a supplementary, more stable companion to MAPE.
    """
    yt, yp = _align(y_true, y_pred)
    denom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, eps)
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)


def evaluate(y_true, y_pred, model_name: str | None = None) -> dict:
    """Compute all metrics and return them as a dictionary.

    Also records ``n`` and ``n_near_zero`` (actuals below 1.0) so the report
    can disclose how trustworthy the MAPE figure is for a given area.
    """
    yt, yp = _align(y_true, y_pred)
    result = {
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
        "MAPE": mape(yt, yp),
        "sMAPE": smape(yt, yp),
        "n": int(yt.size),
        "n_near_zero": int(np.sum(np.abs(yt) < 1.0)),
    }
    if model_name is not None:
        result = {"model": model_name, **result}
    return result


def metrics_table(results: list[dict], index_col: str = "model") -> pd.DataFrame:
    """Assemble a list of :func:`evaluate` dicts into a tidy results table."""
    df = pd.DataFrame(results)
    if index_col in df.columns:
        df = df.set_index(index_col)
    return df
