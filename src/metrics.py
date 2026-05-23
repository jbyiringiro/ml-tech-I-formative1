"""Evaluation metrics for Task 3.

MAE  = (1/n) * sum |y - yhat|             -- average error, same units as y
RMSE = sqrt((1/n) * sum (y - yhat)^2)     -- penalises big errors more
MAPE = (100/n) * sum |y - yhat| / |y|     -- a percentage, comparable across areas

MAPE is unstable near y=0 so I clip the denominator with a small epsilon and
also report sMAPE as a sanity check.
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
    """Symmetric MAPE -- nicer around zero than MAPE. Bounded in [0, 200]."""
    yt, yp = _align(y_true, y_pred)
    denom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, eps)
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)


def evaluate(y_true, y_pred, model_name: str | None = None) -> dict:
    """Compute MAE/RMSE/MAPE/sMAPE and return them as a dict.
    Also records n and how many actuals are near zero (warns about MAPE).
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
    """Turn a list of evaluate() dicts into a DataFrame."""
    df = pd.DataFrame(results)
    if index_col in df.columns:
        df = df.set_index(index_col)
    return df
