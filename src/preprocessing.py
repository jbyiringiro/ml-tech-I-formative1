"""Preprocessing for the forecasting models (Task 3).

Turns a single area's 10-minute traffic series into model-ready tensors:

* **Train / validation / test split** -- strictly chronological. The test
  window (16-22 December) is held out and *never* used for fitting or for
  scaler calibration, satisfying the assignment's leakage constraint.
* **Normalisation** -- the scaler is fitted on the training span only and then
  applied to the whole series. Three options are supported (min-max, standard,
  log-min-max); traffic is non-negative and heavily right-skewed, so
  ``log-minmax`` (log1p followed by min-max) is often the most effective.
* **Supervised windowing** -- a sliding window of ``lookback`` past values is
  used as the input history ``x_t`` and the next value as the one-step-ahead
  target ``x_(t+1)``. Test windows are allowed to draw their input history
  from the final ``lookback`` observations of the training span (these are
  past *actuals*, which is permitted), so the model can predict the very first
  test timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import CONFIG


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
class Scaler:
    """Unified 1-D scaler supporting min-max, standard and log-min-max modes.

    ``log-minmax`` applies ``log1p`` before min-max scaling, which compresses
    the long right tail of the traffic distribution and tends to stabilise
    neural-network training.
    """

    def __init__(self, kind: str = "minmax") -> None:
        if kind not in {"minmax", "standard", "log-minmax"}:
            raise ValueError(f"unknown scaler kind: {kind}")
        self.kind = kind
        self._use_log = kind == "log-minmax"
        self._sk = StandardScaler() if kind == "standard" else MinMaxScaler()
        self._fitted = False

    def fit(self, values) -> "Scaler":
        arr = np.asarray(values, dtype="float64").reshape(-1, 1)
        if self._use_log:
            arr = np.log1p(arr)
        self._sk.fit(arr)
        self._fitted = True
        return self

    def transform(self, values) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler.transform called before fit")
        arr = np.asarray(values, dtype="float64").reshape(-1, 1)
        if self._use_log:
            arr = np.log1p(arr)
        return self._sk.transform(arr).astype("float32").ravel()

    def inverse_transform(self, values) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler.inverse_transform called before fit")
        arr = np.asarray(values, dtype="float64").reshape(-1, 1)
        arr = self._sk.inverse_transform(arr)
        if self._use_log:
            arr = np.expm1(arr)
        return arr.astype("float32").ravel()


# ---------------------------------------------------------------------------
# Date-based split of the raw series (used by SARIMA)
# ---------------------------------------------------------------------------
def test_window_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask selecting the Task 3 test week (16-22 December, inclusive)."""
    start = pd.Timestamp(CONFIG.dates["test_start"])
    end = pd.Timestamp(CONFIG.dates["test_end"]) + pd.Timedelta(days=1)
    return (index >= start) & (index < end)


def train_test_split_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split a full area series into (train, test) at the test-week boundary.

    ``train`` is everything strictly before 16 December; ``test`` is the
    16-22 December week. Returned in original (un-scaled) units -- this is the
    form the SARIMA model consumes directly.
    """
    mask = test_window_mask(series.index)
    return series[~mask].copy(), series[mask].copy()


# ---------------------------------------------------------------------------
# Supervised windowing for the neural models
# ---------------------------------------------------------------------------
def make_windows(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice a 1-D array into supervised (X, y) one-step-ahead windows.

    For an array of length ``T`` returns ``X`` of shape
    ``(T - lookback, lookback)`` and ``y`` of shape ``(T - lookback,)``; window
    ``k`` predicts the value immediately after its ``lookback`` inputs.
    """
    values = np.asarray(values, dtype="float32")
    if len(values) <= lookback:
        raise ValueError(
            f"series length {len(values)} <= lookback {lookback}; cannot window"
        )
    n = len(values) - lookback
    X = np.empty((n, lookback), dtype="float32")
    for k in range(n):
        X[k] = values[k : k + lookback]
    y = values[lookback:].astype("float32")
    return X, y


@dataclass
class ForecastData:
    """Container holding everything needed to train and evaluate one area."""

    square_id: int
    lookback: int
    scaler: Scaler
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray                 # scaled test targets
    y_test_actual: np.ndarray          # test targets in original units
    test_index: pd.DatetimeIndex       # timestamps of the test targets
    meta: dict = field(default_factory=dict)

    @property
    def input_shape(self) -> tuple[int, int]:
        """Keras input shape (timesteps, features)."""
        return (self.lookback, 1)

    def inverse(self, scaled_values) -> np.ndarray:
        """Map scaled model outputs back to original traffic units."""
        return self.scaler.inverse_transform(scaled_values)


def prepare_forecasting_data(
    series: pd.Series,
    lookback: int | None = None,
    val_fraction: float | None = None,
    scaler_kind: str | None = None,
) -> ForecastData:
    """Build train/val/test tensors for the neural models from one area series.

    Parameters
    ----------
    series : full traffic series for one area (datetime index, 10-min steps).
    lookback : input history length; defaults to ``config.yaml`` value.
    val_fraction : fraction of the *training windows* (most recent, contiguous)
        used for early-stopping validation.
    scaler_kind : 'minmax' | 'standard' | 'log-minmax'.
    """
    lookback = lookback or CONFIG.forecast["lookback"]
    val_fraction = (
        val_fraction if val_fraction is not None else CONFIG.forecast["val_fraction"]
    )
    scaler_kind = scaler_kind or CONFIG.forecast["scaler"]

    series = series.sort_index()
    index = series.index
    test_mask = test_window_mask(index)
    if test_mask.sum() == 0:
        raise ValueError(
            "No timestamps fall inside the test week; check the series date range."
        )

    # Scaler is fitted ONLY on the pre-test span -> no leakage.
    train_span_values = series[~test_mask].to_numpy(dtype="float32")
    scaler = Scaler(scaler_kind).fit(train_span_values)
    scaled_full = scaler.transform(series.to_numpy(dtype="float32"))

    # Window the entire scaled series, then route each window by target date.
    X_all, y_all = make_windows(scaled_full, lookback)
    target_index = index[lookback:]                       # timestamp of each y
    target_is_test = test_window_mask(target_index)

    X_pool, y_pool = X_all[~target_is_test], y_all[~target_is_test]
    X_test, y_test = X_all[target_is_test], y_all[target_is_test]
    test_index = target_index[target_is_test]

    # Chronological train/val split of the non-test pool.
    n_val = max(1, int(round(len(X_pool) * val_fraction)))
    split = len(X_pool) - n_val
    X_train, y_train = X_pool[:split], y_pool[:split]
    X_val, y_val = X_pool[split:], y_pool[split:]

    # Reshape to (samples, timesteps, 1 feature) for the recurrent nets.
    reshape = lambda a: a.reshape(a.shape[0], a.shape[1], 1)
    y_test_actual = scaler.inverse_transform(y_test)

    return ForecastData(
        square_id=int(series.name.split("_")[-1]) if isinstance(series.name, str)
        and "_" in str(series.name) else -1,
        lookback=lookback,
        scaler=scaler,
        X_train=reshape(X_train),
        y_train=y_train,
        X_val=reshape(X_val),
        y_val=y_val,
        X_test=reshape(X_test),
        y_test=y_test,
        y_test_actual=y_test_actual,
        test_index=test_index,
        meta={
            "scaler_kind": scaler_kind,
            "val_fraction": val_fraction,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        },
    )
