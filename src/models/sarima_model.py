"""Seasonal ARIMA -- the classical statistical forecaster (Task 3).

SARIMA(p, d, q)(P, D, Q)_s extends ARIMA with a seasonal component. For
10-minute traffic the dominant cycle is daily, so the seasonal period is
s = 144 (= 24 h x 6).

Tractable formulation
---------------------
A full state-space SARIMAX with a *seasonal period of 144* is extremely
expensive: the latent state grows with ``s``, so maximum-likelihood estimation
of seasonal AR/MA terms (P, Q > 0) takes minutes per fit and the rolling
forecast becomes intractable. We therefore use the standard, equivalent
**differencing formulation** of SARIMA(p, d, q)(0, D, 0)_s:

1. Apply seasonal differencing ``D`` times at lag ``s``:
   ``y_t = x_t - x_{t-s}`` (for D = 1). This removes the daily cycle and
   yields an (approximately) stationary series.
2. Fit a non-seasonal ARIMA(p, d, q) to ``y_t``.
3. Forecast ``y`` one step ahead, then **invert the differencing** with the
   actual lagged observation: ``x_hat_t = y_hat_t + x_{t-s}``.

This is mathematically a SARIMA model -- the seasonal behaviour is captured by
the seasonal difference (a "seasonal random walk"), and the ARIMA term models
the remaining short-range structure. It keeps the latent state small, so a fit
takes seconds rather than minutes. Seasonal AR/MA terms (P, Q > 0) were also
tried during experimentation but added large computational cost for marginal
accuracy gain (see ``experiment.py`` / the report).

One-step-ahead protocol
-----------------------
The ARIMA is fitted **once** on the (seasonally-differenced) training span.
The true test observations are then differenced, appended
(``append(..., refit=False)`` -- no re-estimation) and the one-step-ahead
predictions for the whole test week are read off in a single Kalman-filter
pass via ``predict``. With actual values used as the lagged history this is
exactly the rolling one-step-ahead forecast the assignment specifies.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..config import CONFIG
from ..preprocessing import train_test_split_series
from ..utils import timer
from .base import ForecastResult


def _square_id_of(series: pd.Series) -> int:
    name = str(series.name)
    if "_" in name and name.split("_")[-1].lstrip("-").isdigit():
        return int(name.split("_")[-1])
    return -1


class SarimaForecaster:
    """Seasonal ARIMA one-step-ahead forecaster (differencing formulation)."""

    name = "SARIMA"

    def __init__(
        self,
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int, int] | None = None,
        train_tail_days: int | None = None,
        maxiter: int | None = None,
    ) -> None:
        cfg = CONFIG.sarima
        self.order = tuple(order or cfg["order"])
        seas = tuple(seasonal_order or cfg["seasonal_order"])
        self.seasonal_order = seas
        # We use the seasonal differencing order D and the period s; seasonal
        # AR/MA terms (P, Q) are intentionally not estimated -- see module docs.
        self.seasonal_diff = int(seas[1])       # D
        self.seasonal_period = int(seas[3])     # s
        if seas[0] or seas[2]:
            print(f"[SARIMA] note: seasonal AR/MA terms P={seas[0]}, Q={seas[2]} "
                  f"are not estimated in the tractable formulation; "
                  f"using seasonal differencing D={self.seasonal_diff} at s="
                  f"{self.seasonal_period}.")
        # train_tail_days kept for API compatibility; None -> use all training data.
        self.train_tail_days = train_tail_days if train_tail_days is not None \
            else cfg.get("train_tail_days")
        self.maxiter = maxiter or cfg["maxiter"]
        self.result_ = None          # fitted ARIMA results on the differenced series

    # ------------------------------------------------------------------
    def _seasonal_difference(self, values: np.ndarray) -> np.ndarray:
        """Apply seasonal differencing D times at lag s."""
        out = values
        for _ in range(self.seasonal_diff):
            out = out[self.seasonal_period:] - out[: -self.seasonal_period]
        return out

    # ------------------------------------------------------------------
    def fit_predict(self, series: pd.Series) -> ForecastResult:
        """Fit on the training span and roll one-step-ahead over the test week."""
        square_id = _square_id_of(series)
        train, test = train_test_split_series(series)

        if self.train_tail_days:                       # optional recency window
            tail = self.train_tail_days * CONFIG.dataset["intervals_per_day"]
            train = train.iloc[-tail:] if len(train) > tail else train

        train_values = train.to_numpy(dtype="float64")
        test_values = test.to_numpy(dtype="float64")
        full = np.concatenate([train_values, test_values])
        n_tr = len(train_values)
        s, D = self.seasonal_period, self.seasonal_diff

        # -- seasonally difference the training data -----------------------
        y_train = self._seasonal_difference(train_values)

        # -- one-off ARIMA fit on the differenced series -------------------
        with timer(f"SARIMA fit (area {square_id})") as t_fit:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y_train,
                    order=self.order,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    concentrate_scale=True,
                )
                self.result_ = model.fit(disp=False, maxiter=self.maxiter)

        # -- one-step-ahead forecasts over the test week -------------------
        with timer(f"SARIMA rolling forecast (area {square_id})") as t_pred:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if D >= 1:
                    # y_t = x_t - x_{t-s}; the lag term is an actual observation.
                    lag = full[n_tr - s: n_tr - s + len(test_values)]
                    y_test = test_values - lag
                else:
                    lag = np.zeros(len(test_values))
                    y_test = test_values
                res_all = self.result_.append(y_test, refit=False)
                y_pred = np.asarray(
                    res_all.predict(start=len(y_train),
                                    end=len(y_train) + len(test_values) - 1)
                ).astype("float64")
                preds = y_pred + lag                    # invert the differencing

        # Traffic cannot be negative -- clip physically impossible forecasts.
        preds = np.clip(preds, 0.0, None)

        return ForecastResult(
            model_name=self.name,
            square_id=square_id,
            y_true=test_values.astype("float32"),
            y_pred=preds.astype("float32"),
            test_index=test.index,
            train_time_s=float(t_fit["seconds"]),
            predict_time_s=float(t_pred["seconds"]),
            extra={
                "order": self.order,
                "seasonal_diff_D": D,
                "seasonal_period_s": s,
                "aic": float(getattr(self.result_, "aic", np.nan)),
                "bic": float(getattr(self.result_, "bic", np.nan)),
                "n_train_fit": len(y_train),
            },
        )

    def summary(self) -> str:  # pragma: no cover - convenience
        if self.result_ is None:
            return "SARIMA model not yet fitted."
        return str(self.result_.summary())
