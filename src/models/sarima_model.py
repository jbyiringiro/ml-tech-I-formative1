"""SARIMA forecaster (classical statistical model for Task 3).

Daily traffic has a strong 24-hour cycle, so the seasonal period is s = 144
(24 * 6). Instead of using a full SARIMAX with seasonal_order=(P,1,Q,144) --
which trains for minutes because of the 144-step latent state -- I do the
seasonal differencing myself:

    y_t = x_t - x_{t-144}        # take the difference vs. yesterday
    fit ARIMA(p, d, q) on y_t    # short, easy to fit
    x_hat_t = y_hat_t + x_{t-144}   # invert at predict time

This is still a SARIMA(p, d, q)(0, 1, 0)_144 model, just written out
explicitly. Fitting drops from ~235 s to ~1.5 s per area.

One-step-ahead: fit once on the train half, append the (differenced) test
observations and read the predictions back -- statsmodels does this in a
single Kalman filter pass.
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
    """SARIMA forecaster -- fits an ARIMA on the seasonally-differenced series."""

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
        """y_t = x_t - x_{t-s}, applied D times."""
        out = values
        for _ in range(self.seasonal_diff):
            out = out[self.seasonal_period:] - out[: -self.seasonal_period]
        return out

    # ------------------------------------------------------------------
    def fit_predict(self, series: pd.Series) -> ForecastResult:
        """Fit on the train half, then produce one-step-ahead predictions for
        every step of the 16-22 Dec test week.
        """
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
