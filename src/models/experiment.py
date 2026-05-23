"""Hyper-parameter experimentation (Task 3 -- Experimentation & Tuning).

The assignment requires a *systematic*, documented tuning process. This module
runs a small grid search for the neural models and a guided order comparison
for SARIMA, logging every trial so the report can show the reasoning behind
each parameter choice.

For the neural sweep, each candidate is scored on a held-out **validation**
split (never the December test week), so model selection stays honest.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product

import pandas as pd

from ..config import CONFIG
from ..metrics import evaluate
from ..preprocessing import prepare_forecasting_data
from ..utils import timer
from .neural import NeuralForecaster
from .sarima_model import SarimaForecaster


# ---------------------------------------------------------------------------
# Neural grid search
# ---------------------------------------------------------------------------
def run_neural_experiments(
    series: pd.Series,
    cell: str = "LSTM",
    lookback_grid: list[int] | None = None,
    units_grid: list[int] | None = None,
    scaler_kind: str | None = None,
    save_name: str | None = None,
) -> pd.DataFrame:
    """Grid-search ``(lookback, units)`` for a recurrent model on one area.

    Each configuration is trained with early stopping and scored on the
    validation split. Returns a DataFrame sorted by validation RMSE (best
    first); the same table is saved to the experiments folder.
    """
    lookback_grid = lookback_grid or CONFIG.experiment["lookback_grid"]
    units_grid = units_grid or CONFIG.experiment["units_grid"]
    rows: list[dict] = []

    for lookback, units in product(lookback_grid, units_grid):
        print(f"\n>>> {cell} experiment: lookback={lookback}, units={units}")
        data = prepare_forecasting_data(
            series, lookback=lookback, scaler_kind=scaler_kind
        )
        # Score on the validation split: treat it as the 'test' set here.
        val_data = replace(
            data,
            X_test=data.X_val,
            y_test=data.y_val,
            y_test_actual=data.scaler.inverse_transform(data.y_val),
            test_index=pd.Index(range(len(data.y_val))),
        )
        model = NeuralForecaster(cell=cell, units=units, verbose=0)
        result = model.fit_predict(val_data)
        scores = evaluate(result.y_true, result.y_pred)
        rows.append(
            {
                "cell": cell,
                "lookback": lookback,
                "units": units,
                "val_MAE": scores["MAE"],
                "val_RMSE": scores["RMSE"],
                "val_MAPE": scores["MAPE"],
                "epochs_run": result.extra["epochs_run"],
                "n_params": result.extra["n_params"],
                "train_time_s": result.train_time_s,
            }
        )

    table = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
    save_name = save_name or f"experiments_{cell.lower()}"
    out = CONFIG.experiments_dir / f"{save_name}.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved experiment log -> {out}")
    return table


# ---------------------------------------------------------------------------
# SARIMA order comparison
# ---------------------------------------------------------------------------
def run_sarima_experiments(
    series: pd.Series,
    candidate_orders: list[tuple] | None = None,
    save_name: str = "experiments_sarima",
) -> pd.DataFrame:
    """Compare a few SARIMA non-seasonal (p, d, q) orders by AIC/BIC.

    The seasonal structure is fixed at D=1 differencing (s=144); candidates
    vary the non-seasonal ARIMA order, which is what the differencing
    formulation actually estimates. Orders are guided by the ACF/PACF of the
    seasonally-differenced series from Task 2 and ranked by AIC (lower better).
    """
    if candidate_orders is None:
        candidate_orders = [
            ((1, 0, 0), (0, 1, 0, 144)),
            ((1, 0, 1), (0, 1, 0, 144)),
            ((2, 0, 2), (0, 1, 0, 144)),
            ((3, 0, 1), (0, 1, 0, 144)),
        ]
    rows: list[dict] = []
    for order, seasonal_order in candidate_orders:
        print(f"\n>>> SARIMA experiment: order={order}, seasonal={seasonal_order}")
        model = SarimaForecaster(order=order, seasonal_order=seasonal_order)
        try:
            with timer("sarima candidate"):
                result = model.fit_predict(series)
            scores = evaluate(result.y_true, result.y_pred)
            rows.append(
                {
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "AIC": result.extra["aic"],
                    "BIC": result.extra["bic"],
                    "test_MAE": scores["MAE"],
                    "test_RMSE": scores["RMSE"],
                    "test_MAPE": scores["MAPE"],
                    "train_time_s": result.train_time_s,
                }
            )
        except Exception as exc:  # pragma: no cover - convergence failures
            print(f"    candidate failed: {exc}")
            rows.append(
                {"order": str(order), "seasonal_order": str(seasonal_order),
                 "AIC": float("nan"), "BIC": float("nan"), "test_MAE": float("nan"),
                 "test_RMSE": float("nan"), "test_MAPE": float("nan"),
                 "train_time_s": float("nan")}
            )

    table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    out = CONFIG.experiments_dir / f"{save_name}.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved experiment log -> {out}")
    return table
