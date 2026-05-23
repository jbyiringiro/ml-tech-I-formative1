"""Task 3 -- Forecasting Models (end-to-end run).

Trains and evaluates the three models (SARIMA, LSTM, GRU) independently on each
of the three target areas, for one-step-ahead prediction of the 16-22 December
test week. Produces the nine prediction plots, three per-area metric tables,
the training/execution-time table and the diagnostic figures.

    python scripts/run_task3.py                 # full run
    python scripts/run_task3.py --quick         # fast smoke run (few epochs)
    python scripts/run_task3.py --experiments   # also run the tuning grid

Prerequisite: run_task1.py must have been executed first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG                                      # noqa: E402
from src.data_loading import (                                     # noqa: E402
    get_area_series,
    load_traffic_matrix,
    resolve_target_areas,
)
from src.metrics import evaluate, metrics_table                    # noqa: E402
from src.models import NeuralForecaster, SarimaForecaster          # noqa: E402
from src.preprocessing import prepare_forecasting_data             # noqa: E402
from src.utils import save_table, savefig, set_global_seed         # noqa: E402
from src.viz import (                                              # noqa: E402
    plot_error_over_time,
    plot_forecasts_for_area,
    plot_training_curves,
)


def forecast_area(label: str, square_id: int, series, quick: bool) -> list:
    """Train and evaluate all three models on one area; return ForecastResults."""
    print(f"\n{'=' * 70}\nAREA: {label}  (Square id {square_id})\n{'=' * 70}")
    results = []

    # --- Model 1: SARIMA (classical statistical) -------------------------
    sarima_kwargs = dict(train_tail_days=7, maxiter=15) if quick else {}
    sarima = SarimaForecaster(**sarima_kwargs)
    results.append(sarima.fit_predict(series))

    # --- Shared preprocessing for the neural models ----------------------
    data = prepare_forecasting_data(series)
    epochs = 3 if quick else None

    # --- Model 2: LSTM ---------------------------------------------------
    lstm = NeuralForecaster(cell="LSTM", epochs=epochs, verbose=2)
    results.append(lstm.fit_predict(data))

    # --- Model 3: GRU ----------------------------------------------------
    gru = NeuralForecaster(cell="GRU", epochs=epochs, verbose=2)
    results.append(gru.fit_predict(data))

    return results


def _square_id(series) -> int:
    """Extract the integer square id from a series named 'area_<id>'."""
    return int(str(series.name).split("_")[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 3.")
    parser.add_argument("--quick", action="store_true",
                        help="fast run: few epochs and a short SARIMA fit")
    parser.add_argument("--experiments", action="store_true",
                        help="also run the hyper-parameter tuning grid")
    parser.add_argument("--from-sample", action="store_true",
                        help="use the committed 3-area sample instead of the "
                             "full traffic matrix (reproduces Task 3 without "
                             "the 20 GB download)")
    args = parser.parse_args()

    set_global_seed()
    print("\n" + "#" * 70)
    print("# TASK 3 -- FORECASTING MODELS (SARIMA / LSTM / GRU)")
    print("#" * 70)
    if args.quick:
        print("  [quick mode: reduced epochs and SARIMA fit]")

    # Source the three target-area series -- either from the full matrix
    # (default) or from the small committed reproducibility sample.
    if args.from_sample:
        from src.data_loading import load_sample_series
        series_map = load_sample_series()
        print("  [using data/sample/target_area_series.csv]")
    else:
        matrix = load_traffic_matrix()
        areas = resolve_target_areas(matrix)
        series_map = {lab: get_area_series(matrix, sid)
                      for lab, sid in areas.items()}
    print(f"\nTarget areas: "
          f"{{ {', '.join(f'{k}: {_square_id(v)}' for k, v in series_map.items())} }}")

    timing_rows: list[dict] = []
    summary_rows: list[dict] = []

    for label, series in series_map.items():
        square_id = _square_id(series)
        results = forecast_area(label, square_id, series, args.quick)

        # Per-area metric table (Task 3.III) ------------------------------
        scored = [evaluate(r.y_true, r.y_pred, r.model_name) for r in results]
        table = metrics_table(scored)
        print(f"\n--- Metrics for {label} (area {square_id}) ---")
        print(table.to_string())
        save_table(table, f"task3_metrics_{label}_area{square_id}")

        # Nine prediction plots: 3 areas x 3 model panels (Task 3.II) -----
        savefig(plot_forecasts_for_area(label, square_id, results),
                f"task3_forecast_{label}_area{square_id}")
        # Diagnostics
        curves = plot_training_curves(results)
        if curves is not None:
            savefig(curves, f"task3_training_curves_{label}_area{square_id}")
        savefig(plot_error_over_time(label, results),
                f"task3_error_{label}_area{square_id}")

        # Persist raw predictions for reproducibility ---------------------
        preds = pd.concat({r.model_name: r.to_frame() for r in results}, axis=1)
        preds.to_csv(CONFIG.tables / f"task3_predictions_{label}_area{square_id}.csv")

        # Accumulate timing + summary ------------------------------------
        for r, s in zip(results, scored):
            timing_rows.append({
                "area": label, "square_id": square_id, "model": r.model_name,
                "train_time_s": r.train_time_s, "predict_time_s": r.predict_time_s,
            })
            summary_rows.append({"area": label, "model": r.model_name, **{
                k: s[k] for k in ("MAE", "RMSE", "MAPE", "sMAPE")}})

    # Training / execution-time table (Task 3.IV) -------------------------
    timing_df = pd.DataFrame(timing_rows)
    timing_summary = (timing_df.groupby("model")[["train_time_s", "predict_time_s"]]
                      .mean().round(3))
    timing_summary["measured_over"] = "mean of 3 areas"
    print("\n--- Training / execution time (mean over 3 areas) ---")
    print(timing_summary.to_string())
    save_table(timing_df, "task3_timing_detailed")
    save_table(timing_summary, "task3_timing_summary")

    # Cross-area summary --------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    save_table(summary_df.set_index(["area", "model"]), "task3_summary_all")
    print("\n--- Mean metrics per model (across areas) ---")
    print(summary_df.groupby("model")[["MAE", "RMSE", "MAPE"]].mean().round(3))

    # Optional: hyper-parameter experiments ------------------------------
    if args.experiments:
        from src.models.experiment import run_neural_experiments
        ref_series = series_map["highest"]
        print("\n--- Hyper-parameter experiments (LSTM grid on highest area) ---")
        exp = run_neural_experiments(ref_series, cell="LSTM")
        print(exp.to_string())
        save_table(exp.set_index(["cell", "lookback", "units"]),
                   "task3_experiments_lstm")

    print("\nTask 3 complete. Figures in results/figures/, tables in results/tables/.")


if __name__ == "__main__":
    main()
