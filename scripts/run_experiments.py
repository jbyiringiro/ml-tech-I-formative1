"""Hyper-parameter experimentation (Task 3 -- Experimentation & Tuning).

Runs the documented tuning experiments and writes their logs to
``experiments/``:

* a SARIMA non-seasonal-order comparison ranked by AIC/BIC and test error;
* a neural grid search over (lookback, units), scored on the validation split.

By default the experiments use the highest-traffic area as the tuning series.

    python scripts/run_experiments.py
    python scripts/run_experiments.py --cell GRU --quick

Prerequisite: run_task1.py must have been executed first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loading import (                                     # noqa: E402
    get_area_series,
    load_traffic_matrix,
    resolve_target_areas,
)
from src.models.experiment import (                                # noqa: E402
    run_neural_experiments,
    run_sarima_experiments,
)
from src.utils import save_table, set_global_seed                  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tuning experiments.")
    parser.add_argument("--cell", default="LSTM", choices=["LSTM", "GRU"],
                        help="recurrent cell for the neural grid search")
    parser.add_argument("--quick", action="store_true",
                        help="smaller grid for a fast check")
    args = parser.parse_args()

    set_global_seed()
    print("\n" + "#" * 70)
    print("# TASK 3 -- HYPER-PARAMETER EXPERIMENTS")
    print("#" * 70 + "\n")

    matrix = load_traffic_matrix()
    areas = resolve_target_areas(matrix)
    series = get_area_series(matrix, areas["highest"])
    print(f"Tuning on the highest-traffic area (Square id {areas['highest']}).\n")

    # 1) SARIMA order comparison ----------------------------------------
    print("--- SARIMA order comparison ---")
    sarima_exp = run_sarima_experiments(series)
    print(sarima_exp.to_string(), "\n")
    save_table(sarima_exp.set_index(["order", "seasonal_order"]),
               "task3_experiments_sarima")

    # 2) Neural grid search ---------------------------------------------
    print(f"--- {args.cell} grid search (lookback x units) ---")
    lookback_grid = [72, 144] if args.quick else None
    units_grid = [32, 64] if args.quick else None
    neural_exp = run_neural_experiments(
        series, cell=args.cell,
        lookback_grid=lookback_grid, units_grid=units_grid,
    )
    print(neural_exp.to_string(), "\n")
    save_table(neural_exp.set_index(["cell", "lookback", "units"]),
               f"task3_experiments_{args.cell.lower()}")

    print("Experiments complete. Logs in experiments/ and results/tables/.")


if __name__ == "__main__":
    main()
