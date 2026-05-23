"""Task 2 -- Exploratory Data Analysis (end-to-end run).

Loads the consolidated traffic matrix produced by Task 1 and generates every
required EDA figure and table. Run from the project root:

    python scripts/run_task2.py

Prerequisite: run_task1.py must have been executed first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loading import (                                     # noqa: E402
    compute_area_totals,
    get_area_series,
    load_traffic_matrix,
    resolve_target_areas,
)
from src.eda import (                                              # noqa: E402
    adf_test,
    decompose_series,
    detect_anomalies,
    plot_acf_pacf,
    plot_area_timeseries,
    plot_spatial_heatmap,
    plot_stationarity,
    plot_traffic_pdf,
    traffic_distribution_summary,
)
from src.utils import save_table, savefig                          # noqa: E402


def main() -> None:
    print("\n" + "#" * 70)
    print("# TASK 2 -- EXPLORATORY DATA ANALYSIS")
    print("#" * 70 + "\n")

    matrix = load_traffic_matrix()
    totals = compute_area_totals(matrix, save=False)
    areas = resolve_target_areas(matrix)
    print(f"Traffic matrix: {matrix.shape[0]} steps x {matrix.shape[1]} areas")
    print(f"Target areas: {areas}\n")

    # -- 2.I  Probability density function of per-area traffic ---------
    print("[2.I] Traffic PDF across all areas")
    savefig(plot_traffic_pdf(totals), "task2_01_traffic_pdf")
    save_table(traffic_distribution_summary(totals), "task2_traffic_distribution")

    # -- 2.II  Time series of the three target areas ------------------
    print("[2.II] Time series of three target areas (first two weeks)")
    savefig(plot_area_timeseries(matrix, areas), "task2_02_area_timeseries")

    # The highest-traffic area is the reference series for 2.III-2.V/2.VII.
    ref_id = areas["highest"]
    ref_series = get_area_series(matrix, ref_id)
    print(f"Reference series for detailed analysis: area {ref_id}\n")

    # -- 2.III  Stationarity: rolling statistics + ADF ----------------
    print("[2.III] Stationarity analysis (rolling stats + ADF)")
    savefig(plot_stationarity(ref_series), "task2_03_stationarity")
    adf_rows = [adf_test(get_area_series(matrix, sid), name=f"{label} (area {sid})")
                for label, sid in areas.items()]
    save_table(pd.DataFrame(adf_rows).set_index("series"), "task2_adf_results")

    # -- 2.IV  Seasonal decomposition ---------------------------------
    print("[2.IV] STL decomposition")
    fig_dec, _ = decompose_series(ref_series)
    savefig(fig_dec, "task2_04_decomposition")

    # -- 2.V  ACF / PACF ----------------------------------------------
    print("[2.V] ACF / PACF")
    savefig(plot_acf_pacf(ref_series), "task2_05_acf_pacf")

    # -- 2.VI  Spatial heatmap ----------------------------------------
    print("[2.VI] Spatial heatmap")
    savefig(plot_spatial_heatmap(totals), "task2_06_spatial_heatmap")

    # -- 2.VII  Anomaly detection -------------------------------------
    print("[2.VII] Anomaly detection")
    fig_anom, anomalies = detect_anomalies(ref_series)
    savefig(fig_anom, "task2_07_anomalies")
    if not anomalies.empty:
        save_table(anomalies.head(50), "task2_anomalies")
    print(f"  {len(anomalies)} anomalous points flagged in area {ref_id}")

    print("\nTask 2 complete. Figures in results/figures/, tables in results/tables/.")


if __name__ == "__main__":
    main()
