"""Generate the three task notebooks programmatically with nbformat.

Keeping the notebooks under a generator script avoids hand-edited JSON, makes
them trivially reproducible, and keeps them in sync with the ``src`` API.

    python tools/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB_DIR.mkdir(exist_ok=True)

KERNEL_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

# Path-bootstrap cell shared by every notebook.
SETUP = """\
# --- Project setup: make the `src` package importable from anywhere ---
import sys
from pathlib import Path

_root = Path.cwd()
while not (_root / "src").exists() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))
print(f"Project root: {_root}")

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
"""


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build(cells: list, name: str) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = KERNEL_META
    path = NB_DIR / name
    nbf.write(nb, path)
    print(f"wrote {path}")


# ===========================================================================
# Notebook 1 -- Task 1: Data Handling & Memory Management
# ===========================================================================
def notebook_task1() -> None:
    cells = [
        md("# Task 1 -- Data Handling & Memory Management\n\n"
           "This notebook demonstrates how the ~19 GB Telecom Italia Milan "
           "dataset is loaded and processed under tight memory constraints. "
           "The heavy lifting lives in `src/data_loading.py`; here we run it "
           "step by step and inspect the evidence.\n\n"
           "**Prerequisite:** the raw daily files must be in `data/raw/telecom/` "
           "(see `README.md`), or run `python scripts/make_synthetic_data.py` "
           "for a runnable sample."),
        code(SETUP),
        md("## 1.V -- Hardware & software environment\n"
           "The environment determines the memory budget and therefore the "
           "data-handling strategy."),
        code("from src.utils import print_system_report\n"
             "_ = print_system_report()"),
        md("## 1.I -- Discover the raw daily files\n"
           "The dataset is delivered as one tab-separated file per day. We "
           "stream them one at a time so the full volume never has to fit in "
           "RAM."),
        code("from src.data_loading import discover_raw_files\n"
             "from src.utils import human_bytes\n\n"
             "files = discover_raw_files()\n"
             "total = sum(f.stat().st_size for f in files)\n"
             "print(f'{len(files)} daily files | total raw size on disk: "
             "{human_bytes(total)}')\n"
             "files[:3]"),
        md("## 1.III -- Memory optimisation: before vs after\n\n"
           "We compare a **naive** load (all 8 columns, default int64/float64) "
           "against the **optimised** load (only the 3 needed columns, "
           "downcast to uint16/float32, country rows aggregated). The table "
           "below is the quantitative evidence required by Task 1.III."),
        code("from src.data_loading import memory_comparison\n"
             "mem = memory_comparison(files[0])\n"
             "mem"),
        md("**Optimisation techniques applied** (Task 1.I / 1.II):\n"
           "1. **Column pruning** -- `usecols` reads only Square id, Time and "
           "Internet traffic, skipping SMS/Call/country-code columns.\n"
           "2. **Dtype downcasting** -- `square_id` -> uint16, `internet` -> "
           "float32 (half the width of the int64/float64 defaults).\n"
           "3. **Early aggregation** -- per-country rows are summed on load, "
           "collapsing each file to one value per (square, 10-min interval).\n"
           "4. **Columnar Parquet storage** -- the consolidated matrix is "
           "written as compressed Parquet, far smaller and faster than TSV."),
        md("## 1.I/1.II -- Build the consolidated traffic matrix\n"
           "Each daily file is streamed, optimised and pivoted to wide form "
           "(timestamp x square_id); the days are concatenated into a single "
           "float32 matrix and saved to Parquet. On the real data this is the "
           "long-running step."),
        code("from src.data_loading import (build_traffic_matrix, "
             "load_traffic_matrix,\n"
             "                              compute_area_totals, "
             "resolve_target_areas)\n\n"
             "try:\n"
             "    matrix = load_traffic_matrix()   # reuse if already built\n"
             "    print('Loaded cached traffic_matrix.parquet')\n"
             "except FileNotFoundError:\n"
             "    matrix = build_traffic_matrix(files, save=True)\n\n"
             "print(matrix.shape)\n"
             "matrix.iloc[:5, :5]"),
        code("totals = compute_area_totals(matrix)\n"
             "areas = resolve_target_areas(matrix)\n"
             "print('Three target areas for Tasks 2 & 3:', areas)"),
        md("## Summary\n"
           "The streaming + downcasting + aggregation strategy turns a ~19 GB "
           "on-disk dataset into a compact in-memory matrix (~0.35 GB for the "
           "full grid). The Parquet file is the single input for Tasks 2 and 3."),
    ]
    build(cells, "01_task1_data_handling.ipynb")


# ===========================================================================
# Notebook 2 -- Task 2: Exploratory Data Analysis
# ===========================================================================
def notebook_task2() -> None:
    cells = [
        md("# Task 2 -- Exploratory Data Analysis & Data Characterisation\n\n"
           "Temporal, spatial and statistical exploration of the traffic "
           "matrix. Each figure is followed by an interpretation cell -- "
           "**replace the bracketed prompts with observations from the real "
           "data** before submission."),
        code(SETUP),
        code("from src.data_loading import (load_traffic_matrix, "
             "compute_area_totals,\n"
             "                              resolve_target_areas, "
             "get_area_series)\n"
             "matrix = load_traffic_matrix()\n"
             "totals = compute_area_totals(matrix, save=False)\n"
             "areas = resolve_target_areas(matrix)\n"
             "print('matrix:', matrix.shape, '| areas:', areas)"),
        md("## 2.I -- Probability density function of per-area traffic"),
        code("from src.eda import plot_traffic_pdf, traffic_distribution_summary\n"
             "display(traffic_distribution_summary(totals))\n"
             "plot_traffic_pdf(totals)"),
        md("**Interpretation.** The distribution is strongly right-skewed / "
           "approximately log-normal: most areas carry modest traffic while a "
           "small number of central business districts dominate. [Comment on "
           "the skewness value and the p95/median ratio, and link the heavy "
           "tail to urban population/employment density.]"),
        md("## 2.II -- Time series of the three target areas (first two weeks)"),
        code("from src.eda import plot_area_timeseries\n"
             "plot_area_timeseries(matrix, areas)"),
        md("**Interpretation.** All three areas show a clear daily cycle and a "
           "weekday/weekend contrast. [Compare amplitudes and peak timing of "
           "the highest-traffic area vs areas 4159 and 4556; speculate on "
           "land use -- residential vs commercial vs nightlife.]"),
        md("## 2.III -- Stationarity: rolling statistics + ADF test"),
        code("from src.eda import plot_stationarity, adf_test\n"
             "ref_id = areas['highest']\n"
             "ref = get_area_series(matrix, ref_id)\n"
             "adf = pd.DataFrame([adf_test(get_area_series(matrix, sid),\n"
             "                             name=f'{lab} (area {sid})')\n"
             "                    for lab, sid in areas.items()]).set_index('series')\n"
             "display(adf)\n"
             "plot_stationarity(ref)"),
        md("**Interpretation.** The ADF p-values are very small, so the unit-"
           "root null is rejected -- the series are (weakly) stationary in the "
           "mean. The rolling mean/std stay bounded but oscillate with the "
           "daily cycle. [State the ADF statistic vs critical values and the "
           "implication for differencing in SARIMA.]"),
        md("## 2.IV -- Seasonal decomposition (trend / seasonal / residual)"),
        code("from src.eda import decompose_series\n"
             "fig, dec = decompose_series(ref)\n"
             "fig"),
        md("**Interpretation.** The seasonal component captures the strong "
           "24-hour cycle; the trend is comparatively flat; residuals are "
           "small except around anomalies. [Comment on any weekly pattern and "
           "on residual heteroscedasticity.]"),
        md("## 2.V -- Autocorrelation & partial autocorrelation"),
        code("from src.eda import plot_acf_pacf\n"
             "plot_acf_pacf(ref)"),
        md("**Interpretation.** The ACF decays slowly and shows pronounced "
           "spikes at multiples of 144 lags (one day), confirming daily "
           "seasonality. The PACF cuts off after a few lags, indicating a "
           "low-order autoregressive structure. [Use this to justify the "
           "SARIMA orders.]"),
        md("## 2.VI -- Spatial analysis (heatmap)"),
        code("from src.eda import plot_spatial_heatmap\n"
             "plot_spatial_heatmap(totals)"),
        md("**Interpretation.** Traffic concentrates in the city centre and a "
           "few secondary hotspots, decaying towards the periphery. [Link the "
           "hotspot(s) to known Milan landmarks -- centre, stadium, airport.]"),
        md("## 2.VII -- Anomalies & unusual behaviour"),
        code("from src.eda import detect_anomalies\n"
             "fig, anomalies = detect_anomalies(ref)\n"
             "display(anomalies.head(10))\n"
             "fig"),
        md("**Interpretation.** Flagged points are mostly sharp spikes (mass "
           "events, public holidays) or drops (possible outages). [Discuss the "
           "specific dates flagged -- e.g. Dec public holidays -- and how they "
           "may challenge the forecasting models in Task 3.]"),
    ]
    build(cells, "02_task2_eda.ipynb")


# ===========================================================================
# Notebook 3 -- Task 3: Forecasting Models
# ===========================================================================
def notebook_task3() -> None:
    cells = [
        md("# Task 3 -- Forecasting Models (SARIMA / LSTM / GRU)\n\n"
           "One-step-ahead forecasting of the 16-22 December test week for the "
           "three target areas. The test week is held out from all training "
           "and from scaler calibration."),
        code(SETUP),
        code("from src.data_loading import (load_traffic_matrix, "
             "resolve_target_areas,\n"
             "                              get_area_series)\n"
             "from src.preprocessing import prepare_forecasting_data\n"
             "from src.models import SarimaForecaster, NeuralForecaster\n"
             "from src.metrics import evaluate, metrics_table\n"
             "from src.utils import set_global_seed\n"
             "set_global_seed()\n"
             "matrix = load_traffic_matrix()\n"
             "areas = resolve_target_areas(matrix)\n"
             "areas"),
        md("## 3.I/3.VI -- Model & input representation\n"
           "* **SARIMA** -- classical statistical model; consumes the raw "
           "series, fitted once then rolled one-step-ahead with Kalman "
           "updates.\n"
           "* **LSTM / GRU** -- recurrent networks; input is a window of the "
           "`lookback` most recent scaled values, shape `(lookback, 1)`; the "
           "scaler is fitted on the training span only.\n\n"
           "We first demonstrate on the highest-traffic area, then loop over "
           "all three."),
        code("ref_id = areas['highest']\n"
             "series = get_area_series(matrix, ref_id)\n"
             "data = prepare_forecasting_data(series)\n"
             "print('input shape (lookback, features):', data.input_shape)\n"
             "print('train/val/test windows:', data.meta)"),
        md("### Train the three models on the reference area"),
        code("sarima = SarimaForecaster()\n"
             "res_sarima = sarima.fit_predict(series)\n"
             "res_sarima"),
        code("lstm = NeuralForecaster(cell='LSTM', verbose=0)\n"
             "res_lstm = lstm.fit_predict(data)\n"
             "print(lstm.summary())\n"
             "res_lstm"),
        code("gru = NeuralForecaster(cell='GRU', verbose=0)\n"
             "res_gru = gru.fit_predict(data)\n"
             "res_gru"),
        md("### Metrics and forecast plot for the reference area"),
        code("results = [res_sarima, res_lstm, res_gru]\n"
             "table = metrics_table([evaluate(r.y_true, r.y_pred, r.model_name)\n"
             "                       for r in results])\n"
             "display(table)\n"
             "from src.viz import plot_forecasts_for_area\n"
             "plot_forecasts_for_area('highest', ref_id, results)"),
        md("## 3.II/3.III -- Full run: all models on all three areas\n"
           "Produces the nine prediction plots and three metric tables. For "
           "speed in the notebook set `verbose=0`; the full batch run is "
           "`python scripts/run_task3.py`."),
        code("from src.viz import plot_error_over_time\n"
             "all_results = {}\n"
             "for label, sid in areas.items():\n"
             "    s = get_area_series(matrix, sid)\n"
             "    d = prepare_forecasting_data(s)\n"
             "    rs = [SarimaForecaster().fit_predict(s),\n"
             "          NeuralForecaster('LSTM', verbose=0).fit_predict(d),\n"
             "          NeuralForecaster('GRU', verbose=0).fit_predict(d)]\n"
             "    all_results[label] = rs\n"
             "    tbl = metrics_table([evaluate(r.y_true, r.y_pred, r.model_name)\n"
             "                         for r in rs])\n"
             "    print(f'\\n=== {label} (area {sid}) ===')\n"
             "    display(tbl)\n"
             "    display(plot_forecasts_for_area(label, sid, rs))"),
        md("## 3.IV -- Training & execution time"),
        code("timing = pd.DataFrame([\n"
             "    {'area': lab, 'model': r.model_name,\n"
             "     'train_time_s': r.train_time_s,\n"
             "     'predict_time_s': r.predict_time_s}\n"
             "    for lab, rs in all_results.items() for r in rs])\n"
             "timing.groupby('model')[['train_time_s', 'predict_time_s']].mean()"),
        md("## 3.VIII -- Failure analysis\n"
           "Plot the absolute error across the test week to locate the period "
           "where the models perform worst."),
        code("plot_error_over_time('highest', all_results['highest'])"),
        md("**Failure analysis.** [Identify the worst period -- e.g. an "
           "anomalous spike or a public holiday in the test week -- and "
           "explain it with reference to the Task 2 anomaly findings.]"),
        md("## 3.V/3.VII -- Discussion\n"
           "[Compare predictive accuracy (MAE/RMSE/MAPE), training time and "
           "suitability. Justify the best model using both the metrics and the "
           "data characteristics from Task 2. Note margins for improvement: "
           "calendar features, multi-area input, attention, etc.]"),
        md("### Optional -- hyper-parameter experiments"),
        code("# from src.models.experiment import run_neural_experiments\n"
             "# run_neural_experiments(series, cell='LSTM')   # grid search"),
    ]
    build(cells, "03_task3_forecasting.ipynb")


if __name__ == "__main__":
    notebook_task1()
    notebook_task2()
    notebook_task3()
    print("All notebooks generated.")
