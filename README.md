# Comparative Time Series Analysis and Forecasting of Mobile Network Traffic

Machine Learning Techniques I — Formative Assignment 1.

This project analyses and forecasts mobile Internet traffic in the city of
Milan using the Telecom Italia (TIM) "Big Data Challenge" dataset. It covers
three tasks: (1) memory-efficient handling of a ~19 GB dataset, (2) exploratory
data analysis, and (3) one-step-ahead forecasting with three models — a
classical statistical model (**SARIMA**) and two recurrent neural networks
(**LSTM**, **GRU**).

## 1. Repository structure

```
formative_1/
├── config.yaml             # all paths & hyper-parameters (single source of truth)
├── requirements.txt
├── README.md
├── data/
│   ├── raw/telecom/        # <- put the downloaded daily .txt files here
│   ├── raw/grid/           # <- put milano-grid.geojson here
│   └── processed/          # Parquet outputs of Task 1 (generated)
├── src/                    # reusable package
│   ├── config.py           # config loader
│   ├── data_loading.py     # Task 1: streaming load + memory optimisation
│   ├── preprocessing.py    # Task 3: windowing, scaling, train/val/test split
│   ├── eda.py              # Task 2: all EDA routines
│   ├── metrics.py          # MAE / MAPE / RMSE
│   ├── viz.py              # Task 3 result plots
│   ├── utils.py            # timing, I/O, hardware report
│   └── models/             # SARIMA, LSTM/GRU, experiment runner
├── scripts/
│   ├── make_synthetic_data.py   # generate a runnable sample (no download needed)
│   ├── run_task1.py / run_task2.py / run_task3.py
├── notebooks/              # one notebook per task (import `src`)
├── results/figures|tables/ # generated figures and tables
├── experiments/            # hyper-parameter experiment logs
└── report/                 # final PDF report
```

## 2. Setup

Requires **Python 3.11+** (developed and tested on 3.13.9, Windows 11).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Getting the data

Two datasets are needed, both from the Harvard Dataverse, both for **Milan**.

### 3.1 Telecommunications activity (the main dataset)

DOI **10.7910/DVN/EGZHFV** —
[https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)

Download the **November + December 2013** daily files
(`sms-call-internet-mi-2013-11-*.txt` and `...-2013-12-*.txt`, 61 files) into
`data/raw/telecom/`.

> **Size warning:** each daily file is ~300–360 MB; the two months total
> **~19 GB** (the assignment's "≈5 GB" estimate is low). Ensure ~20 GB of free
> disk space. Harvard Dataverse caps a single ZIP download at ~2.5 GB, so
> download the files in **batches of ~7**, or per file.

Each file is tab-separated with 8 columns and no header. Field order (with the
assignment's documented correction — country code is the 3rd field):

| idx  | field                     | used?         |
| ---- | ------------------------- | ------------- |
| 0    | Square id                 | ✅            |
| 1    | Time interval (Unix ms)   | ✅            |
| 2    | Country code              | (summed over) |
| 3–6 | SMS-in/out, Call-in/out   | ❌ ignored    |
| 7    | Internet traffic activity | ✅            |

### 3.2 Milano grid

DOI **10.7910/DVN/QJWLFU** —
[https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU)

Download `milano-grid.geojson` into `data/raw/grid/`.

### 3.3 No download? Use the synthetic sample

The pipeline can run end-to-end without the real data:

```bash
python scripts/make_synthetic_data.py --grid-size 100   # structurally identical sample
```

This writes synthetic daily files (same 8-column format) into
`data/raw/telecom/`. Use `--grid-size 20` for a fast, lightweight sample.
**Delete the synthetic files before processing the real data** to avoid mixing.

## 4. Running the project

Run **from the project root**, in order (Task 1 must run first):

```bash
python scripts/run_task1.py        # load, optimise, build data/processed/traffic_matrix.parquet
python scripts/run_task2.py        # EDA: 7 figures + tables in results/
python scripts/run_task3.py        # forecasting: 9 prediction plots + metric/timing tables
python scripts/run_experiments.py  # hyper-parameter tuning logs (SARIMA orders + neural grid)
```

Useful flags:

```bash
python scripts/run_task1.py --max-files 5     # quick check on a subset
python scripts/run_task3.py --quick           # few epochs + short SARIMA fit (smoke run)
python scripts/run_task3.py --experiments     # also run the hyper-parameter grid
python scripts/run_task3.py --from-sample     # reproduce Task 3 from the small committed
                                              #   sample -- no 20 GB download needed
```

**Reproducing Task 3 without the full download.** The repository ships
`data/sample/target_area_series.csv` — the three target-area series extracted
from the real data (<0.5 MB). `python scripts/run_task3.py --from-sample`
reproduces the Task 3 results from it (SARIMA is deterministic; the neural
metrics may vary by ~1 % due to CPU non-determinism in TensorFlow). Tasks 1–2
still need the full dataset or the synthetic sample.

Or work interactively with the notebooks (`notebooks/01..03`), which import the
same `src` package.

All outputs land in `results/figures/`, `results/tables/` and `experiments/`.

## 5. Reproducibility notes

- Every path and hyper-parameter is in `config.yaml`.
- A fixed random seed (`config.yaml: random_seed`) seeds Python, NumPy and
  TensorFlow.
- The 16–22 December test week is **never** used for training or for scaler
  calibration.
- Hardware/software details are printed by every run (`src.utils.system_report`).

## 6. Links

- **GitHub repository:** [https://github.com/jbyiringiro/ml-tech-I-formative1](https://github.com/jbyiringiro/ml-tech-I-formative1)
- **Video demonstration:** <https://drive.google.com/file/d/1dRtSmwCBPSEulHKAT_zqKBbVDY789Vu-/view?usp=sharing>

## 7. References

See the PDF report in `report/` (IEEE style). Primary dataset reference:
G. Barlacchi *et al.*, "A multi-source dataset of urban life in the city of
Milan and the Province of Trentino," *Scientific Data* 2, 150055 (2015).
