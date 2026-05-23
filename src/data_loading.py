"""Task 1: load and process the 19 GB Milan dataset without running out of RAM.

Trick is to stream the daily files one at a time, only read the 3 columns we
need (square_id, time, internet), downcast the dtypes, and aggregate country
codes immediately. The wide matrix is then saved as Parquet for fast reloads
in Tasks 2 and 3.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import CONFIG, PROJECT_ROOT
from .utils import human_bytes, timer

# Names we assign to the 8 raw columns (header-less files). Field order follows
# the assignment's correction: country code is the 3rd field.
RAW_COLUMN_NAMES = [
    "square_id",
    "time_ms",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet",
]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def discover_raw_files(raw_dir: Path | None = None) -> list[Path]:
    """Find all the daily .txt files inside data/raw/telecom (sorted)."""
    raw_dir = Path(raw_dir) if raw_dir else CONFIG.raw_telecom
    files = sorted(p for p in raw_dir.rglob("*.txt") if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"No .txt telecom files found under {raw_dir}.\n"
            "Download the Milan telecommunications dataset (Harvard Dataverse "
            "doi:10.7910/DVN/EGZHFV) and place the daily files there. "
            "See README.md for instructions, or run "
            "`python scripts/make_synthetic_data.py` for a runnable sample."
        )
    return files


# ---------------------------------------------------------------------------
# Memory measurement helpers (Task 1.III)
# ---------------------------------------------------------------------------
def dataframe_memory(df: pd.DataFrame) -> int:
    """Deep memory footprint of a DataFrame in bytes."""
    return int(df.memory_usage(deep=True).sum())


def load_day_naive(path: Path) -> pd.DataFrame:
    """Load a daily file with pandas defaults -- used only to compare memory."""
    return pd.read_csv(path, sep="\t", header=None, names=RAW_COLUMN_NAMES)


def load_day_optimised(path: Path) -> pd.DataFrame:
    """Load one daily file the memory-friendly way: 3 columns, downcasted,
    country codes summed. Returns one row per (square, 10-min interval).
    """
    cfg = CONFIG.dataset
    usecols = [cfg["col_square_id"], cfg["col_time"], cfg["col_internet"]]
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=usecols,
        names=["square_id", "time_ms", "internet"],
        dtype={"square_id": "uint16", "time_ms": "int64", "internet": "float32"},
    )
    # Missing internet activity == no recorded traffic -> 0.
    df["internet"] = df["internet"].fillna(
        np.float32(CONFIG.loading["fillna_internet"])
    )
    # Collapse the per-country rows: total internet traffic per (square, time).
    df = (
        df.groupby(["square_id", "time_ms"], sort=False, observed=True)["internet"]
        .sum()
        .reset_index()
    )
    # Epoch-ms -> datetime, shifted from UTC to CET local time (see config).
    offset = pd.Timedelta(hours=CONFIG.dataset.get("timezone_offset_hours", 0))
    df["timestamp"] = pd.to_datetime(df["time_ms"], unit="ms") + offset
    df["internet"] = df["internet"].astype("float32")
    return df[["square_id", "timestamp", "internet"]]


def memory_comparison(sample_file: Path | None = None) -> pd.DataFrame:
    """Compare the naive vs optimised load on one file -- this is the
    before/after evidence used in the Task 1 section of the report.
    """
    sample_file = Path(sample_file) if sample_file else discover_raw_files()[0]
    print(f"Memory comparison on sample file: {sample_file.name}")

    with timer("naive load"):
        naive = load_day_naive(sample_file)
    naive_bytes = dataframe_memory(naive)

    with timer("optimised load"):
        optimised = load_day_optimised(sample_file)
    optimised_bytes = dataframe_memory(optimised)

    reduction = 100.0 * (1.0 - optimised_bytes / naive_bytes)
    report = pd.DataFrame(
        {
            "rows": [len(naive), len(optimised)],
            "columns": [naive.shape[1], optimised.shape[1]],
            "memory_bytes": [naive_bytes, optimised_bytes],
            "memory_human": [human_bytes(naive_bytes), human_bytes(optimised_bytes)],
            "pct_of_naive": [100.0, 100.0 * optimised_bytes / naive_bytes],
        },
        index=["naive (8 cols, int64/float64)", "optimised (3 cols, downcast+agg)"],
    )
    print(report.to_string())
    print(f"--> optimised load uses {reduction:.1f}% less memory than naive.")
    return report


# ---------------------------------------------------------------------------
# Full-dataset consolidation
# ---------------------------------------------------------------------------
def build_traffic_matrix(
    raw_files: Iterable[Path] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Loop over every daily file and stitch them into one wide matrix
    (rows = 10-min timestamps, columns = square_id, values = traffic).

    With 10,000 areas x 8,784 intervals as float32 this fits in ~350 MB of
    RAM even though the raw text is ~19 GB. Saves the matrix as Parquet.
    """
    files = list(raw_files) if raw_files is not None else discover_raw_files()
    print(f"Consolidating {len(files)} daily files into the traffic matrix...")

    daily_frames: list[pd.DataFrame] = []
    with timer("full consolidation"):
        for i, path in enumerate(files, 1):
            day = load_day_optimised(path)
            # Pivot this day to wide form (timestamp x square_id).
            wide = day.pivot_table(
                index="timestamp",
                columns="square_id",
                values="internet",
                aggfunc="sum",
            ).astype("float32")
            daily_frames.append(wide)
            print(f"  [{i:>3}/{len(files)}] {path.name}: {wide.shape[0]} intervals")

    matrix = pd.concat(daily_frames, axis=0).sort_index()
    # Two squares could be absent on a given day -> NaN; absence == no traffic.
    matrix = matrix.fillna(np.float32(0.0)).astype("float32")
    matrix = matrix[~matrix.index.duplicated(keep="first")]
    matrix.columns.name = "square_id"
    matrix.index.name = "timestamp"

    # Restrict to the official two-month study window.
    start = pd.Timestamp(CONFIG.dates["study_start"])
    end = pd.Timestamp(CONFIG.dates["study_end"]) + pd.Timedelta(days=1)
    matrix = matrix.loc[(matrix.index >= start) & (matrix.index < end)]

    print(
        f"Traffic matrix: {matrix.shape[0]} time steps x "
        f"{matrix.shape[1]} areas | in-memory size "
        f"{human_bytes(dataframe_memory(matrix))}"
    )

    if save:
        out = CONFIG.processed / "traffic_matrix.parquet"
        matrix.to_parquet(out, compression="snappy")
        print(
            f"Saved -> {out} (on-disk {human_bytes(out.stat().st_size)})"
        )
    return matrix


def load_traffic_matrix() -> pd.DataFrame:
    """Load the consolidated traffic matrix written by :func:`build_traffic_matrix`."""
    path = CONFIG.processed / "traffic_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/run_task1.py (or build_traffic_matrix) first."
        )
    matrix = pd.read_parquet(path)
    matrix.index = pd.to_datetime(matrix.index)
    return matrix


# ---------------------------------------------------------------------------
# Derived aggregates (used by Task 2 / Task 3)
# ---------------------------------------------------------------------------
def compute_area_totals(matrix: pd.DataFrame, save: bool = True) -> pd.Series:
    """Sum each area's traffic over the 2 months -- gives 10,000 values for
    the PDF plot in Task 2.
    """
    totals = matrix.sum(axis=0)
    totals.name = "total_traffic"
    totals.index.name = "square_id"
    if save:
        out = CONFIG.processed / "area_totals.parquet"
        totals.to_frame().to_parquet(out)
        print(f"Saved area totals -> {out}")
    return totals


def get_area_series(matrix: pd.DataFrame, square_id: int) -> pd.Series:
    """Extract the full traffic time series for a single area."""
    if square_id not in matrix.columns:
        raise KeyError(
            f"square_id {square_id} not present in the traffic matrix "
            f"(available range {matrix.columns.min()}..{matrix.columns.max()})."
        )
    series = matrix[square_id].astype("float32")
    series.name = f"area_{square_id}"
    return series


def load_sample_series() -> dict[str, pd.Series]:
    """Load the 3 area series from data/sample/ -- lets Task 3 run without
    needing the 20 GB raw download.
    """
    path = PROJECT_ROOT / "data" / "sample" / "target_area_series.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Either run scripts/run_task1.py to build the "
            "full matrix, or restore the committed sample file."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [str(c) for c in df.columns]
    totals = df.sum()
    highest_col = str(totals.idxmax())
    fixed_cols = [f"area_{x}" for x in CONFIG.areas["fixed"]]

    out: dict[str, pd.Series] = {}
    s = df[highest_col].astype("float32"); s.name = highest_col
    out["highest"] = s
    for i, col in enumerate([c for c in fixed_cols if c in df.columns], start=1):
        s = df[col].astype("float32"); s.name = col
        out[f"fixed_{i}"] = s
    return out


def resolve_target_areas(matrix: pd.DataFrame) -> dict[str, int]:
    """Pick the 3 areas the assignment asks for: the busiest one + 4159 + 4556.

    If 4159/4556 are missing (only happens on a small synthetic grid), grab
    sensible replacements so the rest of the pipeline still runs.
    """
    totals = matrix.sum(axis=0)
    highest = int(totals.idxmax())
    available = set(int(c) for c in matrix.columns)
    fixed = [int(x) for x in CONFIG.areas["fixed"]]

    # Candidate substitutes spread across the traffic distribution.
    ranked = totals.sort_values(ascending=False).index.tolist()
    n = len(ranked)
    candidates = [int(ranked[n // 3]), int(ranked[2 * n // 3])]

    resolved: dict[str, int] = {"highest": highest}
    for i, fid in enumerate(fixed, start=1):
        if fid in available:
            resolved[f"fixed_{i}"] = fid
        else:
            sub = next(c for c in candidates if c not in resolved.values())
            print(
                f"[warn] fixed area {fid} absent from this dataset "
                f"(reduced/synthetic grid) -- substituting area {sub}."
            )
            resolved[f"fixed_{i}"] = sub
    return resolved
