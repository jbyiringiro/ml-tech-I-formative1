"""Task 1 -- Data Handling & Memory Management.

The raw Telecom Italia Milan dataset is delivered as ~61 daily tab-separated
text files (one per day, Nov-Dec 2013), each roughly 300-360 MB, ~19 GB in
total. Loading all of it into memory at once with default pandas dtypes is
infeasible on a typical laptop. This module implements the memory-efficient
strategy used in the project:

1. **Stream file-by-file.** Each daily file is processed independently and
   never more than one day is held in RAM at a time.
2. **Read only the 3 relevant columns** (Square id, Time interval, Internet
   traffic activity) via ``usecols`` -- the SMS/Call/country columns are
   skipped entirely, avoiding ~60% of the I/O and memory cost.
3. **Downcast dtypes on read.** ``Square id`` -> uint16, ``Internet`` ->
   float32, timestamp parsed from epoch-ms to datetime64. This roughly halves
   memory versus the int64/float64 defaults.
4. **Aggregate immediately.** Each file contains one row per
   (square, time, country); we sum Internet activity over country codes so the
   per-day result collapses to one value per (square, time interval).
5. **Persist as Parquet.** The consolidated traffic matrix is written to a
   compressed columnar Parquet file, which is far smaller and faster to reload
   than CSV/TSV.

The functions also expose *before/after* memory measurements so the report can
quantify the optimisation (Task 1.III).
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
    """Return a sorted list of raw daily ``.txt`` files under ``raw_dir``.

    Searches recursively so it works whether the user extracted the Dataverse
    archive flat or into sub-folders.
    """
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
    """Load one daily file with pandas *defaults* -- the un-optimised baseline.

    Reads every column with default dtypes (int64/float64/object). Used only to
    demonstrate the 'before optimisation' memory figure for the report.
    """
    return pd.read_csv(path, sep="\t", header=None, names=RAW_COLUMN_NAMES)


def load_day_optimised(path: Path) -> pd.DataFrame:
    """Load one daily file with the memory-optimised strategy.

    Reads only the 3 needed columns, downcasts dtypes, and aggregates Internet
    traffic over country codes. Returns columns ``[square_id, timestamp,
    internet]`` with one row per (square, 10-min interval).
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
    """Quantify the memory optimisation on a single sample daily file.

    Returns a small DataFrame (the 'before vs after' evidence for Task 1.III)
    comparing the naive full-column int64/float64 load against the optimised
    3-column downcast-and-aggregate load.
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
    """Stream every daily file and build the consolidated traffic matrix.

    The result is a *wide* matrix: rows indexed by 10-minute ``timestamp``,
    columns are ``square_id`` (0..N), values are total Internet traffic as
    float32. With the full 100x100 grid over two months this is roughly
    8,784 x 10,000 float32 == ~0.35 GB, which fits comfortably in RAM even
    though the raw text is ~19 GB on disk -- the streaming-and-aggregate design
    is what makes this possible.

    Parameters
    ----------
    raw_files : iterable of paths; defaults to :func:`discover_raw_files`.
    save : when True, also writes ``traffic_matrix.parquet`` to data/processed.
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
    """Total two-month traffic per area -- the 10,000 samples for the Task 2 PDF."""
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
    """Load the three target-area series from the committed reproducibility sample.

    ``data/sample/target_area_series.csv`` is a small (<0.5 MB) extract of the
    three target areas, kept under version control so that Task 3 can be
    reproduced *without* the 20 GB raw download. Returns a mapping
    ``{label: series}`` with the same labels as :func:`resolve_target_areas`.
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
    """Resolve the three target areas required by Task 2.II / Task 3.

    Returns a dict mapping a human label to a square id: ``highest`` (the area
    with the greatest two-month traffic) and the two fixed ids 4159 and 4556.

    On the full 10,000-area dataset the fixed ids always exist. When run on a
    reduced grid (e.g. the synthetic sample) a missing fixed id is replaced by
    a representative substitute drawn from a different traffic rank, with a
    clear warning -- so the pipeline still runs end-to-end for testing.
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
