"""Shared helpers: reproducibility, timing, I/O of figures and tables, and a
machine-readable description of the hardware/software setup (needed for the
Task 1 and Task 3 reporting requirements).
"""
from __future__ import annotations

import ctypes
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .config import CONFIG


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy and (if importable) TensorFlow RNGs.

    Returns the seed actually used so it can be logged in the report.
    """
    if seed is None:
        seed = int(CONFIG.get("random_seed", 42))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # TensorFlow is heavy; only seed it if already/importable.
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:  # pragma: no cover - tf optional at import time
        pass
    return seed


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
@contextmanager
def timer(label: str = "block") -> Iterator[dict]:
    """Context manager that measures wall-clock time of a code block.

    Usage
    -----
    >>> with timer("training") as t:
    ...     train_model()
    >>> t["seconds"]
    """
    record: dict = {"label": label, "seconds": None}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["seconds"] = time.perf_counter() - start
        print(f"[timer] {label}: {record['seconds']:.3f} s")


# ---------------------------------------------------------------------------
# Human-readable sizes
# ---------------------------------------------------------------------------
def human_bytes(n_bytes: float) -> str:
    """Format a byte count as a human-readable string (e.g. '1.42 GB')."""
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


# ---------------------------------------------------------------------------
# Figure / table persistence
# ---------------------------------------------------------------------------
def savefig(fig, name: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure into results/figures as PNG."""
    out = CONFIG.figures / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[figure] saved -> {out}")
    return out


def save_table(df: pd.DataFrame, name: str, float_fmt: str = "%.4f") -> Path:
    """Save a DataFrame to results/tables as both CSV and Markdown.

    The Markdown copy can be pasted straight into the report.
    """
    base = CONFIG.tables / name
    csv_path = base.with_suffix(".csv")
    md_path = base.with_suffix(".md")
    df.to_csv(csv_path, index=True, float_format=float_fmt)
    try:
        df.to_markdown(md_path, floatfmt=float_fmt.replace("%", ""))
    except Exception:  # pragma: no cover - tabulate may be missing
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(df.to_string())
    print(f"[table] saved -> {csv_path} and {md_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Hardware / software description (Task 1.V and Task 3.IV)
# ---------------------------------------------------------------------------
def _total_ram_bytes() -> int | None:
    """Best-effort total physical RAM in bytes (no hard dependency on psutil)."""
    try:  # preferred: psutil if available
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    if platform.system() == "Windows":  # ctypes fallback on Windows
        try:

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        except Exception:  # pragma: no cover
            return None
    try:  # POSIX fallback
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:  # pragma: no cover
        return None


def system_report() -> dict:
    """Return a dictionary describing the hardware/software environment.

    Used to populate the 'hardware and software setup' paragraphs the
    assignment requires in Task 1 and Task 3.
    """
    ram = _total_ram_bytes()
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_logical_cores": os.cpu_count(),
        "total_ram": human_bytes(ram) if ram else "unknown",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import tensorflow as tf

        info["tensorflow"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu"] = ", ".join(g.name for g in gpus) if gpus else "none (CPU only)"
    except Exception:  # pragma: no cover
        info["tensorflow"] = "not available"
        info["gpu"] = "unknown"
    return info


def print_system_report() -> dict:
    """Pretty-print :func:`system_report` and return it."""
    info = system_report()
    width = max(len(k) for k in info)
    print("=" * 60)
    print("HARDWARE / SOFTWARE ENVIRONMENT")
    print("=" * 60)
    for key, value in info.items():
        print(f"  {key.replace('_', ' '):<{width}} : {value}")
    print("=" * 60)
    return info
