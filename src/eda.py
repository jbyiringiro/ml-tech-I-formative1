"""Task 2 -- Exploratory Data Analysis & Data Characterisation.

Every routine returns a matplotlib ``Figure`` (and, where relevant, the
numerical results) so that notebooks and the ``run_task2`` script can both
display and persist them. The functions cover all seven required analyses:
traffic PDF, multi-area time series, stationarity (rolling stats + ADF),
seasonal decomposition, ACF/PACF, a spatial heatmap, and anomaly detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

from .config import CONFIG

sns.set_theme(style="whitegrid", context="notebook")

INTERVALS_PER_DAY = CONFIG.dataset["intervals_per_day"]   # 144


# ---------------------------------------------------------------------------
# Task 2.I -- Probability density function of total per-area traffic
# ---------------------------------------------------------------------------
def plot_traffic_pdf(area_totals: pd.Series):
    """Plot the PDF of total two-month traffic across all areas.

    Two panels: a linear-axis histogram+KDE and a log-axis version, because
    urban mobile traffic is typically heavy-tailed (approximately log-normal):
    a few central business areas dominate while most peripheral cells are low.
    """
    totals = area_totals.to_numpy(dtype="float64")
    totals = totals[np.isfinite(totals)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(totals, bins=80, kde=True, color="steelblue", ax=axes[0])
    axes[0].set_title("PDF of total two-month traffic per area (linear scale)")
    axes[0].set_xlabel("Total Internet traffic activity (2 months)")
    axes[0].set_ylabel("Number of areas")

    positive = totals[totals > 0]
    sns.histplot(positive, bins=80, kde=True, color="indianred",
                 log_scale=True, ax=axes[1])
    axes[1].set_title("PDF of total per-area traffic (log scale)")
    axes[1].set_xlabel("Total Internet traffic activity (log scale)")
    axes[1].set_ylabel("Number of areas")

    fig.tight_layout()
    plt.close(fig)        # prevent double-display in notebooks; savefig still works
    return fig


def traffic_distribution_summary(area_totals: pd.Series) -> pd.DataFrame:
    """Summary statistics of the per-area total-traffic distribution."""
    totals = area_totals.to_numpy(dtype="float64")
    totals = totals[np.isfinite(totals)]
    from scipy import stats

    summary = {
        "n_areas": totals.size,
        "mean": np.mean(totals),
        "median": np.median(totals),
        "std": np.std(totals),
        "min": np.min(totals),
        "max": np.max(totals),
        "skewness": float(stats.skew(totals)),
        "kurtosis": float(stats.kurtosis(totals)),
        "p95 / median ratio": np.percentile(totals, 95) / max(np.median(totals), 1e-9),
    }
    return pd.DataFrame.from_dict(summary, orient="index", columns=["value"])


# ---------------------------------------------------------------------------
# Task 2.II -- Time series of three areas over the first two weeks
# ---------------------------------------------------------------------------
def plot_area_timeseries(matrix: pd.DataFrame, areas: dict[str, int]):
    """Overlay the traffic time series of the three target areas.

    ``areas`` maps a label to a square id (see
    :func:`data_loading.resolve_target_areas`). The window is the first two
    weeks of the study period, as required by Task 2.II.
    """
    start = pd.Timestamp(CONFIG.dates["first_two_weeks_start"])
    end = pd.Timestamp(CONFIG.dates["first_two_weeks_end"]) + pd.Timedelta(days=1)
    window = matrix.loc[(matrix.index >= start) & (matrix.index < end)]

    fig, ax = plt.subplots(figsize=(14, 5))
    palette = {"highest": "crimson", "fixed_1": "royalblue", "fixed_2": "seagreen"}
    for label, sid in areas.items():
        if sid not in window.columns:
            continue
        ax.plot(window.index, window[sid], linewidth=0.9,
                color=palette.get(label, None),
                label=f"{label.replace('_', ' ')} (Square id {sid})")
    ax.set_title("Internet traffic -- first two weeks, three target areas")
    ax.set_xlabel("Date")
    ax.set_ylabel("Internet traffic activity (per 10 min)")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)        # prevent double-display in notebooks; savefig still works
    return fig


# ---------------------------------------------------------------------------
# Task 2.III -- Stationarity: rolling statistics + Augmented Dickey-Fuller
# ---------------------------------------------------------------------------
def adf_test(series: pd.Series, name: str = "series") -> dict:
    """Run the Augmented Dickey-Fuller test and return a structured result.

    Null hypothesis: the series has a unit root (is non-stationary). A p-value
    below 0.05 lets us reject it and treat the series as (weakly) stationary.
    """
    clean = series.dropna().to_numpy(dtype="float64")
    stat, pvalue, used_lag, n_obs, crit, _ = adfuller(clean, autolag="AIC")
    return {
        "series": name,
        "adf_statistic": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "crit_1%": float(crit["1%"]),
        "crit_5%": float(crit["5%"]),
        "crit_10%": float(crit["10%"]),
        "stationary_at_5%": bool(pvalue < 0.05),
    }


def plot_stationarity(series: pd.Series, window: int | None = None):
    """Plot the series with rolling mean and rolling std (stationarity check).

    A stationary series has a roughly constant rolling mean and variance; a
    visible trend or changing spread in these curves signals non-stationarity.
    """
    window = window or INTERVALS_PER_DAY
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(series.index, series, color="lightgray", linewidth=0.7, label="observed")
    ax.plot(roll_mean.index, roll_mean, color="crimson", linewidth=1.3,
            label=f"rolling mean ({window} steps)")
    ax.plot(roll_std.index, roll_std, color="darkorange", linewidth=1.3,
            label=f"rolling std ({window} steps)")
    ax.set_title("Rolling-statistics stationarity check")
    ax.set_xlabel("Date")
    ax.set_ylabel("Internet traffic activity")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)        # prevent double-display in notebooks; savefig still works
    return fig


# ---------------------------------------------------------------------------
# Task 2.IV -- Seasonal decomposition (trend / seasonal / residual)
# ---------------------------------------------------------------------------
def decompose_series(series: pd.Series, period: int | None = None):
    """STL-decompose a series into trend, seasonal and residual components.

    STL (Seasonal-Trend decomposition using LOESS) is robust to outliers and
    handles the strong daily cycle well. ``period`` defaults to 144 (one day
    of 10-minute intervals).
    """
    period = period or INTERVALS_PER_DAY
    clean = series.dropna().astype("float64")
    stl = STL(clean, period=period, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(clean.index, result.observed, color="black", linewidth=0.7)
    axes[0].set_ylabel("observed")
    axes[1].plot(clean.index, result.trend, color="crimson", linewidth=1.0)
    axes[1].set_ylabel("trend")
    axes[2].plot(clean.index, result.seasonal, color="seagreen", linewidth=0.7)
    axes[2].set_ylabel("seasonal")
    axes[3].plot(clean.index, result.resid, color="slateblue", linewidth=0.5)
    axes[3].set_ylabel("residual")
    axes[3].set_xlabel("Date")
    axes[0].set_title(f"STL decomposition (period = {period} intervals = 1 day)")
    fig.tight_layout()
    plt.close(fig)        # prevent double-display in notebooks; savefig still works
    return fig, result


# ---------------------------------------------------------------------------
# Task 2.V -- Autocorrelation / partial autocorrelation
# ---------------------------------------------------------------------------
def plot_acf_pacf(series: pd.Series, lags: int | None = None):
    """Plot ACF and PACF.

    ACF lags default to two full days so the daily seasonal spikes (at lag 144
    and 288) are visible; PACF uses fewer lags to keep the short-range
    dependence structure readable.
    """
    lags = lags or 2 * INTERVALS_PER_DAY
    clean = series.dropna()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    plot_acf(clean, lags=lags, ax=axes[0])
    axes[0].set_title(f"Autocorrelation function (up to {lags} lags)")
    plot_pacf(clean, lags=min(60, lags), ax=axes[1], method="ywm")
    axes[1].set_title("Partial autocorrelation function (up to 60 lags)")
    fig.tight_layout()
    plt.close(fig)        # prevent double-display in notebooks; savefig still works
    return fig


# ---------------------------------------------------------------------------
# Task 2.VI -- Spatial analysis (heatmap over the grid)
# ---------------------------------------------------------------------------
def infer_grid_size(area_totals: pd.Series) -> int:
    """Infer the side length of the square grid from the largest square id.

    The real dataset has 10,000 ids -> a 100x100 grid; synthetic samples use a
    smaller grid. Inferring keeps the heatmap correct in both cases.
    """
    max_id = int(area_totals.index.max())
    side = int(round(np.sqrt(max_id)))
    while side * side < max_id:
        side += 1
    return side


def plot_spatial_heatmap(area_totals: pd.Series, grid_size: int | None = None):
    """Render total traffic as a 2-D heatmap over the city grid.

    Square ids are assumed to enumerate a ``grid_size x grid_size`` lattice in
    row-major order (id 1 -> row 0, col 0). Missing ids are shown as zero. The
    grid size is inferred from the data unless given explicitly.
    """
    grid_size = grid_size or infer_grid_size(area_totals)
    n_cells = grid_size * grid_size
    full = area_totals.reindex(range(1, n_cells + 1), fill_value=0.0)
    grid = full.to_numpy(dtype="float64").reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    im0 = axes[0].imshow(grid, cmap="viridis", origin="lower")
    axes[0].set_title("Total traffic per area (linear)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.log1p(grid), cmap="magma", origin="lower")
    axes[1].set_title("Total traffic per area (log1p)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("grid column")
        ax.set_ylabel("grid row")
    fig.suptitle("Spatial distribution of mobile Internet traffic")
    fig.tight_layout()
    plt.close(fig)        # prevent double-display in notebooks; savefig still works
    return fig


# ---------------------------------------------------------------------------
# Task 2.VII -- Anomaly / outlier detection
# ---------------------------------------------------------------------------
def detect_anomalies(series: pd.Series, z_threshold: float = 4.0):
    """Flag anomalies as deviations from the typical weekly-seasonal profile.

    For each timestamp an *expected* value is taken from the median traffic at
    the same (day-of-week, time-of-day) -- i.e. the recurring weekly pattern.
    The residual (observed - expected) is scaled by a robust estimate (median
    absolute deviation x 1.4826); points beyond ``z_threshold`` are flagged.
    Unlike a rolling-window detector, this surfaces whole-day anomalies (public
    holidays, mass events) because the holiday value is compared against a
    *normal* day rather than against its own neighbourhood.
    """
    s = series.dropna().astype("float64")
    idx = s.index
    frame = pd.DataFrame({"value": s.to_numpy()}, index=idx)
    frame["dow"] = idx.dayofweek
    frame["tod"] = idx.hour * 6 + idx.minute // 10           # 0..143

    expected = frame.groupby(["dow", "tod"])["value"].transform("median")
    # Relative residual: traffic noise is multiplicative (the PDF is roughly
    # log-normal), so a relative deviation is homoscedastic across the daily
    # cycle, unlike the raw residual which is far larger at peak hours.
    floor = max(float(expected.median()) * 0.05, 1e-6)
    rel = (frame["value"] - expected) / expected.clip(lower=floor)
    med = rel.median()
    mad = (rel - med).abs().median() * 1.4826
    z = (rel - med) / (mad if mad > 0 else 1.0)
    flagged = (z.abs() > z_threshold).to_numpy()
    anomalies = s[flagged]

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(s.index, s, color="steelblue", linewidth=0.7, label="traffic")
    ax.plot(idx, expected, color="darkorange", linewidth=0.8, alpha=0.8,
            label="expected (weekly profile)")
    ax.scatter(anomalies.index, anomalies.to_numpy(), color="red", s=20,
               zorder=5, label=f"anomaly (|robust z| > {z_threshold})")
    ax.set_title(f"Anomaly detection -- {len(anomalies)} points flagged")
    ax.set_xlabel("Date")
    ax.set_ylabel("Internet traffic activity")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)        # prevent double-display in notebooks; savefig still works

    anomaly_df = pd.DataFrame(
        {"timestamp": anomalies.index, "value": anomalies.to_numpy(),
         "expected": expected[flagged].to_numpy(),
         "robust_z": z[flagged].to_numpy()}
    ).sort_values("robust_z", key=lambda c: c.abs(), ascending=False)
    return fig, anomaly_df.reset_index(drop=True)
