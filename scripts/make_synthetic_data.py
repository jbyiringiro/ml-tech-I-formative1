"""Generate a synthetic dataset in the exact Telecom Italia Milan file format.

The real 5-19 GB dataset must be downloaded from Harvard Dataverse (see
README.md). This script produces a *small, structurally identical* stand-in so
the entire pipeline can be exercised and graded without the multi-GB download.

Each generated file mimics one real daily file: tab-separated, no header, eight
columns -- ``square_id, time_ms, country_code, sms_in, sms_out, call_in,
call_out, internet`` -- with multiple country-code rows per (square, interval)
so the country-aggregation step is genuinely tested. The synthetic traffic
embeds realistic structure (daily double peak, weekday/weekend effect, a
central-business-district spatial hotspot, multiplicative noise and a few
injected anomalies) so the EDA and forecasting steps yield meaningful output.

Usage
-----
    python scripts/make_synthetic_data.py --grid-size 20
    python scripts/make_synthetic_data.py --grid-size 100 --start 2013-11-01 --end 2013-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CONFIG  # noqa: E402

INTERVALS_PER_DAY = 144
COUNTRY_CODES = [39, 0, 33]          # Italy + two others; summed on load


def daily_profile() -> np.ndarray:
    """Return a 144-length within-day multiplier (two peaks, low at night)."""
    hours = np.arange(INTERVALS_PER_DAY) / 6.0
    morning = 0.55 * np.exp(-((hours - 10.0) / 3.0) ** 2)
    evening = 0.95 * np.exp(-((hours - 21.0) / 3.0) ** 2)
    return 0.10 + morning + evening


def spatial_profile(grid_size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a per-square multiplier with a central hotspot + random texture."""
    n = grid_size * grid_size
    rows, cols = np.divmod(np.arange(n), grid_size)
    centre = (grid_size - 1) / 2.0
    dist2 = (rows - centre) ** 2 + (cols - centre) ** 2
    hotspot = 1.6 * np.exp(-dist2 / (2.0 * (grid_size / 4.0) ** 2))
    texture = rng.lognormal(mean=0.0, sigma=0.35, size=n)
    # A handful of secondary hotspots (e.g. stadium, airport).
    for _ in range(max(1, grid_size // 8)):
        idx = rng.integers(0, n)
        hotspot[idx] += rng.uniform(0.8, 1.8)
    return 0.20 + hotspot * texture


def weekday_factor(date: pd.Timestamp) -> float:
    """Lower traffic on weekends."""
    return {5: 0.82, 6: 0.68}.get(date.weekday(), 1.0)


def generate(grid_size: int, start: str, end: str, out_dir: Path,
             amplitude: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = grid_size * grid_size
    dprofile = daily_profile()
    sprofile = spatial_profile(grid_size, rng)
    dates = pd.date_range(start, end, freq="D")
    shares = np.array([0.6, 0.25, 0.15])             # per-country split

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(dates)} daily files | grid {grid_size}x{grid_size} "
          f"({n} squares) -> {out_dir}")

    for date in dates:
        # (n_squares, 144) total internet activity for this day.
        base = amplitude * np.outer(sprofile, dprofile) * weekday_factor(date)
        noise = rng.lognormal(mean=0.0, sigma=0.22, size=base.shape)
        total = base * noise

        # Inject occasional anomalies (spikes / outages).
        if rng.random() < 0.25:
            sq = rng.integers(0, n)
            t0 = rng.integers(0, INTERVALS_PER_DAY - 6)
            total[sq, t0:t0 + 6] *= rng.uniform(3.0, 6.0)        # spike
        if rng.random() < 0.15:
            sq = rng.integers(0, n)
            t0 = rng.integers(0, INTERVALS_PER_DAY - 12)
            total[sq, t0:t0 + 12] *= 0.05                        # outage

        # Timestamps (epoch ms) for the 144 intervals.
        day_start = pd.Timestamp(date)
        times_ms = [
            int((day_start + pd.Timedelta(minutes=10 * t)).value // 1_000_000)
            for t in range(INTERVALS_PER_DAY)
        ]

        # Expand to one row per (square, interval, country).
        per_country = total[:, :, None] * shares[None, None, :]   # (n,144,3)
        n_rows = per_country.size
        square_col = np.repeat(np.arange(1, n + 1), INTERVALS_PER_DAY * 3)
        time_col = np.tile(np.repeat(times_ms, 3), n)
        country_col = np.tile(COUNTRY_CODES, n * INTERVALS_PER_DAY)
        internet_col = per_country.ravel().astype("float64")

        # ~0.4% of internet values missing, like the real files.
        missing = rng.random(n_rows) < 0.004
        internet_obj = internet_col.round(4).astype(object)
        internet_obj[missing] = ""

        frame = pd.DataFrame({
            0: square_col,
            1: time_col,
            2: country_col,
            3: rng.integers(0, 30, n_rows),      # sms_in   (ignored downstream)
            4: rng.integers(0, 30, n_rows),      # sms_out  (ignored)
            5: rng.integers(0, 20, n_rows),      # call_in  (ignored)
            6: rng.integers(0, 20, n_rows),      # call_out (ignored)
            7: internet_obj,                      # internet activity
        })
        fname = out_dir / f"sms-call-internet-mi-{date:%Y-%m-%d}.txt"
        frame.to_csv(fname, sep="\t", header=False, index=False)
    print(f"Done. {len(dates)} files written to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic Milan-format data.")
    p.add_argument("--grid-size", type=int, default=20,
                   help="grid side length (20 -> 400 squares; 100 -> real 10,000)")
    p.add_argument("--start", default=CONFIG.dates["study_start"])
    p.add_argument("--end", default=CONFIG.dates["study_end"])
    p.add_argument("--out", default=str(CONFIG.raw_telecom),
                   help="output directory for the daily .txt files")
    p.add_argument("--amplitude", type=float, default=350.0,
                   help="overall traffic scale")
    p.add_argument("--seed", type=int, default=CONFIG.get("random_seed", 42))
    args = p.parse_args()
    generate(args.grid_size, args.start, args.end, Path(args.out),
             args.amplitude, args.seed)


if __name__ == "__main__":
    main()
