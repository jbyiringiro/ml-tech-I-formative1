"""Task 1 -- Data Handling & Memory Management (end-to-end run).

Discovers the raw daily files, quantifies the memory optimisation on a sample
file, then streams every file to build the consolidated traffic matrix
(Parquet). Run from the project root:

    python scripts/run_task1.py
    python scripts/run_task1.py --max-files 5      # quick check on a subset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG                                      # noqa: E402
from src.data_loading import (                                     # noqa: E402
    build_traffic_matrix,
    compute_area_totals,
    discover_raw_files,
    memory_comparison,
    resolve_target_areas,
)
from src.utils import human_bytes, print_system_report, save_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 1.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="process only the first N daily files (quick test)")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# TASK 1 -- DATA HANDLING & MEMORY MANAGEMENT")
    print("#" * 70 + "\n")

    # 1.V -- hardware / software environment
    print_system_report()

    # Discover raw files
    files = discover_raw_files()
    if args.max_files:
        files = files[: args.max_files]
    print(f"\n{len(files)} raw daily file(s) selected for processing.")
    total_raw = sum(f.stat().st_size for f in files)
    print(f"Raw text size on disk: {human_bytes(total_raw)}")

    # 1.III -- memory before/after optimisation (single sample file)
    print("\n--- Memory optimisation evidence (sample file) ---")
    mem_report = memory_comparison(files[0])
    save_table(mem_report, "task1_memory_comparison")

    # 1.I/1.II -- stream + transform + consolidate
    print("\n--- Building the consolidated traffic matrix ---")
    matrix = build_traffic_matrix(files, save=True)

    # Derived aggregates used by Tasks 2 and 3
    compute_area_totals(matrix, save=True)
    areas = resolve_target_areas(matrix)
    with open(CONFIG.processed / "target_areas.json", "w", encoding="utf-8") as fh:
        json.dump(areas, fh, indent=2)
    print(f"\nTarget areas resolved: {areas}")
    print(f"  (saved -> {CONFIG.processed / 'target_areas.json'})")

    print("\nTask 1 complete. Outputs in data/processed/ and results/tables/.")


if __name__ == "__main__":
    main()
