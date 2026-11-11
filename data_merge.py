#!/usr/bin/env python3
"""Merge the monthly tixinfo-YYYY-MM.csv files in data/ into a single CSV."""

import re
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
OUTPUT = DATA_DIR / "tixinfo_merged.csv"
PATTERN = re.compile(r"tixinfo[-_]\d{4}[-_]\d{2}\.csv")


def main() -> None:
    files = sorted(p for p in DATA_DIR.glob("tixinfo*.csv") if PATTERN.fullmatch(p.name))
    if not files:
        sys.exit(f"No tixinfo-YYYY-MM.csv files found in {DATA_DIR}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        print(f"{path.name}: {len(df):,} rows")
        frames.append(df)

    columns = frames[0].columns
    for path, df in zip(files, frames):
        if not df.columns.equals(columns):
            sys.exit(f"Column mismatch in {path.name}: {list(df.columns)}")

    merged = pd.concat(frames, ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates()
    print(f"Dropped {n_before - len(merged):,} duplicate rows")

    merged.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(merged):,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
