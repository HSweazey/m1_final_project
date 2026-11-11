#!/usr/bin/env python3
"""Load artists.csv, shows.csv, and venues.csv into data/music.db as tables,
then summarize missing (NaN) values by feature for each table."""

# observations:

# ----- venue
# drop capacity because 83.1% missing
# drop county, a little too detailed for our purposes
# country has 594 missing values
# state has 645 missing values
# drop sg_
# drop timezone

import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "music.db"
TABLES = ["artists", "shows", "venues"]


def upload_tables(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Write each CSV to music.db (replacing any existing table of the same name)."""
    frames = {}
    for name in TABLES:
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        df.to_sql(name, conn, if_exists="replace", index=False)
        n_db = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"Uploaded {name}: {len(df):,} rows x {df.shape[1]} cols ({n_db:,} rows in db)")
        frames[name] = df
    return frames


def nan_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percent of missing values per column, sorted descending."""
    n_missing = df.isna().sum()
    return (
        pd.DataFrame({
            "n_missing": n_missing,
            "pct_missing": (n_missing / len(df) * 100).round(2),
            "dtype": df.dtypes.astype(str),
        })
        .sort_values("n_missing", ascending=False)
    )


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        upload_tables(conn)
        print(f"\nTables now in {DB_PATH.name}: "
              f"{[r[0] for r in conn.execute('SELECT name FROM sqlite_master WHERE type=\"table\"')]}")

        # Read back from the db so the summary reflects what was actually stored
        for name in TABLES:
            df = pd.read_sql(f"SELECT * FROM {name}", conn)
            print(f"\n=== NaN summary: {name} ({len(df):,} rows) ===")
            with pd.option_context("display.max_rows", None, "display.width", 120):
                print(nan_summary(df).to_string())


if __name__ == "__main__":
    main()
