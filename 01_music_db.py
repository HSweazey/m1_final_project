#!/usr/bin/env python3
"""Load the data/gametime*.csv files into a single SQLite table (data/music.db)."""
"""run ONCE to build the database"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "music.db"
TABLE = "gametime"
CHUNKSIZE = 100_000

TEXT_COLS = [
    "_primaryKey", "dayState", "eventId", "eventName", "venueName", "venueCity",
    "category", "eventInfoSource", "section", "zone", "currency",
]
TIMESTAMP_COLS = ["_firstSeenAt", "_lastSeenAt", "eventStartAt"]
DATE_COLS = ["date"]
INT_COLS = [
    "crawls", "truncated", "daysToEvent", "listings", "tickets", "newListings",
    "goneListings", "ticketsSoldSameListing", "listedTotal",
]
REAL_COLS = [
    "minPrice", "p25Price", "medianPrice", "p75Price", "maxPrice",
    "minPriceBeforeFees", "medianPriceBeforeFees",
    "reportedMinPrice", "reportedMaxPrice", "reportedAvgPrice",
]

# Column order as it appears in the CSVs.
COLUMNS = [
    "_primaryKey", "_firstSeenAt", "_lastSeenAt", "date", "dayState", "crawls",
    "truncated", "daysToEvent", "eventId", "eventName", "eventStartAt", "venueName",
    "venueCity", "category", "eventInfoSource", "section", "zone", "currency",
    "listings", "tickets", "minPrice", "p25Price", "medianPrice", "p75Price",
    "maxPrice", "minPriceBeforeFees", "medianPriceBeforeFees", "newListings",
    "goneListings", "ticketsSoldSameListing", "listedTotal", "reportedMinPrice",
    "reportedMaxPrice", "reportedAvgPrice",
]

SQL_TYPES = (
    {c: "TEXT" for c in TEXT_COLS + TIMESTAMP_COLS + DATE_COLS}
    | {c: "INTEGER" for c in INT_COLS}
    | {c: "REAL" for c in REAL_COLS}
)
DTYPES = (
    {c: "string" for c in TEXT_COLS + TIMESTAMP_COLS + DATE_COLS}
    | {c: "Int64" for c in INT_COLS}
    | {c: "float64" for c in REAL_COLS}
)


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    cols = ",\n    ".join(
        f'"{c}" {SQL_TYPES[c]}' + (" PRIMARY KEY" if c == "_primaryKey" else "")
        for c in COLUMNS
    )
    conn.execute(f"CREATE TABLE {TABLE} (\n    {cols}\n)")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # SQLite has no datetime type, so store timestamps/dates as ISO-8601 text.
    for c in TIMESTAMP_COLS:
        df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    return df


def load_file(conn: sqlite3.Connection, path: Path) -> int:
    placeholders = ", ".join("?" for _ in COLUMNS)
    col_list = ", ".join(f'"{c}"' for c in COLUMNS)
    insert = f"INSERT OR IGNORE INTO {TABLE} ({col_list}) VALUES ({placeholders})"

    n_read = 0
    for chunk in pd.read_csv(path, dtype=DTYPES, chunksize=CHUNKSIZE):
        if list(chunk.columns) != COLUMNS:
            sys.exit(f"Column mismatch in {path.name}: {list(chunk.columns)}")
        chunk = clean(chunk)
        rows = chunk.astype(object).where(chunk.notna(), None).itertuples(index=False, name=None)
        conn.executemany(insert, rows)
        n_read += len(chunk)
    conn.commit()
    return n_read


def main() -> None:
    if DB_PATH.exists(): # added so that we dont overwrite the existing database without confirmation
        answer = input(f"{DB_PATH} already exists. Rebuild it from scratch? (y/n): ")
        if answer.strip().lower() != "y":
            sys.exit("Stopped. Nothing was changed.")
    
    files = sorted((DATA_DIR / "gametime").glob("gametime*.csv"))
    if not files:
        sys.exit(f"No gametime*.csv files found in {DATA_DIR}")

    with sqlite3.connect(DB_PATH) as conn:
        create_table(conn)

        n_total = 0
        for path in files:
            n = load_file(conn, path)
            print(f"{path.name}: {n:,} rows")
            n_total += n

        conn.execute(f'CREATE INDEX idx_{TABLE}_eventId ON {TABLE} ("eventId")')
        conn.execute(f'CREATE INDEX idx_{TABLE}_date ON {TABLE} ("date")')

        n_rows = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        print(f"Skipped {n_total - n_rows:,} duplicate _primaryKey rows")
        print(f"Wrote {n_rows:,} rows to {DB_PATH} (table '{TABLE}')")
    conn.close()


if __name__ == "__main__":
    main()