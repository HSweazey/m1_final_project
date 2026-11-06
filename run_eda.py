#!/usr/bin/env python3
"""
eda.py - one-file exploratory data analysis that writes a self-contained HTML report.

What this script is for
-----------------------
In one line: a reusable tool that gives a quick, automatic first look at any dataset and
saves it as an HTML report you can open in a browser.

Why the project needs it: we have several datasets (the two Gametime CSVs, music.db,
tixinfo) and need to understand each one quickly before deciding how to use it. This
makes that check take one command instead of a custom notebook each time.

What it does, step by step:
1. Loads the data (CSV, SQLite database, Excel, parquet, and more).
2. Works out per-column stats, missing values, distributions, correlations and
   data-quality problems (constant columns, duplicates, outliers).
3. Writes everything to one self-contained HTML file (we keep these in outputs/).

Where it fits: it is a side tool, not a step in the data pipeline, and nothing else
depends on it. It produced the reports in outputs/ and is the quick way to check a new
dataset, for example to decide whether to use tixinfo (to-do step 5).

Command line
------------
    python eda.py data.csv
    python eda.py data.xlsx --sheet "Sales 2024" -o sales_report.html
    python eda.py warehouse.db --table orders
    python eda.py warehouse.db --query "SELECT * FROM orders WHERE total > 0"
    python eda.py postgresql://user:pw@host/dbname --table customers

Python / notebook
-----------------
    import eda
    path = eda.generate_report("data.parquet")            # writes eda_report.html
    path = eda.generate_report(df, title="My DataFrame")  # works on DataFrames too

    res = eda.analyze(df)             # no HTML, just the computed tables
    res.findings                      # data-quality issues (DataFrame)
    res.numeric_summary               # per-column stats (DataFrame)

Supported inputs
----------------
Anything pandas can read: csv/tsv/txt (+ .gz/.zip/.bz2/.xz), xlsx/xls/ods, json/jsonl,
parquet, feather, orc, stata, sas, spss, xml, pickle (trusted files only), a pandas
DataFrame, SQLite files, SQLAlchemy URLs (needs `sqlalchemy` + the DB driver), or an
open SQLAlchemy engine / sqlite3 connection.

Requires: pandas, numpy, matplotlib.
Optional: openpyxl (xlsx), pyarrow (parquet/feather/orc), sqlalchemy (database URLs),
          pyreadstat (spss).

Sample command line usage:
----------------
python3 run_eda.py data/tixinfo.csv -o outputs/tixinfo_report.html
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html as _html
import io
import math
import os
import re
import sqlite3
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure  # no pyplot -> no global backend side effects

__all__ = ["load_data", "analyze", "render_html", "generate_report", "EDAResult"]

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
PLACEHOLDERS = {
    "", "na", "n/a", "n.a.", "nan", "null", "none", "nil", "-", "--", "?", "unknown",
    "missing", "#n/a", "not available", "undefined",
}
SENTINELS = {-999.0, -9999.0, 9999.0, 99999.0}
NONNEG_HINTS = ("age", "price", "count", "qty", "quantity", "salary", "income",
                "height", "weight", "amount", "cost", "distance", "duration")
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
COLORS = {"blue": "#4C78A8", "red": "#E45756", "green": "#54A24B", "light": "#9ecae9"}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _strip_compression(name: str) -> str:
    name = name.lower()
    for ext in (".gz", ".bz2", ".zip", ".xz", ".zst"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _read_delimited(path, sep, **kw) -> pd.DataFrame:
    """Read a delimited text file, trying a couple of encodings and sniffing if needed."""
    encodings = [kw.pop("encoding")] if "encoding" in kw else ["utf-8-sig", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            if sep is None:
                df = pd.read_csv(path, sep=None, engine="python", encoding=enc, **kw)
            else:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False, **kw)
            break
        except UnicodeDecodeError as e:
            last_err = e
    else:
        raise last_err  # type: ignore[misc]
    # A ".csv" that is really ;-, tab- or |-separated shows up as one wide column name.
    if df.shape[1] == 1 and sep == ",":
        header = str(df.columns[0])
        for alt in (";", "\t", "|"):
            if alt in header:
                return _read_delimited(path, alt, encoding=enc, **kw)
    return df


def _read_sql(conn_or_engine, table, query, schema=None) -> pd.DataFrame:
    if query:
        return pd.read_sql(query, conn_or_engine)
    if not table:
        raise ValueError("For database sources, give a table name (--table) or a query (--query).")
    if isinstance(conn_or_engine, sqlite3.Connection):
        return pd.read_sql(f"SELECT * FROM {_quote_ident(table)}", conn_or_engine)
    return pd.read_sql_table(table, conn_or_engine, schema=schema)


def load_data(source, table=None, query=None, sheet=0, **read_kwargs):
    """Load `source` into a DataFrame. Returns (DataFrame, human-readable description)."""
    if isinstance(source, pd.DataFrame):
        return source.copy(), "in-memory DataFrame"

    # Open connection / engine objects
    if not isinstance(source, (str, os.PathLike)):
        if hasattr(source, "connect") or isinstance(source, sqlite3.Connection):
            what = query or f"table {table}"
            return _read_sql(source, table, query), f"database ({what})"
        raise TypeError(f"Don't know how to read a source of type {type(source).__name__}")

    src = str(source)

    # Database URL (but not http/ftp/s3 file URLs)
    if "://" in src and not src.lower().startswith(("http://", "https://", "ftp://", "s3://", "gs://", "file://")):
        try:
            from sqlalchemy import create_engine
        except ImportError as e:
            raise ImportError("Reading from a database URL needs `pip install sqlalchemy` (plus the DB driver).") from e
        engine = create_engine(src)
        try:
            df = _read_sql(engine, table, query)
        finally:
            engine.dispose()
        return df, f"database {engine.url.render_as_string(hide_password=True)} ({query or 'table ' + str(table)})"

    base = _strip_compression(src.split("?")[0])
    ext = Path(base).suffix.lower()

    if ext in {".db", ".sqlite", ".sqlite3", ".db3"}:
        uri = Path(src).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            if not table and not query:
                tables = pd.read_sql(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name", conn)["name"].tolist()
                if len(tables) == 1:
                    table = tables[0]
                else:
                    raise ValueError(f"SQLite file has {len(tables)} tables; pick one with --table. Available: {tables}")
            df = _read_sql(conn, table, query)
        finally:
            conn.close()
        return df, f"{src} ({query or 'table ' + str(table)})"

    kw = dict(read_kwargs)
    if ext == ".csv":
        df = _read_delimited(src, ",", **kw)
    elif ext in {".tsv", ".tab"}:
        df = _read_delimited(src, "\t", **kw)
    elif ext in {".txt", ".dat", ""}:
        df = _read_delimited(src, None, **kw)
    elif ext in {".xlsx", ".xlsm", ".xls", ".ods", ".xlsb"}:
        df = pd.read_excel(src, sheet_name=sheet, **kw)
    elif ext in {".jsonl", ".ndjson"}:
        df = pd.read_json(src, lines=True, **kw)
    elif ext == ".json":
        try:
            df = pd.read_json(src, **kw)
        except ValueError:
            import json
            with open(src, encoding="utf-8") as fh:
                df = pd.json_normalize(json.load(fh))
    elif ext in {".parquet", ".pq"}:
        df = pd.read_parquet(src, **kw)
    elif ext in {".feather", ".arrow"}:
        df = pd.read_feather(src, **kw)
    elif ext == ".orc":
        df = pd.read_orc(src, **kw)
    elif ext in {".pkl", ".pickle"}:
        df = pd.read_pickle(src, **kw)  # only open pickles you trust
    elif ext == ".dta":
        df = pd.read_stata(src, **kw)
    elif ext in {".sas7bdat", ".xpt"}:
        df = pd.read_sas(src, **kw)
    elif ext == ".sav":
        df = pd.read_spss(src, **kw)
    elif ext == ".xml":
        df = pd.read_xml(src, **kw)
    else:
        raise ValueError(f"Unrecognised file type '{ext}'. Load it with pandas yourself and pass the DataFrame.")
    return df, src


# --------------------------------------------------------------------------------------
# Preparation & type inference
# --------------------------------------------------------------------------------------
def _try_convert_text(s: pd.Series):
    """If a text column is really numbers or dates, convert it. Returns (series, info|None)."""
    nn = s.dropna()
    if nn.empty:
        return s, None
    strs = nn.astype(str).str.strip()
    strs = strs[strs != ""]
    if strs.empty:
        return s, None

    # --- numeric stored as text
    num = pd.to_numeric(strs.str.replace(",", "", regex=False), errors="coerce")
    frac_num = num.notna().mean()
    if frac_num >= 0.95 and not strs.str.match(r"^[+-]?0\d").any():  # leading zeros => codes (zip, ids)
        full = s.astype(str).str.strip().str.replace(",", "", regex=False).where(s.notna())
        conv = pd.to_numeric(full, errors="coerce")
        failed = int(conv.isna().sum() - s.isna().sum())
        return conv, {"to": "numeric", "failed": failed}

    # --- dates stored as text
    if frac_num < 0.5:
        sample = strs.sample(min(len(strs), 500), random_state=0)
        if sample.str.contains(r"\d{1,4}[-/.:]\d{1,2}").mean() >= 0.9:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.95:
                    conv = pd.to_datetime(s.astype(str).str.strip().where(s.notna()), errors="coerce")
                    if conv.notna().sum() >= 0.95 * len(nn):
                        return conv, {"to": "datetime", "failed": int(conv.isna().sum() - s.isna().sum())}
    return s, None


def _prepare(df: pd.DataFrame):
    """Normalise column names/types. Returns (df, conversions, notes)."""
    df = df.copy()
    notes: list[str] = []
    conversions: dict[str, dict] = {}

    seen: dict[str, int] = {}
    new_cols = []
    for c in map(str, df.columns):
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}.{seen[c]}")
            notes.append(f"Duplicate column name '{c}' was renamed to '{c}.{seen[c]}'.")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            df[c] = s.dt.tz_convert("UTC").dt.tz_localize(None)
            notes.append(f"'{c}': timezone-aware datetimes converted to naive UTC.")
        elif pd.api.types.is_timedelta64_dtype(s):
            df[c] = s.dt.total_seconds()
            notes.append(f"'{c}': timedeltas converted to seconds.")
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            try:
                s.nunique()
            except TypeError:  # lists / dicts / arrays
                s = s.map(lambda v: str(v) if isinstance(v, (list, dict, set, tuple, np.ndarray)) else v)
                df[c] = s
                notes.append(f"'{c}': nested values (lists/dicts) converted to text.")
            conv, info = _try_convert_text(s)
            if info:
                if isinstance(conv.dtype, pd.DatetimeTZDtype):  # e.g. strings ending in 'Z' or '+00:00'
                    conv = conv.dt.tz_convert("UTC").dt.tz_localize(None)
                    notes.append(f"'{c}': timezone-aware datetimes converted to naive UTC.")
                df[c] = conv
                conversions[c] = info
    return df, conversions, notes


def _infer_kind(s: pd.Series, name: str, cat_threshold: int) -> str:
    nn = s.dropna()
    if nn.empty:
        return "empty"
    nunique = nn.nunique()
    if nunique == 1:
        return "constant"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s) or nunique == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_complex_dtype(s):
        x = nn.to_numpy(dtype="float64", na_value=np.nan)
        with np.errstate(invalid="ignore"):
            is_int = pd.api.types.is_integer_dtype(s) or bool(np.all(np.isfinite(x) & (x % 1 == 0)))
        looks_id = (nn.is_monotonic_increasing or nn.is_monotonic_decreasing
                    or re.search(r"(^|[_\s])id$|^id($|[_\s])", name.lower()) is not None)
        if is_int and len(nn) >= 20 and nunique / len(nn) > 0.95 and looks_id:
            return "identifier"
        if is_int and nunique <= cat_threshold:
            return "categorical"
        return "numeric"
    ratio = nunique / len(nn)
    sample = nn.sample(min(len(nn), 10_000), random_state=0)
    avg_len = sample.astype(str).str.len().mean()
    if len(nn) >= 20 and ratio > 0.95:
        return "text" if avg_len > 30 else "identifier"
    if nunique <= 50 or ratio < 0.05:
        return "categorical"
    return "text" if avg_len > 30 else "categorical"


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------
@dataclass
class EDAResult:
    df: pd.DataFrame
    kinds: dict
    overview: dict
    notes: list
    column_summary: pd.DataFrame
    numeric_summary: pd.DataFrame
    categorical_summary: pd.DataFrame
    text_summary: pd.DataFrame
    datetime_summary: pd.DataFrame
    findings: pd.DataFrame
    corr_pairs: pd.DataFrame
    corr_matrix: pd.DataFrame | None = field(default=None, repr=False)


def _numeric_values(s: pd.Series) -> pd.Series:
    return pd.Series(s.to_numpy(dtype="float64", na_value=np.nan), index=s.index)


def _short(v, n=40) -> str:
    v = str(v)
    return v if len(v) <= n else v[: n - 1] + "…"


def analyze(df: pd.DataFrame, cat_threshold: int = 15) -> EDAResult:
    """Compute all tables and data-quality findings (no plotting)."""
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"Dataset is empty (shape {df.shape}).")
    df, conversions, notes = _prepare(df)
    n, p = df.shape
    kinds = {c: _infer_kind(df[c], c, cat_threshold) for c in df.columns}

    findings: list[dict] = []

    def add(sev, col, issue, detail):
        findings.append(dict(severity=sev, column=col, issue=issue, detail=detail))

    # ---------------- dataset-level overview
    miss_per_col = df.isna().sum()
    dup_rows = int(df.duplicated().sum())
    rows_with_missing = int(df.isna().any(axis=1).sum())
    overview = {
        "rows": n,
        "columns": p,
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "missing_cells": int(miss_per_col.sum()),
        "missing_cells_pct": 100 * miss_per_col.sum() / (n * p),
        "rows_with_missing": rows_with_missing,
        "complete_rows_pct": 100 * (n - rows_with_missing) / n,
        "duplicate_rows": dup_rows,
        "duplicate_pct": 100 * dup_rows / n,
        "kind_counts": pd.Series(kinds).value_counts().to_dict(),
    }
    if dup_rows:
        pct = overview["duplicate_pct"]
        add("medium" if pct >= 5 else "low", "(dataset)", "Duplicate rows",
            f"{dup_rows:,} rows ({pct:.1f}%) exactly repeat an earlier row.")
    if n < 30:
        add("info", "(dataset)", "Very small dataset", f"Only {n} rows - statistics will be unstable.")

    # duplicate columns
    if 1 < p <= 300:
        groups: dict = {}
        for c in df.columns:
            try:
                h = int(pd.util.hash_pandas_object(df[c], index=False).sum())
            except Exception:
                continue
            groups.setdefault((h, str(df[c].dtype)), []).append(c)
        for cols in groups.values():
            if len(cols) > 1:
                same = [c for c in cols[1:] if df[cols[0]].equals(df[c])]
                if same:
                    add("low", ", ".join([cols[0]] + same), "Duplicate columns",
                        "These columns contain identical values.")

    # ---------------- per-column
    col_rows, num_rows, cat_rows, text_rows, dt_rows = [], [], [], [], []
    now = pd.Timestamp.now()

    for i, c in enumerate(df.columns, 1):
        s, kind = df[c], kinds[c]
        nn = s.dropna()
        n_nn = len(nn)
        miss = n - n_nn
        miss_pct = 100 * miss / n
        nunique = int(nn.nunique()) if n_nn else 0
        try:
            sample_vals = ", ".join(_short(v, 18) for v in list(nn.drop_duplicates().head(3)))
        except Exception:
            sample_vals = ""
        col_rows.append(dict(
            position=i, column=c, dtype=str(s.dtype), detected_type=kind, non_null=n_nn,
            missing=miss, missing_pct=miss_pct, unique=nunique,
            unique_pct=100 * nunique / n_nn if n_nn else 0.0, sample_values=sample_vals))

        # --- missingness findings
        if miss_pct == 100:
            add("high", c, "Column is entirely empty", "Every value is missing.")
        elif miss_pct > 50:
            add("high", c, "Majority missing", f"{miss:,} values ({miss_pct:.1f}%) are missing.")
        elif miss_pct > 20:
            add("medium", c, "Substantial missing data", f"{miss:,} values ({miss_pct:.1f}%) are missing.")
        elif miss_pct > 5:
            add("low", c, "Some missing data", f"{miss:,} values ({miss_pct:.1f}%) are missing.")

        # --- conversion findings
        if c in conversions:
            info = conversions[c]
            what = "numbers" if info["to"] == "numeric" else "dates"
            add("low", c, f"{what.capitalize()} stored as text",
                f"Column was text but looks like {what}; analysed as {info['to']}.")
            if info["failed"] > 0:
                add("medium", c, "Mixed / unparseable values",
                    f"{info['failed']:,} entries could not be parsed as {what} and were treated as missing.")

        if kind == "empty":
            continue
        if kind == "constant":
            add("low", c, "Constant column", f"Only one distinct value ({_short(nn.iloc[0], 30)!r}); carries no information.")
            cat_rows.append(dict(column=c, unique=1, top_value=_short(nn.iloc[0]), top_freq=n_nn,
                                 top_pct=100.0, top_values=f"{_short(nn.iloc[0])} (100%)"))
            continue

        # --- numeric
        if kind == "numeric":
            x_all = _numeric_values(s)
            n_inf = int(np.isinf(x_all).sum())
            x = x_all.replace([np.inf, -np.inf], np.nan).dropna()
            if n_inf:
                add("high", c, "Infinite values", f"{n_inf:,} values are +/-inf.")
            if x.empty:
                continue
            q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((x < lo) | (x > hi)).sum()) if iqr > 0 else 0
            out_pct = 100 * n_out / len(x)
            skew = float(x.skew()) if len(x) > 2 else float("nan")
            zeros, negs = int((x == 0).sum()), int((x < 0).sum())
            num_rows.append(dict(
                column=c, count=len(x), missing_pct=miss_pct, mean=x.mean(), std=x.std(), min=x.min(),
                p25=q1, median=med, p75=q3, max=x.max(), skew=skew,
                kurtosis=float(x.kurt()) if len(x) > 3 else float("nan"),
                zeros=zeros, negatives=negs, outliers=n_out, outliers_pct=out_pct))
            if out_pct >= 1:
                add("medium" if out_pct > 5 else "low", c, "Outliers (IQR rule)",
                    f"{n_out:,} values ({out_pct:.1f}%) lie outside [{lo:.4g}, {hi:.4g}]; "
                    f"observed range is [{x.min():.4g}, {x.max():.4g}].")
            if not math.isnan(skew) and abs(skew) > 2:
                add("info", c, "Highly skewed", f"Skewness = {skew:.2f}; consider a log/power transform for modelling.")
            if zeros / len(x) > 0.5:
                add("info", c, "Mostly zeros", f"{100 * zeros / len(x):.1f}% of values are exactly 0.")
            if negs and any(h in c.lower() for h in NONNEG_HINTS):
                add("medium", c, "Negative values (suspicious)", f"{negs:,} negative values in a column whose name suggests it should be non-negative.")
            sent = {v for v in SENTINELS if (x == v).sum() >= 3 and (x == v).sum() / len(x) >= 0.005}
            if sent:
                add("low", c, "Possible sentinel values", f"Values like {sorted(sent)} appear often; they may encode 'missing'.")

        # --- datetime
        elif kind == "datetime":
            mn, mx = nn.min(), nn.max()
            cmp_now = now.tz_localize(nn.dt.tz) if nn.dt.tz is not None else now  # safety net
            future = int((nn > cmp_now).sum())
            dt_rows.append(dict(column=c, min=mn, max=mx, span=str(mx - mn), unique=nunique,
                                missing_pct=miss_pct, future_dates=future))
            if future:
                add("low", c, "Dates in the future", f"{future:,} values are later than today ({now:%Y-%m-%d}).")

        # --- categorical / binary / text / identifier
        else:
            vc = nn.value_counts()
            if kind in ("categorical", "binary"):
                top = vc.head(5)
                cat_rows.append(dict(
                    column=c, unique=nunique, top_value=_short(vc.index[0]), top_freq=int(vc.iloc[0]),
                    top_pct=100 * vc.iloc[0] / n_nn,
                    top_values="; ".join(f"{_short(k, 22)} ({100 * v / n_nn:.0f}%)" for k, v in top.items())))
                if kind == "categorical" and nunique > 50:
                    add("info", c, "High cardinality", f"{nunique:,} distinct values; may need grouping/encoding.")
                if len(vc) > 1 and vc.iloc[0] / n_nn > 0.95:
                    add("info", c, "Near-constant", f"'{_short(vc.index[0], 25)}' makes up {100 * vc.iloc[0] / n_nn:.1f}% of values.")
            else:
                lens = nn.astype(str).str.len()
                text_rows.append(dict(column=c, detected_type=kind, unique=nunique, avg_length=lens.mean(),
                                      min_length=int(lens.min()), max_length=int(lens.max()),
                                      example=_short(nn.iloc[0], 50)))
                if kind == "identifier":
                    add("info", c, "Looks like an identifier", "Values are (almost) all unique; exclude from modelling/aggregations.")

            if not (pd.api.types.is_numeric_dtype(s)):
                strs = nn.astype(str)
                ws = int((strs != strs.str.strip()).sum())
                if ws:
                    add("low", c, "Leading/trailing whitespace", f"{ws:,} values have extra whitespace.")
                ph = int(strs.str.strip().str.lower().isin(PLACEHOLDERS).sum())
                if ph:
                    add("medium" if ph / n > 0.05 else "low", c, "Placeholder 'missing' values",
                        f"{ph:,} values look like disguised missing data (e.g. '', 'N/A', 'null', '-').")
                if kind in ("categorical", "binary") and nunique <= 2000:
                    uniq = pd.Series(strs.unique())
                    norm = uniq.str.strip().str.lower()
                    if norm.nunique() < len(uniq):
                        grp = uniq.groupby(norm).agg(list)
                        bad = grp[grp.str.len() > 1]
                        ex = ", ".join(repr(v) for v in bad.iloc[0][:4])
                        add("medium", c, "Inconsistent spelling/case",
                            f"{len(bad)} group(s) of values differ only by case/whitespace, e.g. {ex}.")

    # ---------------- correlations
    num_cols = [c for c, k in kinds.items() if k == "numeric"]
    corr_matrix, corr_pairs = None, pd.DataFrame(columns=["column_a", "column_b", "r"])
    if 2 <= len(num_cols) <= 200:
        corr_matrix = df[num_cols].replace([np.inf, -np.inf], np.nan).corr()
        vals = corr_matrix.values
        iu = np.triu_indices_from(vals, k=1)
        pairs = pd.DataFrame({"column_a": corr_matrix.columns[iu[0]], "column_b": corr_matrix.columns[iu[1]],
                              "r": vals[iu]}).dropna()
        pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index)
        corr_pairs = pairs[pairs["r"].abs() >= 0.7].head(15).reset_index(drop=True)
        for _, row in pairs[pairs["r"].abs() >= 0.95].head(10).iterrows():
            add("info", f"{row.column_a} / {row.column_b}", "Near-perfect correlation",
                f"Pearson r = {row.r:.3f}; the columns are nearly redundant.")

    findings_df = pd.DataFrame(findings, columns=["severity", "column", "issue", "detail"])
    if not findings_df.empty:
        findings_df = (findings_df.assign(_o=findings_df["severity"].map(SEVERITY_ORDER))
                       .sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True))

    return EDAResult(
        df=df, kinds=kinds, overview=overview, notes=notes,
        column_summary=pd.DataFrame(col_rows), numeric_summary=pd.DataFrame(num_rows),
        categorical_summary=pd.DataFrame(cat_rows), text_summary=pd.DataFrame(text_rows),
        datetime_summary=pd.DataFrame(dt_rows), findings=findings_df,
        corr_pairs=corr_pairs, corr_matrix=corr_matrix)


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------
def esc(x) -> str:
    return _html.escape(str(x), quote=True)


def fmt_num(v) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and math.isnan(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    v = float(v)
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v.is_integer() and abs(v) < 1e15:
        return f"{int(v):,}"
    a = abs(v)
    if a >= 1e9 or (0 < a < 1e-3):
        return f"{v:.3e}"
    if a >= 1000:
        return f"{v:,.1f}"
    return f"{v:.4g}"


def fmt_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:,.0f} {unit}" if unit == "B" else f"{b:,.1f} {unit}"
        b /= 1024
    return str(b)


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def html_table(df: pd.DataFrame, pct_cols=(), bar_cols=(), badge_cols=(), sortable=True) -> str:
    if df is None or df.empty:
        return "<p class='muted'>Nothing to show.</p>"
    head = "".join(f"<th>{esc(str(c).replace('_', ' '))}</th>" for c in df.columns)
    body = []
    for row in df.itertuples(index=False):
        tds = []
        for col, v in zip(df.columns, row):
            if col in badge_cols:
                tds.append(f"<td data-v='{SEVERITY_ORDER.get(v, 9)}'><span class='badge {esc(v)}'>{esc(v)}</span></td>")
            elif _is_num(v):
                isnan = isinstance(v, (float, np.floating)) and math.isnan(v)
                dv = "" if isnan else f" data-v='{float(v)}'"
                if col in pct_cols:
                    txt = "–" if isnan else f"{v:.1f}%"
                    if col in bar_cols and not isnan:
                        w = max(0.0, min(100.0, float(v)))
                        tds.append(f"<td class='num'{dv}><div class='bar'><span style='width:{w:.1f}%'></span></div>{txt}</td>")
                        continue
                else:
                    txt = fmt_num(v)
                tds.append(f"<td class='num'{dv}>{txt}</td>")
            elif v is None or (v is pd.NaT) or (isinstance(v, float) and math.isnan(v)):
                tds.append("<td class='num'>–</td>")
            else:
                tds.append(f"<td>{esc(v)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    cls = "sortable" if sortable else ""
    return f"<div class='tablewrap'><table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


# --------------------------------------------------------------------------------------
# Plots (each returns a base64 PNG <img> tag)
# --------------------------------------------------------------------------------------
def _t(s, n=40) -> str:
    """Truncate + escape '$' (matplotlib mathtext) for use in titles/labels."""
    return _short(s, n).replace("$", r"\$")


def _img(fig: Figure, alt: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    return f"<img alt='{esc(alt)}' src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'>"


def _clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _plot_numeric(s: pd.Series, name: str):
    x = _numeric_values(s).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return None
    fig = Figure(figsize=(4.8, 3.4))
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05})
    bins = int(min(60, max(10, np.sqrt(len(x)))))
    # If extreme tails would squash the histogram, zoom to a wide IQR fence (Q1-3*IQR .. Q3+3*IQR).
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    p1, p99 = max(x.min(), q1 - 3 * iqr), min(x.max(), q3 + 3 * iqr)
    trimmed = iqr > 0 and (x.max() - x.min()) > 5 * (p99 - p1)
    if trimmed:
        pad = 0.05 * (p99 - p1)
        xlo, xhi = p1 - pad, p99 + pad
        n_out = int(((x < xlo) | (x > xhi)).sum())
        ax1.hist(x[(x >= xlo) & (x <= xhi)], bins=bins, color=COLORS["blue"], edgecolor="white", linewidth=0.3)
        ax1.set_xlim(xlo, xhi)
    else:
        ax1.hist(x, bins=bins, color=COLORS["blue"], edgecolor="white", linewidth=0.3)
    ax1.axvline(x.mean(), color=COLORS["red"], ls="--", lw=1.2, label=f"mean {fmt_num(x.mean())}")
    ax1.axvline(x.median(), color=COLORS["green"], ls="-", lw=1.2, label=f"median {fmt_num(x.median())}")
    ax1.legend(fontsize=7, frameon=False)
    ax1.set_title(_t(name), fontsize=10, loc="left")
    ax1.tick_params(labelbottom=False, labelsize=8)
    _clean_axes(ax1)
    box_kw = dict(widths=0.6, patch_artist=True, boxprops=dict(facecolor=COLORS["light"]),
                  flierprops=dict(marker=".", markersize=2, alpha=0.4))
    try:  # matplotlib >= 3.10
        ax2.boxplot(x, orientation="horizontal", **box_kw)
    except TypeError:  # older matplotlib
        ax2.boxplot(x, vert=False, **box_kw)
    ax2.set_yticks([])
    ax2.tick_params(labelsize=8)
    _clean_axes(ax2)
    if trimmed:
        ax2.set_xlim(xlo, xhi)
        fig.text(0.99, -0.02, f"x-axis zoomed to central range ({n_out:,} values off-chart)",
                 ha="right", va="top", fontsize=7, color="#6b7785")
    return _img(fig, f"Distribution of {name}")


def _plot_categorical(s: pd.Series, name: str, top_k: int):
    nn = s.dropna()
    vc = nn.value_counts()
    shown = vc.head(top_k)
    fig = Figure(figsize=(4.8, max(2.0, 0.28 * len(shown) + 1.0)))
    ax = fig.subplots()
    ax.barh(range(len(shown)), shown.values, color=COLORS["blue"])
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels([_t(k, 28) for k in shown.index], fontsize=8)
    ax.invert_yaxis()
    for i, v in enumerate(shown.values):
        ax.text(v, i, f" {100 * v / len(nn):.1f}%", va="center", fontsize=7)
    ax.margins(x=0.18)
    ax.tick_params(axis="x", labelsize=8)
    extra = f", top {top_k} shown" if len(vc) > top_k else ""
    ax.set_title(f"{_t(name, 30)}  ({len(vc):,} unique{extra})", fontsize=10, loc="left")
    _clean_axes(ax)
    return _img(fig, f"Value counts of {name}")


def _plot_datetime(s: pd.Series, name: str):
    nn = s.dropna()
    fig = Figure(figsize=(4.8, 3.0))
    ax = fig.subplots()
    ax.hist(nn.to_numpy(), bins=min(60, max(10, int(np.sqrt(len(nn))))), color=COLORS["blue"], edgecolor="white", linewidth=0.3)
    ax.set_title(_t(name), fontsize=10, loc="left")
    ax.tick_params(labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    _clean_axes(ax)
    return _img(fig, f"Timeline of {name}")


def _plot_missing_bar(miss_pct: pd.Series):
    m = miss_pct[miss_pct > 0].sort_values(ascending=False).head(40)
    if m.empty:
        return None
    fig = Figure(figsize=(7.5, max(2.0, 0.28 * len(m) + 1.0)))
    ax = fig.subplots()
    ax.barh(range(len(m)), m.values, color=COLORS["red"])
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([_t(c, 30) for c in m.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(100, m.max() * 1.1) if m.max() > 90 else m.max() * 1.25)
    for i, v in enumerate(m.values):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=7)
    ax.set_xlabel("% missing", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    _clean_axes(ax)
    return _img(fig, "Missing values by column")


def _plot_missing_matrix(df: pd.DataFrame):
    cols = [c for c in df.columns if df[c].isna().any()][:40]
    if not cols:
        return None
    step = max(1, len(df) // 300)
    mat = df[cols].iloc[::step].isna().to_numpy()
    fig = Figure(figsize=(min(12, 0.45 * len(cols) + 2.5), 4.2))
    ax = fig.subplots()
    ax.imshow(mat, aspect="auto", cmap="gray_r", interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([_t(c, 18) for c in cols], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(f"rows (every {step}{'th' if step > 1 else 'st'})" if step > 1 else "rows", fontsize=9)
    ax.set_title("Missingness pattern (dark = missing)", fontsize=10, loc="left")
    ax.tick_params(axis="y", labelsize=8)
    return _img(fig, "Missingness matrix")


def _plot_corr(corr: pd.DataFrame):
    k = len(corr)
    fig = Figure(figsize=(min(11, 0.55 * k + 3), min(9.5, 0.55 * k + 2.5)))
    ax = fig.subplots()
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = [_t(c, 20) for c in corr.columns]
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    if k <= 15:
        vals = corr.to_numpy()
        for i in range(k):
            for j in range(k):
                if not np.isnan(vals[i, j]):
                    ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(vals[i, j]) > 0.6 else "black")
    return _img(fig, "Correlation heatmap")


# --------------------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------------------
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1f2933;--muted:#6b7785;--line:#e3e7ec;--accent:#4C78A8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{background:#1f2d3d;color:#fff;padding:28px 32px}
header h1{margin:0 0 4px;font-size:24px}
header .sub{color:#b8c4d0;font-size:13px}
nav{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:8px 32px;display:flex;gap:18px;flex-wrap:wrap}
nav a{color:var(--accent);text-decoration:none;font-weight:500;font-size:13px}
nav a:hover{text-decoration:underline}
main{max-width:1280px;margin:0 auto;padding:24px 32px 64px}
section{margin-bottom:36px}
h2{font-size:18px;margin:0 0 12px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:14px;margin:20px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.card .v{font-size:22px;font-weight:650}.card .l{color:var(--muted);font-size:12px}
.card.warn .v{color:#c0392b}
.tablewrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{background:#fafbfc;position:sticky;top:0;white-space:nowrap;font-weight:600}
table.sortable th{cursor:pointer}table.sortable th:hover{background:#eef2f6}
tr:nth-child(even) td{background:#fbfcfd}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{display:inline-block;width:60px;height:8px;background:#eceff3;border-radius:4px;margin-right:6px;vertical-align:middle;overflow:hidden}
.bar span{display:block;height:100%;background:#E45756}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;text-transform:uppercase}
.badge.high{background:#fde2e1;color:#a61e1a}.badge.medium{background:#ffe9cf;color:#a35a00}
.badge.low{background:#fff6c7;color:#7a6200}.badge.info{background:#e1ecf7;color:#2a5a8a}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}
.grid .plot{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;text-align:center}
.plot img{max-width:100%;height:auto}
.muted{color:var(--muted)}.ok{color:#2e7d32;font-weight:600}
ul.notes{margin:8px 0;padding-left:20px;color:var(--muted)}
footer{text-align:center;color:var(--muted);font-size:12px;padding:20px}
"""

JS = """
document.querySelectorAll('table.sortable').forEach(function(t){
  t.querySelectorAll('th').forEach(function(th,i){
    var asc=true;
    th.addEventListener('click',function(){
      var tb=t.tBodies[0],rows=Array.from(tb.rows);
      var val=function(r){var c=r.cells[i];return c.dataset.v!==undefined?parseFloat(c.dataset.v):c.textContent.trim().toLowerCase();};
      rows.sort(function(a,b){var x=val(a),y=val(b);
        if(typeof x==='number'&&typeof y==='number')return asc?x-y:y-x;
        if(typeof x==='number')return -1; if(typeof y==='number')return 1;
        return asc?String(x).localeCompare(String(y)):String(y).localeCompare(String(x));});
      rows.forEach(function(r){tb.appendChild(r)});asc=!asc;
    });
  });
});
"""


def _card(label, value, warn=False):
    return f"<div class='card{' warn' if warn else ''}'><div class='v'>{esc(value)}</div><div class='l'>{esc(label)}</div></div>"


def _grid(items):
    return "<div class='grid'>" + "".join(f"<div class='plot'>{i}</div>" for i in items if i) + "</div>"


def render_html(res: EDAResult, title="EDA Report", source_desc="", plot_sample=50_000,
                max_plot_cols=40, max_categories=15, corr=True) -> str:
    df, kinds, ov = res.df, res.kinds, res.overview
    plot_df = df.sample(plot_sample, random_state=0) if len(df) > plot_sample else df
    sampled_note = (f" Charts use a random sample of {plot_sample:,} rows." if len(df) > plot_sample else "")
    sections, nav = [], []

    def section(sid, name, body):
        nav.append(f"<a href='#{sid}'>{esc(name)}</a>")
        sections.append(f"<section id='{sid}'><h2>{esc(name)}</h2>{body}</section>")

    # ---- overview
    kc = ov["kind_counts"]
    cards = [
        _card("rows", f"{ov['rows']:,}"), _card("columns", f"{ov['columns']:,}"),
        _card("memory", fmt_bytes(ov["memory_bytes"])),
        _card("missing cells", f"{ov['missing_cells_pct']:.2f}%", ov["missing_cells_pct"] > 5),
        _card("complete rows", f"{ov['complete_rows_pct']:.1f}%"),
        _card("duplicate rows", f"{ov['duplicate_rows']:,} ({ov['duplicate_pct']:.1f}%)", ov["duplicate_rows"] > 0),
    ]
    type_cards = [_card(f"{k} columns", f"{v:,}") for k, v in sorted(kc.items(), key=lambda kv: -kv[1])]
    body = f"<div class='cards'>{''.join(cards)}</div><h3>Detected column types</h3><div class='cards'>{''.join(type_cards)}</div>"
    if res.notes:
        body += "<h3>Preparation notes</h3><ul class='notes'>" + "".join(f"<li>{esc(n)}</li>" for n in res.notes) + "</ul>"
    section("overview", "Overview", body)

    # ---- data quality
    f = res.findings
    if f.empty:
        body = "<p class='ok'>✓ No data-quality issues detected by the automatic checks.</p>"
    else:
        counts = f["severity"].value_counts()
        pills = " ".join(f"<span class='badge {s}'>{counts[s]} {s}</span>" for s in SEVERITY_ORDER if s in counts)
        body = f"<p>{pills}</p>" + html_table(f, badge_cols={"severity"})
    section("quality", "Data quality findings", body)

    # ---- column summary
    section("columns", "Column summary",
            html_table(res.column_summary, pct_cols={"missing_pct", "unique_pct"}, bar_cols={"missing_pct"}))

    # ---- statistics
    parts = []
    if not res.numeric_summary.empty:
        parts.append("<h3>Numeric columns</h3>" + html_table(res.numeric_summary, pct_cols={"missing_pct", "outliers_pct"}, bar_cols={"missing_pct"}))
    if not res.categorical_summary.empty:
        parts.append("<h3>Categorical & binary columns</h3>" + html_table(res.categorical_summary, pct_cols={"top_pct"}))
    if not res.datetime_summary.empty:
        parts.append("<h3>Date / time columns</h3>" + html_table(res.datetime_summary, pct_cols={"missing_pct"}))
    if not res.text_summary.empty:
        parts.append("<h3>Text & identifier columns</h3>" + html_table(res.text_summary))
    if parts:
        section("stats", "Summary statistics", "".join(parts))

    # ---- missing-value visuals
    bar = _plot_missing_bar(100 * df.isna().mean())
    mat = _plot_missing_matrix(plot_df)
    if bar:
        section("missing", "Missing values", _grid([bar]).replace("grid'", "grid' style='grid-template-columns:1fr'", 1) + (_grid([mat]) if mat else ""))
    else:
        section("missing", "Missing values", "<p class='ok'>✓ No missing values.</p>")

    # ---- distributions
    def capped(cols, label):
        if len(cols) > max_plot_cols:
            return cols[:max_plot_cols], f"<p class='muted'>Showing the first {max_plot_cols} of {len(cols)} {label} columns.{sampled_note}</p>"
        return cols, (f"<p class='muted'>{sampled_note.strip()}</p>" if sampled_note else "")

    num_cols, note = capped([c for c, k in kinds.items() if k == "numeric"], "numeric")
    if num_cols:
        section("numeric", "Numeric distributions", note + _grid([_plot_numeric(plot_df[c], c) for c in num_cols]))

    cat_cols, note = capped([c for c, k in kinds.items() if k in ("categorical", "binary")], "categorical")
    if cat_cols:
        section("categorical", "Categorical distributions", note + _grid([_plot_categorical(df[c], c, max_categories) for c in cat_cols]))

    dt_cols, note = capped([c for c, k in kinds.items() if k == "datetime"], "datetime")
    if dt_cols:
        section("datetime", "Date / time distributions", note + _grid([_plot_datetime(plot_df[c], c) for c in dt_cols]))

    # ---- correlations
    if corr and res.corr_matrix is not None and len(res.corr_matrix) >= 2:
        cm = res.corr_matrix
        if len(cm) > 40:  # keep the heatmap readable: the 40 columns with the strongest correlations
            strongest = cm.abs().where(~np.eye(len(cm), dtype=bool)).max().sort_values(ascending=False).head(40).index
            cm = cm.loc[strongest, strongest]
        body = _grid([_plot_corr(cm)]).replace("grid'", "grid' style='grid-template-columns:1fr'", 1)
        if not res.corr_pairs.empty:
            body += "<h3>Strongest correlated pairs (|r| ≥ 0.7)</h3>" + html_table(res.corr_pairs)
        section("correlations", "Correlations (Pearson)", body)

    # ---- preview
    section("preview", "Data preview (first 10 rows)", html_table(df.head(10).reset_index(drop=True), sortable=False))

    nav_html = "<nav>" + "".join(nav) + "</nav>"
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>{esc(title)}</title>"
            f"<style>{CSS}</style></head><body>"
            f"<header><h1>{esc(title)}</h1><div class='sub'>Source: {esc(source_desc)} · generated {stamp}</div></header>"
            f"{nav_html}<main>{''.join(sections)}</main>"
            f"<footer>Generated by eda.py</footer><script>{JS}</script></body></html>")


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def generate_report(source, output="eda_report.html", *, table=None, query=None, sheet=0, title=None,
                    max_plot_cols=40, plot_sample=50_000, cat_threshold=15, max_categories=15,
                    corr=True, read_kwargs=None) -> Path:
    """Load `source`, run the analysis and write a self-contained HTML report. Returns the output path."""
    df, desc = load_data(source, table=table, query=query, sheet=sheet, **(read_kwargs or {}))
    res = analyze(df, cat_threshold=cat_threshold)
    if title is None:
        title = f"EDA Report – {Path(str(source)).name}" if isinstance(source, (str, os.PathLike)) and "://" not in str(source) else "EDA Report"
    page = render_html(res, title=title, source_desc=desc, plot_sample=plot_sample,
                       max_plot_cols=max_plot_cols, max_categories=max_categories, corr=corr)
    out = Path(output)
    out.write_text(page, encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate a self-contained HTML EDA report for a dataset.")
    ap.add_argument("source", help="File path, SQLAlchemy URL, or SQLite file")
    ap.add_argument("-o", "--output", help="Output HTML path (default: <name>_eda_report.html)")
    ap.add_argument("--table", help="Table name (database sources)")
    ap.add_argument("--query", help="SQL query (database sources)")
    ap.add_argument("--sheet", default="0", help="Excel sheet name or index (default: first)")
    ap.add_argument("--skiprows", type=int, default=0,
                    help="Number of rows to skip at the top of the file/sheet before the header row")
    ap.add_argument("--title", help="Report title")
    ap.add_argument("--max-plot-cols", type=int, default=40, help="Max charts per chart type (default 40)")
    ap.add_argument("--plot-sample", type=int, default=50_000, help="Rows sampled for charts (default 50000)")
    ap.add_argument("--cat-threshold", type=int, default=15,
                    help="Integer columns with <= this many unique values are treated as categorical (default 15)")
    ap.add_argument("--no-corr", action="store_true", help="Skip the correlation section")
    args = ap.parse_args(argv)

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    if args.output:
        out = args.output
    else:
        stem = Path(_strip_compression(args.source)).stem if "://" not in args.source else "database"
        out = f"{stem}_eda_report.html"
    try:
        path = generate_report(args.source, out, table=args.table, query=args.query, sheet=sheet,
                               title=args.title, read_kwargs={"skiprows": args.skiprows} if args.skiprows else None,
                               max_plot_cols=args.max_plot_cols, plot_sample=args.plot_sample,
                               cat_threshold=args.cat_threshold, corr=not args.no_corr)
    except (ValueError, ImportError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")
    print(f"Report written to {path.resolve()}")


if __name__ == "__main__":
    main()