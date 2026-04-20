"""Data loaders for lab CSV / Excel files.

``read_table`` is a thin ``pd.read_csv`` / ``pd.read_excel`` wrapper that
normalises column-name whitespace (spaces, no-break spaces), which is by far
the most common cause of silent column-lookup failures in notebooks whose
headers were pasted from Word or Excel.

``apparatus_row`` returns a single row from an apparatus CSV with a NaN guard
on a caller-supplied list of required columns — meant for the global-constants
file every lab session has.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["read_table", "apparatus_row", "assert_no_nan", "add_error_column"]


def read_table(path: str | Path, *, strip_nbsp: bool = True,
               trim_cols: bool = True, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV (``.csv``) or Excel (``.xlsx`` / ``.xls``) file.

    Extra keyword arguments are forwarded to pandas.

    * ``strip_nbsp`` — replaces every U+00A0 in column names with a plain space.
    * ``trim_cols``  — ``.strip()`` every column name.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, **kwargs)
    else:
        df = pd.read_csv(path, **kwargs)
    cols = [str(c) for c in df.columns]
    if strip_nbsp:
        cols = [c.replace("\xa0", " ") for c in cols]
    if trim_cols:
        cols = [c.strip() for c in cols]
    df.columns = cols
    return df


def apparatus_row(path: str | Path, *, row: int = 0,
                  required: list[str] | None = None) -> pd.Series:
    """Read a single-row apparatus-constants CSV and return it as a Series.

    If ``required`` is given, raises ``ValueError`` listing every column that
    is missing from the header or whose row-0 value is NaN.  This is the
    "defensive NaN guard" every lab notebook ends up writing by hand.
    """
    df = read_table(path)
    if len(df) <= row:
        raise ValueError(f"{path}: no row {row} (table has {len(df)} rows)")
    s = df.iloc[row]
    if required:
        missing = [c for c in required
                   if c not in df.columns or pd.isna(s[c])]
        if missing:
            raise ValueError(
                f"{path} is missing / NaN for required columns {missing}"
            )
    return s


def assert_no_nan(df: pd.DataFrame, columns: list[str], *,
                  source: str = "data") -> None:
    """Raise ``ValueError`` if any of ``columns`` contain NaN."""
    bad = df[columns].isna()
    if bad.any().any():
        offending = [c for c in columns if bad[c].any()]
        raise ValueError(f"{source} has NaN values in columns {offending}")


def add_error_column(df: pd.DataFrame, col: str, sigma_func,
                     *, dst: str | None = None,
                     inplace: bool = False) -> pd.DataFrame:
    """Apply ``sigma_func`` to every entry of ``df[col]`` and store it as a new column.

    Default destination is ``f"d{col}"`` (matches the convention used across
    past labs: ``wavelength`` -> ``dwavelength``).  Override with ``dst=``.

    ``sigma_func`` receives the scalar reading and returns a 1-sigma.  For
    array-aware per-column broadcasts (e.g. ``taulab.instruments.dso7012a_dual_cursor``
    on a whole column) you can pass ``sigma_func=lambda v: func(v)``; pandas
    applies it element-wise.
    """
    target = dst or f"d{col}"
    if inplace:
        df[target] = df[col].apply(sigma_func)
        return df
    out = df.copy()
    out[target] = out[col].apply(sigma_func)
    return out
