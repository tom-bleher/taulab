"""Statistics helpers with the 1-sigma conventions used across the TAU lab series.

* ``resolution_sigma`` — rectangular reading window of full width ``resolution``
  maps to ``sigma = resolution / sqrt(12)``.
* ``sem`` — standard error of the mean (``sigma_sample / sqrt(N)``, ddof=1).
* ``combine`` — quadrature sum of independent 1-sigma contributions.
* ``nsigma`` — two-value discrepancy ``|v1 - v2| / sqrt(s1^2 + s2^2)``.
* ``weighted_mean`` — variance-weighted mean of independent measurements.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .datatypes import PhysicalSize

__all__ = [
    "nsigma",
    "resolution_sigma",
    "sem",
    "combine",
    "weighted_mean",
    "drop_outliers",
]


def _unpack(x: Any) -> tuple[float, float]:
    """Accept PhysicalSize, ufloat, (value, sigma) tuple, or plain number."""
    if isinstance(x, PhysicalSize):
        return float(x.value), float(x.uncertainty)
    if hasattr(x, "std_dev") and hasattr(x, "nominal_value"):
        return float(x.nominal_value), float(x.std_dev)
    if isinstance(x, tuple) and len(x) == 2:
        return float(x[0]), float(x[1])
    return float(x), 0.0


def nsigma(v1: Any, v2: Any, *, signed: bool = False) -> float:
    """Combined 1-sigma discrepancy between two measurements.

    Returns ``|v1 - v2| / sqrt(sigma1^2 + sigma2^2)``, or the signed version
    if ``signed=True``.  An exact reference (``sigma = 0``) on either side is
    handled naturally, as is the degenerate case where both sigmas are zero.
    """
    a, da = _unpack(v1)
    b, db = _unpack(v2)
    denom = float(np.hypot(da, db))
    if denom == 0.0:
        return 0.0 if a == b else float("inf")
    raw = (a - b) / denom
    return float(raw if signed else abs(raw))


def resolution_sigma(resolution: float) -> float:
    """1-sigma of a uniform distribution whose full width equals ``resolution``."""
    return float(resolution) / float(np.sqrt(12))


def sem(values: Iterable[float]) -> float:
    """Standard error of the mean, ``sigma_sample / sqrt(N)`` with ddof=1."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def combine(*sigmas: float) -> float:
    """Quadrature sum of independent 1-sigma uncertainties."""
    total = 0.0
    for s in sigmas:
        s = float(s)
        total += s * s
    return float(np.sqrt(total))


def drop_outliers(values, *, n_sigma: float = 3.0, column: str | None = None):
    """Drop entries more than ``n_sigma`` from the sample mean.

    Accepts:

    * A 1-D array — returns a new array with the offending entries removed.
    * A DataFrame with ``column=name`` — returns a new DataFrame with the
      corresponding rows dropped.  The mean / std come from ``column``.

    Uses ``ddof=1``.  If the sample standard deviation is zero the input is
    returned unchanged (nothing can be an outlier).
    """
    try:
        import pandas as pd
    except ImportError:                                   # pragma: no cover
        pd = None
    if pd is not None and isinstance(values, pd.DataFrame):
        if column is None:
            raise ValueError("drop_outliers requires column= for DataFrame input.")
        col = values[column].to_numpy(dtype=float)
        mu, sd = float(np.mean(col)), float(np.std(col, ddof=1))
        if sd == 0.0:
            return values.copy()
        return values.loc[np.abs(col - mu) <= n_sigma * sd].copy()
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return arr.copy()
    mu, sd = float(np.mean(arr)), float(np.std(arr, ddof=1))
    if sd == 0.0:
        return arr.copy()
    return arr[np.abs(arr - mu) <= n_sigma * sd]


def weighted_mean(values: Iterable[Any],
                  errors: Iterable[float] | None = None) -> PhysicalSize:
    """Variance-weighted mean of independent measurements.

    Two call forms:

    * ``weighted_mean([1.0, 1.1, 0.9], [0.05, 0.03, 0.07])``
    * ``weighted_mean([PhysicalSize(1.0, 0.05), ufloat(1.1, 0.03), (0.9, 0.07)])``

    Uses ``wi = 1 / sigma_i^2``; output sigma is ``1 / sqrt(sum wi)``.
    """
    vals: list[float] = []
    errs: list[float] = []
    if errors is None:
        for v in values:
            a, da = _unpack(v)
            vals.append(a)
            errs.append(da)
    else:
        vals = [float(v) for v in values]
        errs = [float(e) for e in errors]
    x = np.asarray(vals, dtype=float)
    s = np.asarray(errs, dtype=float)
    if x.size == 0:
        raise ValueError("weighted_mean needs at least one measurement.")
    if np.any(s <= 0):
        raise ValueError("weighted_mean requires strictly positive sigmas.")
    w = 1.0 / (s * s)
    mean = float(np.sum(w * x) / np.sum(w))
    sigma = float(1.0 / np.sqrt(np.sum(w)))
    return PhysicalSize(mean, sigma)
