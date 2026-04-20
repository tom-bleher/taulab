"""Instrument uncertainty helpers for the TAU lab bench.

Every function returns a 1-sigma in the same units as its input.  DMM accuracies
follow the manufacturer's ``% of reading + % of range`` spec, combined
linearly; rectangular reading resolution is always ``sigma = res / sqrt(12)``.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .stats import resolution_sigma

__all__ = [
    "keysight_34401a_resistance",
    "keysight_34401a_voltage",
    "dso7012a_dual_cursor",
    "ruler",
    "caliper",
    "micrometer",
    "reading_lsd",
    "column_lsd",
]


# --- Keysight / Agilent / HP 34401A 6.5-digit DMM, 1-year spec (23 C +/- 5 C) -
#  Values from the 34401A user manual, page 216.
#  Each tuple is (% of reading, % of range).  Range coefficient varies.
_34401A_RESISTANCE_RANGES: dict[int, tuple[float, float]] = {
    100:        (0.010e-2, 0.004e-2),   # 100 Ohm  -- R_x (~3 Ohm) lives here
    1_000:      (0.010e-2, 0.001e-2),
    10_000:     (0.010e-2, 0.001e-2),
    100_000:    (0.010e-2, 0.001e-2),   # 100 kOhm -- R_y (~11.1 kOhm) lives here
    1_000_000:  (0.010e-2, 0.001e-2),
}
_34401A_DCV_RANGES: dict[float, tuple[float, float]] = {
    0.1:   (0.0050e-2, 0.0035e-2),
    1.0:   (0.0040e-2, 0.0007e-2),
    10.0:  (0.0035e-2, 0.0005e-2),
    100.0: (0.0045e-2, 0.0006e-2),
    1000.0:(0.0045e-2, 0.0010e-2),
}


def _pick_range(magnitude: float, ranges) -> float:
    for r in sorted(ranges):
        if magnitude <= r:
            return r
    return max(ranges)


def keysight_34401a_resistance(R: float, range_: float | str = "auto",
                               *, include_lsd: bool = True) -> float:
    """Resistance 1-sigma for the HP / Agilent / Keysight 34401A DMM.

    ``range_='auto'`` picks the smallest of 100, 1k, 10k, 100k, 1M Ohm that
    contains ``|R|``.  When ``include_lsd=True`` the least-significant-decimal
    quantisation ``LSD/sqrt(12)`` of the displayed reading is combined in
    quadrature; this matters only for the lowest ranges.
    """
    rng = _pick_range(abs(R), _34401A_RESISTANCE_RANGES) if range_ == "auto" else float(range_)
    pct_rdg, pct_rng = _34401A_RESISTANCE_RANGES[int(rng)]
    sigma_dev = pct_rdg * abs(R) + pct_rng * rng
    if include_lsd:
        sigma_dev = float(np.hypot(sigma_dev, resolution_sigma(reading_lsd(R))))
    return float(sigma_dev)


def keysight_34401a_voltage(V: float, range_: float | str = "auto",
                            *, include_lsd: bool = True) -> float:
    """DC-voltage 1-sigma for the 34401A DMM."""
    rng = _pick_range(abs(V), _34401A_DCV_RANGES) if range_ == "auto" else float(range_)
    pct_rdg, pct_rng = _34401A_DCV_RANGES[float(rng)]
    sigma_dev = pct_rdg * abs(V) + pct_rng * rng
    if include_lsd:
        sigma_dev = float(np.hypot(sigma_dev, resolution_sigma(reading_lsd(V))))
    return float(sigma_dev)


# --- Keysight DSO 7012A oscilloscope, dual-cursor Delta-V --------------------
def dso7012a_dual_cursor(V, *, vdiv: float | None = None,
                         gain_frac: float = 0.024, floor: float = 5e-3,
                         lsd: float | None = None):
    """InfiniiVision 7000 Series dual-cursor Delta-V accuracy.

    Spec (data-sheet 5989-7736):

        sigma(Delta-V) = sqrt( (gain_frac * |V|)^2 + (0.4 % FS)^2 )

    where FS = 8 * V/div.  When V/div is matched so |Delta-V| spans most of
    the screen, FS ~ |V| and the 0.4 % FS term is negligible except at the
    smallest readings; for that regime we fall back to a hard floor (default
    5 mV).  The offset term cancels in a two-cursor reading and is NOT added
    here.  Pass ``vdiv`` explicitly if you logged it — the function will then
    use the exact 0.4 % FS quantisation instead of the floor.

    Optionally adds ``LSD / sqrt(12)`` in quadrature.  Accepts scalars and
    arrays.  Returns the same shape as ``V``.
    """
    V_arr = np.asarray(V, dtype=float)
    if vdiv is not None:
        fs = 8.0 * float(vdiv)
        base = np.sqrt((gain_frac * np.abs(V_arr)) ** 2 + (0.004 * fs) ** 2)
    else:
        base = np.sqrt((gain_frac * np.abs(V_arr)) ** 2 + float(floor) ** 2)
    if lsd is not None and lsd > 0 and math.isfinite(lsd):
        base = np.sqrt(base ** 2 + resolution_sigma(lsd) ** 2)
    return base if V_arr.ndim else float(base)


# --- Length instruments -----------------------------------------------------
def ruler(resolution: float = 1e-3) -> float:
    """1-sigma of a single ruler reading (default 1 mm resolution)."""
    return resolution_sigma(resolution)


def caliper(resolution: float = 0.05e-3) -> float:
    """1-sigma of a caliper reading (default 0.05 mm resolution)."""
    return resolution_sigma(resolution)


def micrometer(resolution: float = 1e-5) -> float:
    """1-sigma of a micrometer reading (default 10 um resolution)."""
    return resolution_sigma(resolution)


# --- Least-significant-decimal helpers --------------------------------------
def reading_lsd(v) -> float:
    """Least-significant decimal place of a stored numeric reading.

    Examples
    --------
    >>> reading_lsd(0.8)         # one decimal place
    0.1
    >>> reading_lsd(1.28125)     # five decimal places
    1e-05
    >>> reading_lsd(11100)       # two trailing zeros are insignificant
    100.0
    >>> reading_lsd(2e-05)       # scientific-notation aware
    1e-05

    Uses Python's shortest round-trip float repr and counts trailing zeros
    on integers as insignificant (so ``11100 -> 100``).  Non-finite inputs
    and zero return 0.0 so they contribute no spurious quantisation term.
    """
    f = float(v)
    if not math.isfinite(f) or f == 0.0:
        return 0.0
    s = format(abs(f), ".15g")
    if "e" in s:
        mant, exp_str = s.split("e")
        exp = int(exp_str)
        decimals = len(mant.split(".")[1]) if "." in mant else 0
        return 10.0 ** (exp - decimals)
    if "." in s:
        return 10.0 ** -len(s.split(".")[1])
    trimmed = s.rstrip("0")
    trailing = len(s) - len(trimmed)
    return 10.0 ** trailing


def column_lsd(values: Iterable[float]) -> float:
    """Finest LSD seen in a column of readings.

    When a LabVIEW / CSV export round-trips through Python's shortest-repr,
    a cell that lands on a round number (``2``) appears to have a much
    coarser LSD than its neighbours (``1.28125``).  Taking the column
    minimum recovers the true recording precision and prevents per-row
    uncertainty cliffs at round values.
    """
    arr = np.asarray(list(values), dtype=float)
    lsds = np.array([reading_lsd(float(v)) for v in arr])
    positive = lsds[lsds > 0]
    return float(positive.min()) if positive.size else 0.0
