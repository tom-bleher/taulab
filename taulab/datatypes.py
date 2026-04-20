"""Physical-quantity datatypes.

`PhysicalSize` carries a ``(value, uncertainty)`` pair and delegates arithmetic to
the ``uncertainties`` package so that linear error propagation happens
automatically.  It is intentionally interoperable with ``uncertainties.ufloat``:
any expression mixing the two (``K * ps``, ``1 / ps``, ``ps / u``) returns the
``uncertainties`` result, and ``PhysicalSize.from_ufloat`` /
``PhysicalSize.to_ufloat`` round-trip cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from uncertainties import ufloat

__all__ = ["PhysicalSize", "Measurement", "ParseResult"]


@dataclass(frozen=True)
class PhysicalSize:
    """A ``(value, uncertainty)`` pair with uncertainty-propagating arithmetic.

    Mirrors ``taulab.datatypes.PhysicalSize`` from the reference library at the
    field level (``.value``, ``.uncertainty``), and delegates arithmetic to
    ``uncertainties.ufloat`` so that

        >>> a = PhysicalSize(10.0, 0.3)
        >>> b = PhysicalSize(2.0, 0.1)
        >>> a / b
        PhysicalSize(5.0, 0.291...)
    """

    value: float
    uncertainty: float = 0.0

    # ---- constructors ---------------------------------------------------
    @classmethod
    def from_ufloat(cls, u) -> "PhysicalSize":
        return cls(float(u.nominal_value), float(u.std_dev))

    def to_ufloat(self):
        return ufloat(self.value, self.uncertainty)

    # ---- derived quantities ---------------------------------------------
    @property
    def rel(self) -> float:
        """Relative uncertainty ``|sigma / value|``; ``inf`` if ``value == 0``."""
        return abs(self.uncertainty / self.value) if self.value else float("inf")

    @property
    def relative(self) -> float:
        return self.rel

    def latex(self, unit: str = "", *, fmt: str = ".2uL", with_rel: bool = True) -> str:
        """Render as siunitx-style LaTeX: ``$(v ± s)\\,\\mathrm{unit}$ (rel %)``."""
        core = format(self.to_ufloat(), fmt)
        umod = rf"\,\mathrm{{{unit}}}" if unit else ""
        if not with_rel:
            return f"${core}{umod}$"
        rel = 100 * self.rel if np.isfinite(self.rel) else float("nan")
        return f"${core}{umod}$ ({rel:.2f} %)"

    # ---- formatting -----------------------------------------------------
    def __format__(self, spec: str) -> str:
        return self.to_ufloat().__format__(spec or "P")

    def __str__(self) -> str:
        return self.__format__("P")

    def __repr__(self) -> str:
        return f"PhysicalSize({self.value!r}, {self.uncertainty!r})"

    # ---- arithmetic (via ufloat, which propagates linear error) ---------
    def _wrap(self, op: Callable, other: Any, *, reflected: bool = False) -> "PhysicalSize":
        a = self.to_ufloat()
        b = other.to_ufloat() if isinstance(other, PhysicalSize) else other
        result = op(b, a) if reflected else op(a, b)
        if hasattr(result, "nominal_value"):
            return PhysicalSize(float(result.nominal_value), float(result.std_dev))
        return PhysicalSize(float(result), 0.0)

    def __add__(self, o):       return self._wrap(lambda a, b: a + b, o)
    def __radd__(self, o):      return self._wrap(lambda a, b: a + b, o, reflected=True)
    def __sub__(self, o):       return self._wrap(lambda a, b: a - b, o)
    def __rsub__(self, o):      return self._wrap(lambda a, b: a - b, o, reflected=True)
    def __mul__(self, o):       return self._wrap(lambda a, b: a * b, o)
    def __rmul__(self, o):      return self._wrap(lambda a, b: a * b, o, reflected=True)
    def __truediv__(self, o):   return self._wrap(lambda a, b: a / b, o)
    def __rtruediv__(self, o):  return self._wrap(lambda a, b: a / b, o, reflected=True)
    def __neg__(self):          return PhysicalSize(-self.value, self.uncertainty)
    def __abs__(self):          return PhysicalSize(abs(self.value), self.uncertainty)

    def __pow__(self, n):
        u = self.to_ufloat() ** n
        return PhysicalSize(float(u.nominal_value), float(u.std_dev))


@dataclass(frozen=True)
class Measurement:
    """Container for two-axis data with optional errors (upstream-compat)."""

    x: np.ndarray
    y: np.ndarray
    x_err: np.ndarray | None = None
    y_err: np.ndarray | None = None

    def as_arrays(self):
        return (
            np.asarray(self.x, dtype=float),
            np.asarray(self.y, dtype=float),
            None if self.x_err is None else np.asarray(self.x_err, dtype=float),
            None if self.y_err is None else np.asarray(self.y_err, dtype=float),
        )


@dataclass(frozen=True)
class ParseResult:
    """A parsed file: ``data`` DataFrame plus regex-group ``metadata`` dict.

    Returned by :func:`taulab.parse.parse_csv_folder` and
    :func:`taulab.parse.parse_excel_folder`.  ``metadata`` comes from
    ``re.search(pattern, filename).groupdict()`` — use named capture groups
    (``(?P<wavelength>\\d+)``) so the dict is self-describing.
    """

    data: Any                                    # pandas.DataFrame
    metadata: dict = field(default_factory=dict)
    source: str = ""                             # file path for provenance
