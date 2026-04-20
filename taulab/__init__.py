"""taulab — TAU physics-lab helpers, marimo edition.

A rewrite of `github.com/itayf-tau/taulab` for use inside marimo notebooks.
Stays API-compatible with the reference (same module surface: PhysicalSize,
Measurement, nsigma, fit_functions, FitResult, odr_fit, Graph) and adds:

* **Uncertainty-propagating arithmetic on PhysicalSize** via the
  ``uncertainties`` package — add / subtract / multiply / divide / power
  between PhysicalSize, ufloat, and plain numbers all propagate correctly.
* **Instrument uncertainty library** (``taulab.instruments``) — stock 1-sigma
  formulae for the Keysight 34401A DMM (resistance and DC voltage), the
  DSO7012A scope (dual-cursor Delta-V), rulers, calipers, and micrometers.
* **Weighted-mean helper** that accepts PhysicalSize / ufloat / tuple inputs.
* **Auto-seeded ODR** — ``init_values=None`` picks ``np.polyfit`` seeds for
  ``linear``, ``polynomial``, and ``constant`` fit functions.
* **Richer FitResult** — covariance, correlation matrix, pulls, parameter
  dictionaries, one-line ``quality()`` summary.
* **Graph.plot_with_residuals()** — one-call two-panel fit + residuals figure
  with an optional 1-sigma confidence band.
* **CSV loaders** that normalise column-name whitespace (incl. U+00A0) and
  a single-row apparatus-constants loader with a built-in NaN guard.
* **LSD / column_lsd** helpers for reading-resolution uncertainty from
  LabVIEW / CSV exports.
* **Marimo display helpers** (``taulab.display``) — siunitx LaTeX formatters,
  ``fit_callout``, ``params_table``, ``results_table``, ``uncertainty_budget``,
  and ``fit_metrics_accordion``.

Upstream reference: https://github.com/itayf-tau/taulab
"""
from __future__ import annotations

from .datatypes import PhysicalSize, Measurement, ParseResult
from .stats import nsigma, resolution_sigma, sem, combine, weighted_mean, drop_outliers
from .fits import fit_functions, FitResult, odr_fit, extrapolate_to_zero
from .graph import Graph
from .io import read_table, apparatus_row, assert_no_nan, add_error_column
from . import instruments, display, parse, fit

__version__ = "0.1.0"

__all__ = [
    # datatypes
    "PhysicalSize",
    "Measurement",
    "ParseResult",
    # stats
    "nsigma",
    "resolution_sigma",
    "sem",
    "combine",
    "weighted_mean",
    "drop_outliers",
    # fits
    "fit_functions",
    "FitResult",
    "odr_fit",
    "extrapolate_to_zero",
    # plotting
    "Graph",
    # i/o
    "read_table",
    "apparatus_row",
    "assert_no_nan",
    "add_error_column",
    # submodules
    "instruments",
    "display",
    "parse",
    "fit",           # upstream-import shim: `from taulab.fit import odr_fit`
    # version
    "__version__",
]
