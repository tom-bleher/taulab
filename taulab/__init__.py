"""taulab — TAU physics-lab helpers, marimo edition.

A rewrite of `github.com/itayf-tau/taulab` for use inside marimo notebooks.
Exports generic, cross-experiment primitives only: PhysicalSize/Measurement
datatypes, statistics helpers, ODR fitting, and the Graph plotting class.
Experiment-specific instrument models, I/O adapters, and display helpers
live in the individual analysis notebooks that use them.

Upstream reference: https://github.com/itayf-tau/taulab
"""
from __future__ import annotations

from .datatypes import PhysicalSize, Measurement, ParseResult
from .stats import nsigma, resolution_sigma, sem, combine, weighted_mean, drop_outliers
from .fits import fit_functions, FitResult, odr_fit, extrapolate_to_zero
from .graph import Graph
from . import parse
from .parse import read_table

__version__ = "0.2.0"

__all__ = [
    "PhysicalSize",
    "Measurement",
    "ParseResult",
    "nsigma",
    "resolution_sigma",
    "sem",
    "combine",
    "weighted_mean",
    "drop_outliers",
    "fit_functions",
    "FitResult",
    "odr_fit",
    "extrapolate_to_zero",
    "Graph",
    "parse",
    "read_table",
    "__version__",
]
