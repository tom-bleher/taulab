"""Upstream-taulab compatibility shim.

The reference library at ``github.com/itayf-tau/taulab`` exposes ``odr_fit``,
``fit_functions``, and ``FitResult`` under ``taulab.fit``.  Notebooks migrating
from upstream therefore write ``from taulab.fit import odr_fit, fit_functions``;
this module preserves that exact import path.  Implementations live in
:mod:`taulab.fits` (plural) — this file is a thin re-export layer.
"""
from __future__ import annotations

from .fits import FitResult, extrapolate_to_zero, fit_functions, odr_fit

__all__ = ["odr_fit", "fit_functions", "FitResult", "extrapolate_to_zero"]
