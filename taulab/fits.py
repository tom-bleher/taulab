"""Fit-function library, orthogonal-distance regression, and ``FitResult``.

``fit_functions`` exposes named ``(A, x)`` callables that plug straight into
``scipy.odr``.  ``odr_fit`` is a thin, reference-compatible wrapper that seeds
``beta0`` from OLS when it can (``linear``, ``polynomial``, ``constant``) and
returns a ``FitResult`` with chi^2, p-value, covariance, and reporting helpers.

Convention (matches the original TAU taulab): ``A[0]`` is the constant term;
every seed vector therefore begins with the intercept/offset.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
from scipy import odr as _odr
from scipy.stats import chi2 as _chi2_dist

from uncertainties import ufloat

from .datatypes import PhysicalSize

__all__ = ["fit_functions", "FitResult", "odr_fit", "extrapolate_to_zero"]


# --- Fit-function library --------------------------------------------------
def _f_constant(A, x):      return np.full_like(np.asarray(x, dtype=float), A[0])
def _f_linear(A, x):        return A[1] * x + A[0]
def _f_polynomial(A, x):    return A[2] * x * x + A[1] * x + A[0]
def _f_exponential(A, x):   return A[2] * np.exp(A[1] * x) + A[0]
def _f_logarithmic(A, x):   return A[2] * np.log(A[1] * x) + A[0]
def _f_sinusoidal(A, x):    return A[3] * np.sin(A[1] * x + A[2]) + A[0]
def _f_gaussian(A, x):      return A[3] * np.exp(-((x - A[1]) ** 2) / (2 * A[2] ** 2)) + A[0]
def _f_lorentzian(A, x):    return A[3] / (1.0 + ((x - A[1]) / A[2]) ** 2) + A[0]
def _f_power_law(A, x):     return A[2] * np.power(x, A[1]) + A[0]
def _f_optics(A, x):
    """Thin-lens variant from upstream taulab: ``A[1]*x / (x - A[1]) + A[0]``."""
    return A[1] * x / (x - A[1]) + A[0]


def _make_polynomial_n(n: int):
    """Return an ``(A, x) -> sum_{i=0..n} A[i] * x**i`` callable.

    Horner evaluation for numerical stability.  ``A[0]`` is the constant
    term, matching the rest of the fit-function library.
    """
    if n < 0:
        raise ValueError("polynomial_n needs a non-negative degree.")

    def _poly(A, x):
        x = np.asarray(x, dtype=float)
        result = np.full_like(x, float(A[n]))
        for i in range(n - 1, -1, -1):
            result = result * x + A[i]
        return result

    _poly.__name__ = f"polynomial_n({n})"
    return _poly


def _make_exponent_sum_n(n: int):
    """Return an ``(A, x) -> sum_{i=0..n-1} A[2i] * exp(A[2i+1] * x)`` callable.

    ``A`` must have length ``2n``.  Matches the upstream signature (no
    constant term; add a near-zero-rate pair to emulate one if needed).
    """
    if n < 1:
        raise ValueError("exponent_sum_n needs n >= 1.")

    def _expsum(A, x):
        x = np.asarray(x, dtype=float)
        total = np.zeros_like(x)
        for i in range(n):
            total = total + A[2 * i] * np.exp(A[2 * i + 1] * x)
        return total

    _expsum.__name__ = f"exponent_sum_n({n})"
    return _expsum


fit_functions = SimpleNamespace(
    constant       = _f_constant,
    linear         = _f_linear,
    polynomial     = _f_polynomial,
    polynomial_n   = _make_polynomial_n,
    exponential    = _f_exponential,
    exponent_sum_n = _make_exponent_sum_n,
    logarithmic    = _f_logarithmic,
    sinusoidal     = _f_sinusoidal,
    gaussian       = _f_gaussian,
    lorentzian     = _f_lorentzian,
    power_law      = _f_power_law,
    optics         = _f_optics,
)


# --- FitResult --------------------------------------------------------------
class FitResult:
    """Reporting-friendly wrapper around ``scipy.odr.Output``.

    Attributes
    ----------
    params, errors, error : np.ndarray
        Best-fit parameters and their uncertainty estimates.  ``error`` is a
        legacy alias for ``errors`` kept for reference-taulab compatibility.
    cov : np.ndarray | None
        Parameter covariance matrix.
    chi2, redchi, dof, p_value : float
        Goodness-of-fit summary.
    x, y, sx, sy : np.ndarray
        The inputs the fit was run on.
    param_names : list[str]
        Defaults to ``['a0', 'a1', ...]``; callers may override.
    """

    def __init__(self, out, fit_func, x, y, sx, sy, *, param_names=None):
        self._out      = out
        self._fit_func = fit_func
        self.x  = np.asarray(x,  dtype=float)
        self.y  = np.asarray(y,  dtype=float)
        self.sx = np.asarray(sx, dtype=float) if sx is not None else None
        self.sy = np.asarray(sy, dtype=float) if sy is not None else None

        self.params = np.asarray(out.beta,    dtype=float)
        self.errors = np.asarray(out.sd_beta, dtype=float)
        self.error  = self.errors  # reference-taulab alias

        self.cov = (np.asarray(out.cov_beta, dtype=float)
                    if getattr(out, "cov_beta", None) is not None else None)

        self.dof     = max(1, len(self.x) - len(self.params))
        self.chi2    = float(out.sum_square)
        self.redchi  = self.chi2 / self.dof
        self.p_value = float(_chi2_dist.sf(self.chi2, self.dof))

        self.param_names = (list(param_names) if param_names is not None
                            else [f"a{i}" for i in range(len(self.params))])

    # ---- parameter accessors -------------------------------------------
    def param(self, i: int) -> PhysicalSize:
        return PhysicalSize(float(self.params[i]), float(self.errors[i]))

    def params_as_ufloats(self) -> tuple:
        return tuple(ufloat(p, e) for p, e in zip(self.params, self.errors))

    def params_dict(self) -> dict[str, PhysicalSize]:
        return {n: self.param(i) for i, n in enumerate(self.param_names)}

    def correlation_matrix(self) -> np.ndarray | None:
        if self.cov is None:
            return None
        s = np.sqrt(np.diag(self.cov))
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.cov / np.outer(s, s)

    # ---- prediction / residuals ----------------------------------------
    def predict(self, x):
        return self._fit_func(self.params, np.asarray(x, dtype=float))

    def extrapolate(self, x):
        """Alias for :meth:`predict` — matches the upstream taulab name."""
        return self.predict(x)

    @property
    def measurement(self):
        """The inputs bundled as a :class:`Measurement` (upstream-compat)."""
        from .datatypes import Measurement
        return Measurement(self.x, self.y, self.sx, self.sy)

    @property
    def raw_output(self):
        """The underlying :class:`scipy.odr.Output` (upstream alias for ``_out``)."""
        return self._out

    def residuals(self) -> np.ndarray:
        return self.y - self.predict(self.x)

    def pulls(self) -> np.ndarray | None:
        """Residuals divided by ``sigma_y`` (None if ``sy`` was not given)."""
        if self.sy is None:
            return None
        return self.residuals() / self.sy

    # ---- text reports ---------------------------------------------------
    def metrics_report(self) -> str:
        return (
            f"Degrees of Freedom:   {self.dof}\n"
            f"Reduced Chi-Squared:  {self.redchi:.4g}\n"
            f"P-Value:              {self.p_value:.4g}"
        )

    def params_report(self) -> str:
        lines = ["Fit parameters:"]
        for name, p, e in zip(self.param_names, self.params, self.errors):
            rel = 100 * abs(e / p) if p else float("inf")
            lines.append(f"  {name}: {p:.6g} +/- {e:.2g}   ({rel:.2g}%)")
        return "\n".join(lines)

    def summary(self) -> str:
        return self.metrics_report() + "\n\n" + self.params_report()

    def __str__(self) -> str:                         # upstream parity
        return self.summary()

    def print(self) -> None:                          # upstream convenience
        """Print the summary report (upstream-compat one-call dump)."""
        import builtins
        builtins.print(self.summary())

    def quality(self) -> str:
        """One-line English summary of goodness-of-fit."""
        r, p = self.redchi, self.p_value
        if r < 0.5:
            head = "Overfit / errors likely overestimated"
        elif r > 2.0:
            head = "Poor fit / errors likely underestimated"
        else:
            head = "Acceptable fit"
        return f"{head}  (chi^2/dof = {r:.2f}, p = {p:.3f}, dof = {self.dof})"


# --- odr_fit ----------------------------------------------------------------
def _seed_from_ols(fit_func, x_data, y_data) -> Sequence[float] | None:
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    if fit_func is _f_linear:
        slope, intercept = np.polyfit(x, y, 1)
        return (float(intercept), float(slope))
    if fit_func is _f_polynomial:
        c2, c1, c0 = np.polyfit(x, y, 2)
        return (float(c0), float(c1), float(c2))
    if fit_func is _f_constant:
        return (float(np.mean(y)),)
    return None


def odr_fit(fit_func, init_values, x_data, x_err, y_data, y_err,
            *, param_names=None) -> FitResult:
    """Orthogonal-distance regression with errors on both axes.

    Call signature matches the reference taulab exactly:

        odr_fit(fit_func, init_values, x_data, x_err, y_data, y_err)

    Passing ``init_values=None`` auto-seeds ``beta0`` with ``np.polyfit`` for
    ``fit_functions.linear``, ``.polynomial``, and ``.constant``; other fits
    require an explicit seed.

    ``param_names`` is forwarded to ``FitResult`` so reports and tables carry
    meaningful labels.
    """
    if init_values is None:
        init_values = _seed_from_ols(fit_func, x_data, y_data)
        if init_values is None:
            name = getattr(fit_func, "__name__", repr(fit_func))
            raise ValueError(
                f"init_values is required for fit function {name!r} -- "
                "no automatic seed is implemented."
            )
    model = _odr.Model(fit_func)
    data = _odr.RealData(
        x_data, y_data,
        sx=x_err if x_err is not None else None,
        sy=y_err if y_err is not None else None,
    )
    out = _odr.ODR(data, model, beta0=list(init_values)).run()
    return FitResult(out, fit_func, x_data, y_data, x_err, y_err,
                     param_names=param_names)


def extrapolate_to_zero(result: FitResult, x0: float = 0.0) -> PhysicalSize:
    """Predict ``result`` at ``x0`` with uncertainty from the parameter covariance.

    Numerically propagates the parameter covariance through the fit function
    via symmetric finite differences, which works for every entry in
    :data:`fit_functions` without hand-coded gradients.  ``x0=0`` recovers the
    intercept-at-zero pattern used throughout past labs (laser wavelengths,
    black-body extrapolations).

    Returns a :class:`PhysicalSize`.  If the fit has no covariance matrix
    (e.g. a degenerate one-parameter fit), uncertainty is reported as zero.
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float))
    y_val = float(result._fit_func(result.params, x)[0])
    if result.cov is None:
        return PhysicalSize(y_val, 0.0)
    eps = 1e-6 * (np.abs(result.params) + 1.0)
    grads = np.zeros(len(result.params))
    for i in range(len(result.params)):
        p_plus  = result.params.copy(); p_plus[i]  += eps[i]
        p_minus = result.params.copy(); p_minus[i] -= eps[i]
        grads[i] = (result._fit_func(p_plus, x)[0]
                    - result._fit_func(p_minus, x)[0]) / (2 * eps[i])
    var = float(grads @ result.cov @ grads)
    return PhysicalSize(y_val, float(np.sqrt(max(var, 0.0))))
