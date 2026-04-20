"""Marimo-specific display helpers.

siunitx-style LaTeX formatters, result callouts, parameter tables, and
fit-metrics accordions.  ``marimo`` is imported lazily on each call so this
module is still importable in non-marimo contexts (the functions just won't
run).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from uncertainties import ufloat

from .datatypes import PhysicalSize
from .fits import FitResult
from .stats import nsigma as _nsigma

__all__ = [
    "latex_value",
    "format_ufloat",
    "rel_pct",
    "fit_callout",
    "params_table",
    "results_table",
    "uncertainty_budget",
    "fit_metrics_accordion",
    "chi_squared_badge",
    "nsigma_inline",
    "result_comparison_table",
    "statistical_assessment_prose",
    "siunitx_preamble",
    "SIUNITX_PREAMBLE",
]


# --- Core formatters --------------------------------------------------------
def _as_ufloat(v: Any):
    if isinstance(v, PhysicalSize):
        return v.to_ufloat()
    if hasattr(v, "nominal_value") and hasattr(v, "std_dev"):
        return v
    if isinstance(v, tuple) and len(v) == 2:
        return ufloat(float(v[0]), float(v[1]))
    return ufloat(float(v), 0.0)


def rel_pct(v: Any) -> float:
    """Relative uncertainty in percent (NaN if the central value is zero)."""
    uf = _as_ufloat(v)
    if not uf.nominal_value:
        return float("nan")
    return 100 * abs(uf.std_dev / uf.nominal_value)


def format_ufloat(v: Any, *, fmt: str = ".2uL") -> str:
    """Raw siunitx-compatible core, e.g. ``(1.14 \\pm 0.10) \\times 10^{6}``."""
    return format(_as_ufloat(v), fmt)


def latex_value(v: Any, unit: str = "", *, fmt: str = ".2uL",
                with_rel: bool = True) -> str:
    """Dollar-wrapped siunitx value, optional ``\\,\\mathrm{unit}``, optional ``(rel %)``.

    Accepts ``PhysicalSize``, ``ufloat``, ``(value, sigma)`` tuple, or plain
    number.  Designed to drop straight into a marimo ``mo.md()`` call.
    """
    core = format_ufloat(v, fmt=fmt)
    umod = rf"\,\mathrm{{{unit}}}" if unit else ""
    if not with_rel:
        return f"${core}{umod}$"
    return f"${core}{umod}$ ({rel_pct(v):.2f} %)"


# --- Marimo widgets ---------------------------------------------------------
def fit_callout(result: FitResult, *, title: str = "Fit result",
                kind: str = "info",
                param_units: dict[str, str] | None = None,
                extra_lines: Sequence[str] = ()):
    """``mo.callout`` showing fit parameters, chi^2/dof, and p-value."""
    import marimo as mo
    param_units = param_units or {}
    lines = [f"**{title}**", ""]
    for i, name in enumerate(result.param_names):
        unit = param_units.get(name, "")
        lines.append(f"* ${name}$ = {latex_value(result.param(i), unit)}")
    lines.extend(extra_lines)
    lines.append("")
    lines.append(
        f"$\\chi^2/\\nu = {result.redchi:.2f}$  "
        f"({result.chi2:.2f} / {result.dof}),  "
        f"$p = {result.p_value:.3f}$"
    )
    lines.append(f"*{result.quality()}*")
    return mo.callout(mo.md("\n".join(lines)), kind=kind)


def params_table(result: FitResult, *, units: dict[str, str] | None = None,
                 show_correlation: bool = False):
    """Markdown table of fit parameters with siunitx-style cells."""
    import marimo as mo
    units = units or {}
    head = "| Parameter | Value | Relative |"
    sep = "|---|---|---|"
    rows = []
    for i, name in enumerate(result.param_names):
        p = result.param(i)
        u = units.get(name, "")
        rows.append(
            f"| ${name}$ | {latex_value(p, u, with_rel=False)} "
            f"| {rel_pct(p):.2f} % |"
        )
    parts = [mo.md("\n".join([head, sep, *rows]))]
    if show_correlation and result.cov is not None:
        corr = result.correlation_matrix()
        if corr is not None:
            n = len(result.param_names)
            head_c = "| | " + " | ".join(f"${x}$" for x in result.param_names) + " |"
            sep_c = "|---" * (n + 1) + "|"
            rows_c = []
            for i, name in enumerate(result.param_names):
                cells = " | ".join(f"{corr[i, j]:+.2f}" for j in range(n))
                rows_c.append(f"| ${name}$ | {cells} |")
            parts.append(mo.md("**Correlation matrix**\n\n"
                               + "\n".join([head_c, sep_c, *rows_c])))
    return mo.vstack(parts) if len(parts) > 1 else parts[0]


def results_table(rows: Iterable[tuple[str, Any, str]]):
    """Markdown table of ``(label, value, unit)`` triples.

    Each row renders as a siunitx value with a relative-uncertainty column.
    Use for a headline results block that summarises several derived
    quantities in one place.
    """
    import marimo as mo
    head = "| Quantity | Value | Relative |"
    sep = "|---|---|---|"
    lines = [head, sep]
    for label, v, unit in rows:
        lines.append(
            f"| {label} | {latex_value(v, unit, with_rel=False)} "
            f"| {rel_pct(v):.2f} % |"
        )
    return mo.md("\n".join(lines))


def uncertainty_budget(entries: Iterable[tuple[str, Any, str, str]]):
    """Markdown uncertainty-budget table.

    Rows are ``(quantity, value, relative_note, origin)``.  ``value`` may be a
    PhysicalSize, ufloat, ``(value, sigma)`` pair, or a raw LaTeX string if
    you want full control.  ``relative_note`` and ``origin`` are passed
    through verbatim (they are meant to be hand-authored — e.g.
    ``"0.17 %"`` and ``"ruler, 1 mm resolution"``).
    """
    import marimo as mo
    head = "| Quantity | Value | Relative | Origin |"
    sep = "|---|---|---|---|"
    lines = [head, sep]
    for quantity, v, relative, origin in entries:
        if isinstance(v, str):
            value_cell = v
        else:
            value_cell = latex_value(v, "", with_rel=False)
        lines.append(f"| {quantity} | {value_cell} | {relative} | {origin} |")
    return mo.md("\n".join(lines))


def fit_metrics_accordion(
    results: dict[str, FitResult],
    *,
    formatter: Callable[[FitResult], str] | None = None,
    extra_lines: Callable[[FitResult], Sequence[str]] | None = None,
):
    """``mo.accordion`` mapping run-name -> its metrics / params report.

    ``formatter`` overrides the default ``FitResult.summary()`` body.
    ``extra_lines`` returns additional markdown lines appended below the
    summary — useful for per-run derived quantities like ``mu0_exp``.
    """
    import marimo as mo
    fmt = formatter or (lambda r: r.summary())
    out = {}
    for name, r in results.items():
        body = "```\n" + fmt(r) + "\n```"
        if extra_lines is not None:
            extra = "\n".join(extra_lines(r))
            if extra:
                body = body + "\n\n" + extra
        header = (f"{name}  —  chi^2/nu = {r.redchi:.2f},  "
                  f"p = {r.p_value:.2f}")
        out[header] = mo.md(body)
    return mo.accordion(out)


# --- Compact inline / callout helpers ---------------------------------------
def nsigma_inline(v1: Any, v2: Any, *, ref_label: str = "theory") -> str:
    """Inline markdown: ``$n_\\sigma$ vs theory = 2.3``.

    Pair with :func:`latex_value` when embedding inside a sentence.
    """
    return f"$n_\\sigma$ vs {ref_label} = {_nsigma(v1, v2):.1f}"


def chi_squared_badge(result: FitResult, *, p_threshold: float = 0.05,
                      chi_max: float = 2.0, chi_min: float = 0.5) -> str:
    """One-line markdown pass/warn/fail badge for a fit's goodness of fit.

    * **PASS** when ``chi_min <= chi^2/dof <= chi_max`` and ``p >= p_threshold``.
    * **WARN** when only one of those holds (e.g. p is low but chi^2 is fine).
    * **FAIL** otherwise.

    Rendered as bold markdown suitable for inline use inside an ``mo.md`` cell.
    """
    redchi, p = result.redchi, result.p_value
    chi_ok = chi_min <= redchi <= chi_max
    p_ok = p >= p_threshold
    if chi_ok and p_ok:
        tag = "**PASS**"
    elif chi_ok or p_ok:
        tag = "**WARN**"
    else:
        tag = "**FAIL**"
    return f"{tag}  $\\chi^2/\\nu = {redchi:.2f}$, $p = {p:.3f}$"


def result_comparison_table(rows: Iterable[tuple[str, Any, Any, str]]):
    """Markdown table with a theoretical column and an automatic ``n_sigma``.

    ``rows`` is an iterable of ``(label, measured, theoretical, unit)``; each
    of ``measured`` / ``theoretical`` may be a :class:`PhysicalSize`,
    ``ufloat``, ``(value, sigma)`` pair, or plain number.
    """
    import marimo as mo
    head = "| Quantity | Measured | Theoretical | $n_\\sigma$ |"
    sep = "|---|---|---|---|"
    lines = [head, sep]
    for label, measured, theoretical, unit in rows:
        n = _nsigma(measured, theoretical)
        lines.append(
            f"| {label} "
            f"| {latex_value(measured, unit, with_rel=False)} "
            f"| {latex_value(theoretical, unit, with_rel=False)} "
            f"| ${n:.1f}$ |"
        )
    return mo.md("\n".join(lines))


def statistical_assessment_prose(result: FitResult,
                                 *,
                                 nsigma_ref: Any = None,
                                 ref_label: str = "theory",
                                 param_index: int = 0) -> str:
    """One-paragraph English assessment, matching the tone of past lab reports.

    Describes the goodness-of-fit (from ``chi^2/nu`` and ``p``) and, if
    ``nsigma_ref`` is given, compares the named parameter (default: the first)
    to that reference value in sigma units.  Returns raw markdown suitable
    for embedding in an ``mo.md`` cell or an ``mo.callout``.
    """
    r, p = result.redchi, result.p_value
    if r < 0.5:
        gof = "suggests the uncertainties may be overestimated"
    elif r > 2.0:
        gof = "suggests unmodeled scatter or underestimated uncertainties"
    else:
        gof = "is consistent with the estimated uncertainties"
    lead = (f"The fit yields $\\chi^2/\\nu = {r:.2f}$ "
            f"($p = {p:.3f}$), which {gof}.")
    if nsigma_ref is None:
        return lead
    p_val = result.param(param_index)
    n = _nsigma(p_val, nsigma_ref)
    if n < 2.0:
        verdict = "agrees with"
    elif n < 3.0:
        verdict = "is marginally consistent with"
    else:
        verdict = "differs from"
    return (
        f"{lead}  The best-fit value "
        f"{latex_value(p_val, with_rel=False)} {verdict} the {ref_label} "
        f"value at ${n:.1f}\\,\\sigma$."
    )


# --- LaTeX preamble reused across past reports ------------------------------
SIUNITX_PREAMBLE = r"""\sisetup{
    per-mode                 = fraction,
    bracket-unit-denominator = true,
    separate-uncertainty     = true,
    multi-part-units         = single,
}"""


def siunitx_preamble() -> str:
    """Return the shared ``\\sisetup{...}`` block used across past lab reports.

    Convenient for generating report-ready LaTeX fragments directly from a
    marimo notebook when exporting results.
    """
    return SIUNITX_PREAMBLE
