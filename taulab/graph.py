"""Plotting helpers — ``Graph`` wraps a ``FitResult`` with data / fit / residuals
painters and an optional uncertainty confidence band computed from the parameter
covariance.
"""
from __future__ import annotations

import numpy as np

from .fits import FitResult

__all__ = ["Graph"]


class Graph:
    """Painter for a ``FitResult``.

    The three methods mirror how past lab notebooks draw their fits:

    * ``plot(ax)`` — data with error bars + fit line + optional uncertainty band.
    * ``residuals_plot(ax)`` — residuals with ``y_err`` bars and a zero line.
    * ``plot_with_residuals()`` — a ready-made two-panel figure combining
      both; returns ``(fig, (ax_fit, ax_res))`` for further customisation.
    """

    def __init__(self, fit_result: FitResult):
        self.r = fit_result

    def plot(self, ax, *, data_label: str | None = None,
             fit_label: str | None = None,
             color: str = "C0", fit_color: str = "red", marker: str = "o",
             show_band: bool = False, band_alpha: float = 0.2,
             n_grid: int = 200, extrapolate: float = 0.0):
        """Draw data + fit on ``ax``.

        ``extrapolate`` extends the fit curve by that fraction of the data
        span on either side (e.g. ``0.05`` for a 5 % overshoot).
        """
        r = self.r
        ax.errorbar(r.x, r.y, xerr=r.sx, yerr=r.sy,
                    fmt=marker, color=color, mfc="white",
                    markersize=5, ecolor=color, elinewidth=0.9, capsize=2.5,
                    label=data_label)
        span = float(r.x.max() - r.x.min())
        pad = extrapolate * span
        xs = np.linspace(r.x.min() - pad, r.x.max() + pad, n_grid)
        yhat = r.predict(xs)
        ax.plot(xs, yhat, "-", color=fit_color, alpha=0.85, label=fit_label)
        if show_band:
            band = self._band(xs)
            if np.any(band > 0):
                ax.fill_between(xs, yhat - band, yhat + band,
                                color=fit_color, alpha=band_alpha,
                                label=r"1-$\sigma$ band")
        if data_label or fit_label:
            ax.legend()

    def residuals_plot(self, ax, *, color: str = "C0", marker: str = "o",
                       use_pulls: bool = False):
        """Draw residuals (or pulls, if ``use_pulls=True``) on ``ax``."""
        r = self.r
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        if use_pulls and r.sy is not None:
            ax.errorbar(r.x, r.pulls(), yerr=np.ones_like(r.x),
                        fmt=marker, color=color, mfc="white",
                        markersize=5, ecolor=color, elinewidth=0.9, capsize=2.5)
            ax.set_ylabel("pull  (res / $\\sigma_y$)")
        else:
            ax.errorbar(r.x, r.residuals(), yerr=r.sy,
                        fmt=marker, color=color, mfc="white",
                        markersize=5, ecolor=color, elinewidth=0.9, capsize=2.5)

    def plot_with_residuals(self, *, figsize: tuple[float, float] = (8.4, 6.2),
                            color: str = "C0", fit_color: str = "red",
                            marker: str = "o",
                            data_label: str | None = None,
                            fit_label: str | None = None,
                            show_band: bool = False,
                            height_ratio: tuple[float, float] = (2.4, 1.0),
                            use_pulls: bool = False):
        """Build a two-panel figure with the fit up top and residuals below."""
        import matplotlib.pyplot as plt
        fig, (ax_fit, ax_res) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, constrained_layout=True,
            gridspec_kw={"height_ratios": list(height_ratio)},
        )
        self.plot(ax_fit, color=color, fit_color=fit_color, marker=marker,
                  data_label=data_label, fit_label=fit_label, show_band=show_band)
        self.residuals_plot(ax_res, color=color, marker=marker, use_pulls=use_pulls)
        if not use_pulls:
            ax_res.set_ylabel("residual")
        return fig, (ax_fit, ax_res)

    # ---- internal -------------------------------------------------------
    def _band(self, xs: np.ndarray) -> np.ndarray:
        """uncertainty band from parameter covariance, via numerical gradients."""
        r = self.r
        if r.cov is None:
            return np.zeros_like(xs)
        eps = 1e-6 * (np.abs(r.params) + 1.0)
        grads = []
        for i in range(len(r.params)):
            p_plus = r.params.copy();  p_plus[i]  += eps[i]
            p_minus = r.params.copy(); p_minus[i] -= eps[i]
            grads.append(
                (r._fit_func(p_plus, xs) - r._fit_func(p_minus, xs)) / (2 * eps[i])
            )
        G = np.asarray(grads)                                # (n_params, n_grid)
        var = np.einsum("in,ij,jn->n", G, r.cov, G)
        return np.sqrt(np.maximum(var, 0.0))
