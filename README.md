# taulab

Physics-lab helpers for TAU coursework, built for [marimo] notebooks.

A rewrite of [`itayf-tau/taulab`][upstream] with uncertainty-propagating
arithmetic, a stock instrument-uncertainty library (Keysight 34401A,
DSO7012A, ruler/caliper), auto-seeded ODR, confidence-band plotting, and
marimo-native display helpers.

## Install

```bash
pip install taulab              # core
pip install taulab[marimo]      # + marimo display helpers
```

## Quick example

```python
import numpy as np
from taulab import fit_functions, odr_fit, Graph

x  = np.linspace(0, 1, 10)
y  = 2 * x + 1 + 0.02 * np.random.default_rng(0).standard_normal(10)
sx = np.full_like(x, 0.01)
sy = np.full_like(y, 0.02)

# Auto-seeded linear ODR — no init_values needed
res = odr_fit(fit_functions.linear, None, x, sx, y, sy,
              param_names=["intercept", "slope"])
print(res)                                              # metrics + params

fig, _ = Graph(res).plot_with_residuals(show_band=True)
```

In a marimo cell:

```python
from taulab.display import fit_callout, params_table
mo.vstack([fit_callout(res), params_table(res, show_correlation=True)])
```

## Highlights over upstream

- **Uncertainty arithmetic on `PhysicalSize`** via the `uncertainties` package.
- **`stats`** — `resolution_sigma`, `sem`, `combine`, `weighted_mean`, `nsigma`, `drop_outliers`.
- **`instruments`** — Keysight 34401A (resistance + DCV, auto-ranged), DSO7012A dual-cursor, ruler/caliper/micrometer, `reading_lsd`, `column_lsd`.
- **ODR auto-seeded** for linear / polynomial / constant fits.
- **`FitResult`** — `cov`, `correlation_matrix`, `pulls`, `quality`, `extrapolate`, `params_dict`.
- **`Graph.plot_with_residuals`** with a 1-σ confidence band from the parameter covariance.
- **`display`** — siunitx LaTeX formatters, `fit_callout`, `params_table`, `results_table`, `uncertainty_budget`, `fit_metrics_accordion`, `chi_squared_badge`, `statistical_assessment_prose`.

Full upstream import paths are preserved:
`from taulab.fit import odr_fit, fit_functions` and
`from taulab.parse import parse_csv_folder` both work unchanged.

## License

MIT — see [LICENSE](LICENSE).

[marimo]: https://marimo.io/
[upstream]: https://github.com/itayf-tau/taulab
