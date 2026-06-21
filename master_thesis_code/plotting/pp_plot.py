"""Bilby-style PP-plot / coverage figure (fig24, VR-NEW-02).

The calibration proof referees treat as non-negotiable for a simulation-based
inference thesis (viz-redesign proposal §5.2): a *probability-probability* (PP)
plot showing that the inferred credible levels are well-calibrated, i.e. that an
``X%`` credible interval contains the truth for ``X%`` of the injections. A
well-calibrated pipeline traces the diagonal; nested grey 1/2/3-sigma binomial
confidence bands (the bilby construction) bound the expected statistical scatter,
and a per-parameter Kolmogorov-Smirnov p-value quantifies departure from
uniformity.

The ranks plotted here are SYNTHETIC (calibrated uniform) — a scaffold standing
in for a real injection-recovery campaign — behind a single, clearly-commented
DATA GATE (``DEFAULT_PP_PARAMS`` + ``load_pp_ranks``): once a real campaign
produces per-parameter credible-level ranks, finalization is a one-line swap
(point the loader at ``<data_dir>/injection_recovery/ranks.json``).

No physics: the bands are a binomial-quantile construction and the curves are
empirical CDFs of credible levels. ``scipy.stats`` (already a core dependency)
supplies ``binom`` and ``kstest``.

All functions follow the project convention: data in, ``(fig, ax)`` out.
None call ``plt.show()`` or ``plt.savefig()``.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import binom, kstest

from master_thesis_code.plotting._colors import CYCLE, REFERENCE
from master_thesis_code.plotting._helpers import _fig_from_ax, get_figure


def binomial_confidence_bands(
    n_injections: int,
    confidence_levels: tuple[float, ...] = (0.68, 0.95, 0.997),
    n_grid: int = 101,
) -> dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Nested binomial confidence bands for a PP-plot (bilby construction).

    At each expected fraction ``x`` along the [0, 1] credible-level grid, the band
    for confidence level ``cl`` is the ``binom(n, x)`` quantile interval scaled by
    ``1 / n`` — i.e. the range of empirical fractions consistent with a perfectly
    calibrated pipeline at that point. ``(0.68, 0.95, 0.997)`` give the nested
    1/2/3-sigma bands.

    Parameters
    ----------
    n_injections:
        Number of injections ``n`` (sets the binomial sample size).
    confidence_levels:
        Confidence levels for the bands (default 1/2/3-sigma).
    n_grid:
        Number of grid points over [0, 1] (default 101).

    Returns
    -------
    dict[float, tuple[NDArray, NDArray]]
        ``{cl: (lower, upper)}`` envelopes, each of length ``n_grid``.

    References
    ----------
    bilby ``bilby.core.result.make_pp_plot`` confidence-band construction —
    Romero-Shaw et al. 2020, MNRAS 499 3295, arXiv:2006.00714.
    """
    x = np.linspace(0.0, 1.0, n_grid)
    bands: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    n = max(int(n_injections), 1)
    for cl in confidence_levels:
        edge = (1.0 - cl) / 2.0
        lower = np.asarray(binom.ppf(edge, n, x), dtype=np.float64) / n
        upper = np.asarray(binom.ppf(1.0 - edge, n, x), dtype=np.float64) / n
        bands[cl] = (lower, upper)
    return bands


def make_synthetic_ranks(
    n_injections: int,
    param_names: Sequence[str],
    *,
    calibrated: bool = True,
    seed: int = 0,
) -> dict[str, npt.NDArray[np.float64]]:
    """SYNTHETIC per-parameter credible-level ranks for the PP-plot scaffold.

    A well-calibrated pipeline produces credible levels uniform on [0, 1]; a
    mis-calibrated one produces a skewed distribution whose empirical CDF bows
    away from the diagonal. This is a SCAFFOLD — real injection-recovery ranks
    replace it via :func:`load_pp_ranks`.

    Parameters
    ----------
    n_injections:
        Number of injections per parameter.
    param_names:
        Parameter names (one rank array per name).
    calibrated:
        ``True`` (default) → uniform ranks (well-calibrated). ``False`` →
        Beta(2, 5)-skewed ranks (mis-calibrated; for the sanity test).
    seed:
        Base RNG seed (each parameter offsets it so the curves differ).

    Returns
    -------
    dict[str, NDArray[np.float64]]
        ``{param: ranks}`` with each ``ranks`` array in [0, 1].
    """
    n = max(int(n_injections), 1)
    ranks: dict[str, npt.NDArray[np.float64]] = {}
    for i, name in enumerate(param_names):
        rng = np.random.default_rng(seed + i)
        if calibrated:
            values = rng.uniform(0.0, 1.0, n)
        else:
            # Skewed away from uniform so the empirical CDF bows off the diagonal.
            values = rng.beta(2.0, 5.0, n)
        ranks[name] = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return ranks


# ============================ DATA GATE (VR-NEW-02) ============================
# PP-plot ranks are SYNTHETIC (calibrated uniform) until a real injection-recovery
# campaign produces per-parameter credible-level ranks. FINALIZE: point
# load_pp_ranks at the real ranks file (expected:
# <data_dir>/injection_recovery/ranks.json mapping param -> list[float] in [0,1])
# — a ONE-LINE swap. See .planning/ROADMAP.md Phase 4 success criterion 4.
DEFAULT_PP_PARAMS: tuple[str, ...] = ("M", "mu", "a", "p0", "e0", "d_L", "qS", "phiS")


def load_pp_ranks(
    data_dir: Path | None = None,
    *,
    n_injections: int = 200,
) -> dict[str, npt.NDArray[np.float64]]:
    """Return per-parameter PP-plot ranks (data-gated).

    The DATA GATE auto-closes: when ``<data_dir>/injection_recovery/ranks.json``
    exists and parses to ``{param: list[float]}``, those real ranks are loaded
    (coerced to ``np.float64`` and clipped to [0, 1], T-04-02). On any
    missing/malformed input the synthetic calibrated ranks over
    :data:`DEFAULT_PP_PARAMS` are returned — so the figure always renders.

    Parameters
    ----------
    data_dir:
        Directory holding ``injection_recovery/ranks.json``. When ``None`` the
        synthetic ranks are returned directly.
    n_injections:
        Number of synthetic injections when falling back.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        ``{param: ranks}`` with each array in [0, 1].
    """
    if data_dir is not None:
        ranks_path = data_dir / "injection_recovery" / "ranks.json"
        if ranks_path.is_file():
            try:
                with open(ranks_path) as f:
                    raw = json.load(f)
                if isinstance(raw, dict) and raw:
                    parsed: dict[str, npt.NDArray[np.float64]] = {}
                    for key, vals in raw.items():
                        arr = np.asarray(vals, dtype=np.float64)
                        if arr.ndim != 1 or arr.size == 0:
                            raise ValueError(f"malformed ranks for {key}")
                        parsed[str(key)] = np.clip(arr, 0.0, 1.0)
                    return parsed
            except (ValueError, TypeError, OSError, json.JSONDecodeError):
                # Fall through to synthetic rather than plot garbage (T-04-02).
                pass
    return make_synthetic_ranks(n_injections, DEFAULT_PP_PARAMS, calibrated=True)


def plot_pp_coverage(
    ranks: dict[str, npt.NDArray[np.float64]],
    *,
    n_injections: int | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Bilby-style PP-plot / coverage figure (fig24, VR-NEW-02).

    Draws square [0, 1]^2 axes with nested grey 1/2/3-sigma binomial confidence
    bands (darkest innermost), the calibration diagonal, one cumulative
    empirical-CDF line per parameter (CYCLE colors), and per-parameter +
    combined Kolmogorov-Smirnov p-values (vs uniform) in the legend.

    Parameters
    ----------
    ranks:
        ``{param: credible_levels}`` with each array of credible levels in
        [0, 1] (one per injection).
    n_injections:
        Number of injections; inferred from the rank arrays when ``None``.
    ax:
        Optional pre-existing Axes. When ``None`` a REVTeX single-column figure
        is created via :func:`get_figure` (no hardcoded figsize).

    Returns
    -------
    tuple[Figure, Axes]
        The figure and the populated PP-plot Axes.

    References
    ----------
    - PP-plot / binomial bands: bilby ``make_pp_plot`` — Romero-Shaw et al. 2020,
      MNRAS 499 3295, arXiv:2006.00714.
    - Kolmogorov-Smirnov uniformity test: ``scipy.stats.kstest`` against the
      ``"uniform"`` reference distribution.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    if n_injections is None:
        sizes = [int(np.asarray(v).size) for v in ranks.values()]
        n_injections = max(sizes) if sizes else 1

    x = np.linspace(0.0, 1.0, 101)
    bands = binomial_confidence_bands(n_injections, n_grid=101)

    # Nested grey bands: paint widest (3-sigma, lightest) first so the innermost
    # (1-sigma, darkest) lands on top. Greys are grayscale-native by construction.
    band_greys: dict[float, str] = {0.997: "0.85", 0.95: "0.7", 0.68: "0.55"}
    for cl in (0.997, 0.95, 0.68):
        lower, upper = bands[cl]
        ax.fill_between(x, lower, upper, color=band_greys[cl], alpha=1.0, zorder=0)

    # Calibration diagonal.
    ax.plot([0.0, 1.0], [0.0, 1.0], color=REFERENCE, linestyle="--", linewidth=0.8, zorder=1)

    # Per-parameter empirical CDF of credible levels + KS p-value.
    pooled: list[float] = []
    for i, (name, values) in enumerate(ranks.items()):
        v = np.sort(np.asarray(values, dtype=np.float64))
        n = v.size
        if n == 0:
            continue
        y = np.arange(1, n + 1, dtype=np.float64) / n
        color = CYCLE[i % len(CYCLE)]
        p = float(kstest(v, "uniform").pvalue)
        ax.plot(v, y, color=color, linewidth=1.2, label=f"{name} (p={p:.2f})", zorder=2)
        pooled.extend(v.tolist())

    # Combined KS p across all parameters pooled.
    if pooled:
        p_all = float(kstest(np.asarray(pooled, dtype=np.float64), "uniform").pvalue)
        legend_title = f"combined p={p_all:.2f}"
    else:
        legend_title = None

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("credible level")
    ax.set_ylabel("fraction of injections")
    ax.legend(loc="upper left", fontsize="x-small", title=legend_title)

    # No fig.tight_layout: constrained_layout (project mplstyle) owns packing.
    return fig, ax
