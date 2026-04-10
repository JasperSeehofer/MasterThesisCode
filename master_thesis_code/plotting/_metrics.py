"""Pure-function constraint-quality metrics for H0 posteriors.

Metrics used by the M_z improvement analysis (see
``plotting/convergence_analysis.py``).  All functions consume a 1-D
posterior on a fixed h-grid and return scalars; none of them mutate
their inputs or perform IO.

Literature grounding:

* HDI / minimal credible interval — LIGO/Virgo H0 reporting convention
  (Abbott et al. 2017, GW170817 H0).
* Information gain via Kullback–Leibler divergence — Ashton et al. 2019,
  *Bayesian inference in gravitational-wave astronomy*, arXiv:1809.02293,
  Sec. 5.
* Jensen–Shannon divergence with the 50 mbits "agreement" rule of thumb
  for posterior comparison — same reference, Sec. 5.
* 1/sqrt(N) figure of merit and "effective event gain" framing — standard
  in dark-siren H0 papers (e.g. Alfradique et al. 2024, MNRAS 528, 3249).
"""

import numpy as np
import numpy.typing as npt

from master_thesis_code.plotting._helpers import compute_hdi_interval

# ---------------------------------------------------------------------------
# Single-posterior metrics
# ---------------------------------------------------------------------------


def hdi_width(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    level: float = 0.683,
) -> float:
    """Width of the highest-density credible interval at *level*.

    Returns ``nan`` for an unnormalizable posterior (zero/negative mass).
    """
    lo, hi = compute_hdi_interval(h_values, posterior, level=level)
    if np.isnan(lo) or np.isnan(hi):
        return float("nan")
    return float(hi - lo)


def map_h(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
) -> float:
    """Maximum-a-posteriori h.  Returns ``nan`` if all densities are zero."""
    if not np.any(posterior > 0):
        return float("nan")
    return float(h_values[int(np.argmax(posterior))])


def rel_precision(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    level: float = 0.683,
) -> float:
    """Fractional precision ``HDI_width / MAP_h``.

    The headline number reported in dark-siren H0 papers ("we measure H0
    to X percent").  Returns ``nan`` if the MAP is at zero density or
    the HDI is undefined.
    """
    width = hdi_width(h_values, posterior, level=level)
    m = map_h(h_values, posterior)
    if np.isnan(width) or np.isnan(m) or m == 0:
        return float("nan")
    return float(width / m)


def bias_pct(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    h_true: float,
) -> float:
    """Percent bias of the MAP relative to the injected truth ``h_true``."""
    m = map_h(h_values, posterior)
    if np.isnan(m) or h_true == 0:
        return float("nan")
    return float((m - h_true) / h_true * 100.0)


def kl_from_uniform(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
) -> float:
    """KL divergence (in nats) of the posterior from a flat prior on the grid.

    Information-theoretic figure of merit: monotonic in "how constraining"
    the posterior is.  Equivalent to the information gain from the data
    when the prior is uniform on ``[h_values[0], h_values[-1]]``.

    Definition::

        KL(p || u) = ∫ p(h) log(p(h) / u(h)) dh
                   = ∫ p(h) log(p(h)) dh + log(h_max - h_min)

    Returns ``0`` for a perfectly flat posterior, ``nan`` for an
    unnormalizable one.
    """
    norm = np.trapezoid(posterior, h_values)
    if norm <= 0:
        return float("nan")
    p = posterior / norm
    h_range = float(h_values[-1] - h_values[0])
    if h_range <= 0:
        return float("nan")
    # log(p) is undefined where p == 0; mask those cells out (they
    # contribute 0 in the limit p log p -> 0).
    integrand = np.zeros_like(p)
    nonzero = p > 0
    integrand[nonzero] = p[nonzero] * np.log(p[nonzero] * h_range)
    return float(np.trapezoid(integrand, h_values))


# ---------------------------------------------------------------------------
# Two-posterior comparison metrics
# ---------------------------------------------------------------------------


def jsd_between(
    h_values: npt.NDArray[np.float64],
    posterior_a: npt.NDArray[np.float64],
    posterior_b: npt.NDArray[np.float64],
) -> float:
    """Jensen–Shannon divergence between two posteriors on the same grid.

    Returned in **bits** (base-2 logarithm) so the GW-community
    "< 50 mbits = good agreement" rule of thumb (Ashton et al. 2019)
    applies directly.

    Always lies in ``[0, 1]`` bits; zero iff the two posteriors are
    identical.  Returns ``nan`` if either posterior fails to normalize.
    """
    norm_a = np.trapezoid(posterior_a, h_values)
    norm_b = np.trapezoid(posterior_b, h_values)
    if norm_a <= 0 or norm_b <= 0:
        return float("nan")
    pa = posterior_a / norm_a
    pb = posterior_b / norm_b
    m = 0.5 * (pa + pb)

    def _kl(p: npt.NDArray[np.float64], q: npt.NDArray[np.float64]) -> float:
        mask = (p > 0) & (q > 0)
        integrand = np.zeros_like(p)
        integrand[mask] = p[mask] * np.log2(p[mask] / q[mask])
        return float(np.trapezoid(integrand, h_values))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


# ---------------------------------------------------------------------------
# Curve-level summaries
# ---------------------------------------------------------------------------


def effective_event_gain(
    sizes_ref: npt.NDArray[np.float64],
    widths_ref: npt.NDArray[np.float64],
    sizes_query: npt.NDArray[np.float64],
    widths_query: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """How many *reference* events match each *query* width?

    For each ``(N_q, w_q)`` pair in the query curve, find the reference
    event count ``N_r`` whose width equals ``w_q`` by monotone-decreasing
    interpolation in ``log(N)``.  Returns ``K(N_q) = N_r / N_q`` — the
    "effective event gain" of the query analysis at that subset size.

    A return value > 1 means the query analysis at N events matches the
    precision of the reference analysis at K*N events (i.e. the query
    is *more* informative per event).

    Both curves must be sorted by N ascending.  Reference widths must be
    monotonically non-increasing in N (which holds for any 1/sqrt(N)-like
    convergence — small violations from bootstrap noise are tolerated by
    the interpolation).

    Parameters
    ----------
    sizes_ref, widths_ref:
        Reference curve (e.g. without-M_z).
    sizes_query, widths_query:
        Query curve (e.g. with-M_z).

    Returns
    -------
    npt.NDArray[np.float64]
        Array of K factors with the same shape as ``sizes_query``.
        ``nan`` where the query width falls outside the reference range
        (cannot interpolate).
    """
    if len(sizes_ref) < 2:
        return np.full_like(sizes_query, np.nan, dtype=np.float64)

    log_n_ref = np.log(np.asarray(sizes_ref, dtype=np.float64))
    w_ref = np.asarray(widths_ref, dtype=np.float64)

    # Drop any non-positive reference widths (cannot take log).
    mask_ref = w_ref > 0
    if mask_ref.sum() < 2:
        return np.full_like(sizes_query, np.nan, dtype=np.float64)
    log_w_ref = np.log(w_ref[mask_ref])
    log_n_ref = log_n_ref[mask_ref]

    # Interpolate in (log_w, log_n) space — for any power-law width
    # curve (e.g. 1/sqrt(N)) this is exact.  Sort by log_w ascending so
    # np.interp gets a monotone x-axis.
    order = np.argsort(log_w_ref)
    logw_sorted = log_w_ref[order]
    logn_sorted = log_n_ref[order]

    out = np.full_like(sizes_query, np.nan, dtype=np.float64)
    for i, (n_q, w_q) in enumerate(zip(sizes_query, widths_query, strict=True)):
        if np.isnan(w_q) or w_q <= 0:
            continue
        lw = float(np.log(w_q))
        if lw < logw_sorted[0] or lw > logw_sorted[-1]:
            continue
        log_n_match = float(np.interp(lw, logw_sorted, logn_sorted))
        out[i] = float(np.exp(log_n_match) / n_q)
    return out
