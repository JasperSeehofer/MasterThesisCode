"""M_z improvement analysis: bank computation, static figure, and IO.

This module answers the central scientific question
"Did adding the black-hole mass channel ``M_z`` to the H0 evaluation
improve the constraints, and by how much?" for varying numbers of
detections.  It provides:

* :func:`compute_m_z_improvement_bank` — paired-bootstrap aggregator
  over per-event posteriors that returns an :class:`ImprovementBank`
  dataclass with all metrics and a representative posterior at each
  subset size.
* :func:`plot_m_z_improvement_panels` — static three-panel matplotlib
  figure consuming an :class:`ImprovementBank` (used by ``main.py`` and
  available as a paper figure).
* The shared loaders ``_load_per_event_no_mass`` and
  ``_load_per_event_with_mass_scalars``, originally defined in
  ``paper_figures.py``.  They live here so that ``paper_figures`` and
  ``convergence_analysis`` do not import each other circularly; the old
  module re-exports them for back-compat.

The bank is cached on disk under
``<data_dir>/diagnostics/m_z_improvement_bank.json`` and reused on
subsequent calls — bootstrap recomputation is skipped when the cached
parameters (subset sizes, bootstrap count, seed, h_true) match.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import (
    REFERENCE,
    TRUTH,
    VARIANT_NO_MASS,
    VARIANT_WITH_MASS,
)
from master_thesis_code.plotting._helpers import get_figure
from master_thesis_code.plotting._metrics import (
    bias_pct,
    effective_event_gain,
    hdi_width,
    jsd_between,
    kl_from_uniform,
    map_h,
    rel_precision,
)

_LOGGER = logging.getLogger(__name__)

# Inherited from paper_figures._CONVERGENCE_SUBSET_SIZES so the new
# analysis lines up exactly with the legacy convergence plot.
DEFAULT_SUBSET_SIZES: list[int] = [10, 20, 50, 100, 150, 200, 300, 400, 500]
DEFAULT_BOOTSTRAP: int = 200
DEFAULT_SEED: int = 20260410


# ---------------------------------------------------------------------------
# Per-event loaders (moved from paper_figures.py to break circular import)
# ---------------------------------------------------------------------------


def _h_from_filename(fname: str) -> float:
    parts = fname.replace(".json", "").split("_")
    return float(parts[1] + "." + parts[2])


def _load_per_event_no_mass(
    base: Path,
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
    """Load all per-event likelihoods from the no-mass posteriors directory.

    Returns
    -------
    h_values : sorted array of h grid points
    events : dict mapping event-ID string to likelihood array (same order as h_values)
    """
    files = sorted([f for f in os.listdir(base) if f.startswith("h_") and f.endswith(".json")])

    raw: dict[float, dict[str, list[float]]] = {}
    for f in files:
        h = _h_from_filename(f)
        with open(base / f) as fh:
            raw[h] = json.load(fh)

    h_sorted = sorted(raw.keys())
    h_arr = np.array(h_sorted)

    event_ids = sorted(
        [k for k in raw[h_sorted[0]] if k.isdigit()],
        key=int,
    )

    events: dict[str, npt.NDArray[np.float64]] = {}
    for eid in event_ids:
        vals = []
        for h in h_sorted:
            v = raw[h].get(eid, [])
            vals.append(v[0] if v else 0.0)
        events[eid] = np.array(vals)

    return h_arr, events


def _load_per_event_with_mass_scalars(
    base: Path,
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
    """Load aggregated per-event likelihoods from with-mass posteriors.

    The with-BH-mass JSON files are large (~585 MB) because they contain
    per-galaxy breakdowns.  The aggregated scalar likelihoods are stored
    at the end of each file.  This function reads only the last 300 KB
    of each file and extracts the scalars with a regex, avoiding full
    JSON parsing.

    Returns
    -------
    h_values : sorted array of h grid points
    events : dict mapping event-ID string to likelihood array
    """
    files = sorted([f for f in os.listdir(base) if f.startswith("h_") and f.endswith(".json")])

    pattern = re.compile(r'"(\d+)": \[(\d+\.\d+(?:e[+-]?\d+)?)\]')

    raw: dict[float, dict[str, float]] = {}
    for f in files:
        h = _h_from_filename(f)
        filepath = base / f
        with open(filepath, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            read_size = min(300_000, size)
            fh.seek(-read_size, 2)
            tail = fh.read().decode("utf-8")
        matches = pattern.findall(tail)
        raw[h] = {k: float(v) for k, v in matches}

    h_sorted = sorted(raw.keys())
    h_arr = np.array(h_sorted)

    event_ids = sorted(raw[h_sorted[0]].keys(), key=int)
    events: dict[str, npt.NDArray[np.float64]] = {}
    for eid in event_ids:
        vals = [raw[h].get(eid, 0.0) for h in h_sorted]
        events[eid] = np.array(vals)

    return h_arr, events


# ---------------------------------------------------------------------------
# Bank dataclass + computation
# ---------------------------------------------------------------------------


# Names of metrics computed per (variant, N).  Stored in the bank as
# parallel arrays of (median, p16, p84) under each metric key.
_PER_VARIANT_METRICS: tuple[str, ...] = (
    "hdi68_width",
    "rel_precision",
    "kl_from_uniform",
    "map_h",
    "bias_pct",
)


@dataclass
class ImprovementBank:
    """Aggregated bootstrap results for the M_z improvement analysis.

    All array fields are aligned by index along *sizes*.

    Attributes
    ----------
    h_grid:
        Shared 1-D h-grid used by both variants (length ``n_h``).
    h_true:
        Injected truth value used for bias computation.
    sizes:
        Subset sizes (number of detections) actually probed (capped at
        the smaller of the two variant pools).
    n_bootstrap:
        Number of paired bootstrap draws per subset size.
    seed:
        Random seed used for the paired draws (reproducibility).
    metrics_no_mass, metrics_with_mass:
        Mappings ``metric_name -> {"median": [...], "p16": [...], "p84": [...]}``
        with one entry per element of ``sizes``.
    fractional_improvement:
        Paired-bootstrap distribution of
        ``(width_without − width_with) / width_without`` summarized as
        ``{"median": [...], "p16": [...], "p84": [...]}``.
    effective_event_gain:
        Median K(N) factor and 16/84 percentiles from the paired bootstrap
        (interpolated against the bootstrap-mean reference curve).
    jsd_bits:
        Jensen–Shannon divergence in bits between the per-bootstrap
        with-/without-M_z combined posteriors, summarized.
    representative_posteriors_no_mass, representative_posteriors_with_mass:
        For each subset size, the combined posterior of one bootstrap
        draw (the first) — used by panel (B) of the static figure and the
        Plotly slider frames.  Lists of length ``len(sizes)`` of arrays
        on ``h_grid``.
    n_events_no_mass, n_events_with_mass:
        Total valid events available in each variant pool (cap of
        ``sizes``).
    """

    h_grid: npt.NDArray[np.float64]
    h_true: float
    sizes: list[int]
    n_bootstrap: int
    seed: int
    metrics_no_mass: dict[str, dict[str, list[float]]]
    metrics_with_mass: dict[str, dict[str, list[float]]]
    fractional_improvement: dict[str, list[float]]
    effective_event_gain: dict[str, list[float]]
    jsd_bits: dict[str, list[float]]
    representative_posteriors_no_mass: list[npt.NDArray[np.float64]]
    representative_posteriors_with_mass: list[npt.NDArray[np.float64]]
    n_events_no_mass: int
    n_events_with_mass: int
    cache_meta: dict[str, Any] = field(default_factory=dict)

    def to_serializable(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dict (lists, no numpy)."""
        return {
            "h_grid": self.h_grid.tolist(),
            "h_true": float(self.h_true),
            "sizes": list(self.sizes),
            "n_bootstrap": int(self.n_bootstrap),
            "seed": int(self.seed),
            "metrics_no_mass": self.metrics_no_mass,
            "metrics_with_mass": self.metrics_with_mass,
            "fractional_improvement": self.fractional_improvement,
            "effective_event_gain": self.effective_event_gain,
            "jsd_bits": self.jsd_bits,
            "representative_posteriors_no_mass": [
                p.tolist() for p in self.representative_posteriors_no_mass
            ],
            "representative_posteriors_with_mass": [
                p.tolist() for p in self.representative_posteriors_with_mass
            ],
            "n_events_no_mass": int(self.n_events_no_mass),
            "n_events_with_mass": int(self.n_events_with_mass),
            "cache_meta": self.cache_meta,
        }

    @classmethod
    def from_serializable(cls, data: dict[str, Any]) -> ImprovementBank:
        return cls(
            h_grid=np.asarray(data["h_grid"], dtype=np.float64),
            h_true=float(data["h_true"]),
            sizes=[int(s) for s in data["sizes"]],
            n_bootstrap=int(data["n_bootstrap"]),
            seed=int(data["seed"]),
            metrics_no_mass=data["metrics_no_mass"],
            metrics_with_mass=data["metrics_with_mass"],
            fractional_improvement=data["fractional_improvement"],
            effective_event_gain=data["effective_event_gain"],
            jsd_bits=data["jsd_bits"],
            representative_posteriors_no_mass=[
                np.asarray(p, dtype=np.float64) for p in data["representative_posteriors_no_mass"]
            ],
            representative_posteriors_with_mass=[
                np.asarray(p, dtype=np.float64) for p in data["representative_posteriors_with_mass"]
            ],
            n_events_no_mass=int(data["n_events_no_mass"]),
            n_events_with_mass=int(data["n_events_with_mass"]),
            cache_meta=data.get("cache_meta", {}),
        )


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    """Return (median, p16, p84) of a list, propagating nan if empty."""
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (
        float(np.median(arr)),
        float(np.percentile(arr, 16)),
        float(np.percentile(arr, 84)),
    )


def _combine_log(
    log_event_matrix: npt.NDArray[np.float64],
    indices: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """Combine log-likelihoods of selected events into a normalized posterior."""
    log_combined = np.sum(log_event_matrix[indices, :], axis=0)
    log_combined -= np.max(log_combined)
    return np.asarray(np.exp(log_combined), dtype=np.float64)


def _build_event_matrix(
    h_values: npt.NDArray[np.float64],
    events: dict[str, npt.NDArray[np.float64]],
) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Stack per-event likelihoods into an (n_events, n_h) matrix.

    Drops events whose likelihood is zero everywhere.
    """
    event_ids = sorted(events.keys(), key=int)
    valid_ids = [eid for eid in event_ids if np.max(events[eid]) > 0]
    n_h = len(h_values)
    matrix = np.empty((len(valid_ids), n_h), dtype=np.float64)
    for i, eid in enumerate(valid_ids):
        matrix[i, :] = events[eid]
    log_matrix = np.log(np.clip(matrix, 1e-300, None))
    return valid_ids, matrix, log_matrix


def _cache_signature(
    sizes: list[int],
    n_bootstrap: int,
    seed: int,
    h_true: float,
    n_events_no_mass: int,
    n_events_with_mass: int,
) -> dict[str, Any]:
    return {
        "sizes": list(sizes),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "h_true": float(h_true),
        "n_events_no_mass": int(n_events_no_mass),
        "n_events_with_mass": int(n_events_with_mass),
    }


def compute_m_z_improvement_bank(
    data_dir: Path,
    *,
    subset_sizes: list[int] | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    h_true: float = 0.73,
    use_cache: bool = True,
    cache_path: Path | None = None,
) -> ImprovementBank | None:
    """Compute the paired-bootstrap M_z improvement bank.

    Loads per-event likelihoods from ``data_dir/posteriors/`` and
    ``data_dir/posteriors_with_bh_mass/``, runs ``n_bootstrap`` paired
    random subsets at each size in ``subset_sizes``, and aggregates all
    metrics defined in :mod:`master_thesis_code.plotting._metrics`.

    Parameters
    ----------
    data_dir:
        Working directory with ``posteriors/`` and
        ``posteriors_with_bh_mass/`` subdirectories.
    subset_sizes:
        Subset sizes to probe.  Defaults to :data:`DEFAULT_SUBSET_SIZES`.
    n_bootstrap:
        Number of paired bootstrap draws per subset size.
    seed:
        RNG seed (deterministic across runs).
    h_true:
        Injected truth — used for the ``bias_pct`` metric.
    use_cache:
        If True, look for ``data_dir/diagnostics/m_z_improvement_bank.json``;
        return cached bank when its parameter signature matches the
        request.
    cache_path:
        Override the default cache location.

    Returns
    -------
    ImprovementBank or ``None`` if either posterior directory is missing
    or contains no valid events.
    """
    sizes_in = list(subset_sizes) if subset_sizes is not None else list(DEFAULT_SUBSET_SIZES)

    no_dir = data_dir / "posteriors"
    with_dir = data_dir / "posteriors_with_bh_mass"
    if not no_dir.is_dir() or not with_dir.is_dir():
        _LOGGER.warning(
            "M_z improvement bank: missing %s or %s",
            no_dir,
            with_dir,
        )
        return None

    if cache_path is None:
        cache_path = data_dir / "diagnostics" / "m_z_improvement_bank.json"

    # --- Load per-event matrices ---
    h_no, events_no = _load_per_event_no_mass(no_dir)
    h_with, events_with = _load_per_event_with_mass_scalars(with_dir)

    if not np.allclose(h_no, h_with):
        _LOGGER.warning("h-grids of without- and with-M_z variants differ; using without-M_z grid")
    h_grid = h_no

    valid_no, _matrix_no, log_no = _build_event_matrix(h_no, events_no)
    valid_with, _matrix_with, log_with = _build_event_matrix(h_with, events_with)

    if len(valid_no) == 0 or len(valid_with) == 0:
        _LOGGER.warning("M_z improvement bank: no valid events in one of the variants")
        return None

    # Restrict to events present in both variants so the paired
    # bootstrap is well-defined.
    common_ids = sorted(set(valid_no) & set(valid_with), key=int)
    if len(common_ids) == 0:
        _LOGGER.warning("M_z improvement bank: no events common to both variants")
        return None

    common_idx_no = np.array([valid_no.index(eid) for eid in common_ids], dtype=np.intp)
    common_idx_with = np.array([valid_with.index(eid) for eid in common_ids], dtype=np.intp)
    log_no_common = log_no[common_idx_no, :]
    log_with_common = log_with[common_idx_with, :]
    n_common = len(common_ids)

    # Cap subset sizes
    used_sizes = [s for s in sizes_in if s <= n_common]
    if not used_sizes:
        _LOGGER.warning(
            "M_z improvement bank: no subset sizes <= n_common (%d)",
            n_common,
        )
        return None

    signature = _cache_signature(
        used_sizes,
        n_bootstrap,
        seed,
        h_true,
        n_events_no_mass=n_common,
        n_events_with_mass=n_common,
    )

    # --- Cache lookup ---
    if use_cache and cache_path.is_file():
        try:
            with open(cache_path) as fh:
                cached = json.load(fh)
            if cached.get("cache_meta", {}).get("signature") == signature:
                _LOGGER.info("M_z improvement bank: using cached %s", cache_path)
                return ImprovementBank.from_serializable(cached)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            _LOGGER.warning("M_z improvement bank: cache read failed (%s)", exc)

    # --- Bootstrap ---
    rng = np.random.default_rng(seed)
    metrics_no: dict[str, dict[str, list[float]]] = {
        m: {"median": [], "p16": [], "p84": []} for m in _PER_VARIANT_METRICS
    }
    metrics_with: dict[str, dict[str, list[float]]] = {
        m: {"median": [], "p16": [], "p84": []} for m in _PER_VARIANT_METRICS
    }
    frac_improvement: dict[str, list[float]] = {"median": [], "p16": [], "p84": []}
    jsd_summary: dict[str, list[float]] = {"median": [], "p16": [], "p84": []}

    rep_no: list[npt.NDArray[np.float64]] = []
    rep_with: list[npt.NDArray[np.float64]] = []

    # We collect the median width per N for each variant first; then
    # K(N) is computed once at the end against those medians.
    median_widths_no: list[float] = []
    median_widths_with: list[float] = []

    for n_sub in used_sizes:
        widths_no_sub: list[float] = []
        widths_with_sub: list[float] = []
        rel_no_sub: list[float] = []
        rel_with_sub: list[float] = []
        kl_no_sub: list[float] = []
        kl_with_sub: list[float] = []
        map_no_sub: list[float] = []
        map_with_sub: list[float] = []
        bias_no_sub: list[float] = []
        bias_with_sub: list[float] = []
        frac_sub: list[float] = []
        jsd_sub: list[float] = []

        for b in range(n_bootstrap):
            idx = rng.choice(n_common, size=n_sub, replace=False)
            p_no = _combine_log(log_no_common, idx)
            p_with = _combine_log(log_with_common, idx)

            w_no = hdi_width(h_grid, p_no)
            w_with = hdi_width(h_grid, p_with)
            widths_no_sub.append(w_no)
            widths_with_sub.append(w_with)
            rel_no_sub.append(rel_precision(h_grid, p_no))
            rel_with_sub.append(rel_precision(h_grid, p_with))
            kl_no_sub.append(kl_from_uniform(h_grid, p_no))
            kl_with_sub.append(kl_from_uniform(h_grid, p_with))
            map_no_sub.append(map_h(h_grid, p_no))
            map_with_sub.append(map_h(h_grid, p_with))
            bias_no_sub.append(bias_pct(h_grid, p_no, h_true))
            bias_with_sub.append(bias_pct(h_grid, p_with, h_true))
            jsd_sub.append(jsd_between(h_grid, p_no, p_with))
            if not (np.isnan(w_no) or np.isnan(w_with)) and w_no > 0:
                frac_sub.append((w_no - w_with) / w_no)
            else:
                frac_sub.append(float("nan"))

            # Capture the first bootstrap as the representative for
            # panel (B) — peak-normalize for display
            if b == 0:
                pn_no = p_no / (np.max(p_no) if np.max(p_no) > 0 else 1.0)
                pn_with = p_with / (np.max(p_with) if np.max(p_with) > 0 else 1.0)
                rep_no.append(pn_no)
                rep_with.append(pn_with)

        # Aggregate
        for name, vals_no, vals_with in (
            ("hdi68_width", widths_no_sub, widths_with_sub),
            ("rel_precision", rel_no_sub, rel_with_sub),
            ("kl_from_uniform", kl_no_sub, kl_with_sub),
            ("map_h", map_no_sub, map_with_sub),
            ("bias_pct", bias_no_sub, bias_with_sub),
        ):
            for store, vals in ((metrics_no, vals_no), (metrics_with, vals_with)):
                med, p16, p84 = _percentiles(vals)
                store[name]["median"].append(med)
                store[name]["p16"].append(p16)
                store[name]["p84"].append(p84)

        med_w_no = metrics_no["hdi68_width"]["median"][-1]
        med_w_with = metrics_with["hdi68_width"]["median"][-1]
        median_widths_no.append(med_w_no)
        median_widths_with.append(med_w_with)

        med_f, p16_f, p84_f = _percentiles(frac_sub)
        frac_improvement["median"].append(med_f)
        frac_improvement["p16"].append(p16_f)
        frac_improvement["p84"].append(p84_f)

        med_j, p16_j, p84_j = _percentiles(jsd_sub)
        jsd_summary["median"].append(med_j)
        jsd_summary["p16"].append(p16_j)
        jsd_summary["p84"].append(p84_j)

    # --- Effective event gain at each N (with-M_z vs without-M_z) ---
    sizes_arr = np.asarray(used_sizes, dtype=np.float64)
    widths_no_arr = np.asarray(median_widths_no, dtype=np.float64)
    widths_with_arr = np.asarray(median_widths_with, dtype=np.float64)
    K_median = effective_event_gain(sizes_arr, widths_no_arr, sizes_arr, widths_with_arr)
    # Lower/upper from the variant 16/84 envelopes (not a full bootstrap
    # of K, but a useful uncertainty band: K computed against the p16/p84
    # of the reference width).
    K_lo = effective_event_gain(
        sizes_arr,
        np.asarray(metrics_no["hdi68_width"]["p84"], dtype=np.float64),
        sizes_arr,
        np.asarray(metrics_with["hdi68_width"]["p16"], dtype=np.float64),
    )
    K_hi = effective_event_gain(
        sizes_arr,
        np.asarray(metrics_no["hdi68_width"]["p16"], dtype=np.float64),
        sizes_arr,
        np.asarray(metrics_with["hdi68_width"]["p84"], dtype=np.float64),
    )
    K_summary: dict[str, list[float]] = {
        "median": K_median.tolist(),
        "p16": K_lo.tolist(),
        "p84": K_hi.tolist(),
    }

    bank = ImprovementBank(
        h_grid=h_grid,
        h_true=float(h_true),
        sizes=used_sizes,
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        metrics_no_mass=metrics_no,
        metrics_with_mass=metrics_with,
        fractional_improvement=frac_improvement,
        effective_event_gain=K_summary,
        jsd_bits=jsd_summary,
        representative_posteriors_no_mass=rep_no,
        representative_posteriors_with_mass=rep_with,
        n_events_no_mass=n_common,
        n_events_with_mass=n_common,
        cache_meta={"signature": signature},
    )

    # --- Persist cache ---
    if use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as fh:
                json.dump(bank.to_serializable(), fh, indent=2)
            _LOGGER.info("M_z improvement bank: cached to %s", cache_path)
        except OSError as exc:
            _LOGGER.warning("M_z improvement bank: cache write failed (%s)", exc)

    return bank


# ---------------------------------------------------------------------------
# Static three-panel matplotlib figure
# ---------------------------------------------------------------------------


def plot_m_z_improvement_panels(
    bank: ImprovementBank,
) -> tuple[Figure, Any]:
    """Static three-panel figure of the M_z improvement analysis.

    Layout:

    * Top-left:  HDI 68% width vs N for both variants, with bootstrap
                 16–84 percentile band and a 1/sqrt(N) reference line.
    * Top-right: Representative combined posterior at the largest N for
                 both variants.
    * Bottom:    Fractional tightening Δ(N) and effective event gain
                 K(N), shared x-axis with the top-left panel.

    Parameters
    ----------
    bank:
        Result of :func:`compute_m_z_improvement_bank`.

    Returns
    -------
    (fig, axes) following the project factory convention.
    """
    sizes = np.asarray(bank.sizes, dtype=np.float64)
    w_no_med = np.asarray(bank.metrics_no_mass["hdi68_width"]["median"], dtype=np.float64)
    w_no_lo = np.asarray(bank.metrics_no_mass["hdi68_width"]["p16"], dtype=np.float64)
    w_no_hi = np.asarray(bank.metrics_no_mass["hdi68_width"]["p84"], dtype=np.float64)
    w_with_med = np.asarray(bank.metrics_with_mass["hdi68_width"]["median"], dtype=np.float64)
    w_with_lo = np.asarray(bank.metrics_with_mass["hdi68_width"]["p16"], dtype=np.float64)
    w_with_hi = np.asarray(bank.metrics_with_mass["hdi68_width"]["p84"], dtype=np.float64)

    frac_med = np.asarray(bank.fractional_improvement["median"], dtype=np.float64)
    frac_lo = np.asarray(bank.fractional_improvement["p16"], dtype=np.float64)
    frac_hi = np.asarray(bank.fractional_improvement["p84"], dtype=np.float64)

    K_med = np.asarray(bank.effective_event_gain["median"], dtype=np.float64)
    K_lo = np.asarray(bank.effective_event_gain["p16"], dtype=np.float64)
    K_hi = np.asarray(bank.effective_event_gain["p84"], dtype=np.float64)

    fig, axes = get_figure(nrows=2, ncols=2, figsize=(7.0, 5.4))
    ax_w: Axes = axes[0, 0]
    ax_p: Axes = axes[0, 1]
    ax_d: Axes = axes[1, 0]
    ax_k: Axes = axes[1, 1]

    # ---- Top-left: HDI68 width vs N ----
    ax_w.fill_between(sizes, w_no_lo, w_no_hi, color=VARIANT_NO_MASS, alpha=0.18, zorder=2)
    ax_w.plot(
        sizes,
        w_no_med,
        "o-",
        color=VARIANT_NO_MASS,
        markersize=4,
        linewidth=1.2,
        label=r"Without $M_z$",
        zorder=3,
    )
    ax_w.fill_between(sizes, w_with_lo, w_with_hi, color=VARIANT_WITH_MASS, alpha=0.18, zorder=2)
    ax_w.plot(
        sizes,
        w_with_med,
        "s--",
        color=VARIANT_WITH_MASS,
        markersize=4,
        linewidth=1.2,
        label=r"With $M_z$",
        zorder=3,
    )
    if len(sizes) > 1 and not np.isnan(w_no_med[0]) and w_no_med[0] > 0:
        n0 = sizes[0]
        ref_line = w_no_med[0] * np.sqrt(n0 / sizes)
        ax_w.plot(
            sizes,
            ref_line,
            ":",
            color=REFERENCE,
            linewidth=1.0,
            label=r"$\propto N^{-1/2}$",
            zorder=1,
        )
    ax_w.set_xscale("log")
    ax_w.set_yscale("log")
    ax_w.set_ylabel(r"68% HDI width of $h$")
    ax_w.set_xlabel(r"Number of events $N_\mathrm{det}$")
    ax_w.legend(loc="upper right", fontsize=7)
    ax_w.set_title("Posterior tightening", fontsize=9)

    # ---- Top-right: representative combined posterior at the largest N ----
    if bank.representative_posteriors_no_mass and bank.representative_posteriors_with_mass:
        last = -1
        n_show = bank.sizes[last]
        ax_p.plot(
            bank.h_grid,
            bank.representative_posteriors_no_mass[last],
            "-",
            color=VARIANT_NO_MASS,
            linewidth=1.2,
            label=r"Without $M_z$",
        )
        ax_p.plot(
            bank.h_grid,
            bank.representative_posteriors_with_mass[last],
            "--",
            color=VARIANT_WITH_MASS,
            linewidth=1.2,
            label=r"With $M_z$",
        )
        ax_p.axvline(
            bank.h_true,
            color=TRUTH,
            linestyle=":",
            linewidth=1.0,
            label="Injected",
        )
        ax_p.set_xlabel(r"$h$")
        ax_p.set_ylabel("Posterior (peak-norm.)")
        ax_p.set_title(f"Combined posterior @ N = {n_show}", fontsize=9)
        ax_p.legend(loc="upper right", fontsize=7)
        ax_p.set_xlim(0.59, 0.87)
        ax_p.set_ylim(-0.05, 1.15)

    # ---- Bottom-left: fractional improvement Δ(N) ----
    ax_d.fill_between(sizes, frac_lo * 100.0, frac_hi * 100.0, color=VARIANT_WITH_MASS, alpha=0.25)
    ax_d.plot(
        sizes,
        frac_med * 100.0,
        "o-",
        color=VARIANT_WITH_MASS,
        markersize=4,
        linewidth=1.2,
    )
    ax_d.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    ax_d.set_xscale("log")
    ax_d.set_xlabel(r"Number of events $N_\mathrm{det}$")
    ax_d.set_ylabel(r"$\Delta(N)$ [%]  (paired)")
    ax_d.set_title("Fractional tightening from $M_z$", fontsize=9)

    # ---- Bottom-right: effective event gain K(N) ----
    valid = ~np.isnan(K_med)
    if valid.any():
        ax_k.fill_between(
            sizes[valid], K_lo[valid], K_hi[valid], color=VARIANT_WITH_MASS, alpha=0.25
        )
        ax_k.plot(
            sizes[valid],
            K_med[valid],
            "s-",
            color=VARIANT_WITH_MASS,
            markersize=4,
            linewidth=1.2,
        )
    ax_k.axhline(1.0, color="black", linewidth=0.6, alpha=0.5)
    ax_k.set_xscale("log")
    ax_k.set_xlabel(r"Number of events $N_\mathrm{det}$")
    ax_k.set_ylabel(r"$K(N) = N_\mathrm{eff,without} / N$")
    ax_k.set_title("Effective event gain", fontsize=9)

    fig.tight_layout()
    return fig, axes
