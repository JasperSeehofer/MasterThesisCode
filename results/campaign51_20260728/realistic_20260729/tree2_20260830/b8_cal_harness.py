r"""B8.2 S2 -- universe driver + score-only aggregator for the [CAL] calibration harness.

`launched under rows #255/#268 -- tree 2 node B8.2.S2`

Design of record: ``results/campaign51_20260728/realistic_20260729/fanout1_20260829/
B8_2_HARNESS_DESIGN_20260829.md`` (sections 1-5, 8 S2 row). Generator of record (S1, verified):
``darksiren_emri/validation/correspondence_1d.py``'s ``host_mode="mixture_selected"`` +
``gw_scatter`` knob + resolved-flags out-parameter (``B8_2_S1_RECORD.md``,
``B8_2_S1_VERIFIER_REPORT.md`` -- generator code CONFIRMED correct; this driver additionally
closes the verifier's one gap, must_fix 1, the grid-split bit-identity property, see
:func:`verify_grid_split_bit_identity`).

No git operation is performed by this script (the orchestrator commits). No ssh. Foreground
only -- the driver is checkpointed PER UNIVERSE and re-invocable (``--max-wall-s`` makes it exit
cleanly, having checkpointed whatever completed, before a hard timeout kills it; re-running the
same command resumes by skipping seeds whose checkpoint already exists). Append-only outputs
(one JSON file per universe, never overwritten by a later run once written). Bounded-scope rule
(design §8): this script may not change any band, statistic definition, or the mixture law --
every statistic below is either a verbatim reuse of an existing implementation (attributed in
each docstring) or a restricted-domain sum of a formula ``bayesian_statistics.py`` already
computes globally (the per-bin count-audit terms, §1.2(b) -- NOT a new physics formula, see
:func:`alpha_g_phi_per_bin`/:func:`beta_gbar_phi_per_bin`).

Two run modes:

  1. **Driver** (default): draws ``--n-universes`` mirror universes at the estimator's own
     mixture law and scores each with the REAL ``BayesianStatistics.evaluate()`` (via
     ``run_mirror_seed_inprocess``), checkpointing one JSON per universe under ``--work-root``.
  2. **``--score-only``**: reads every checkpoint JSON under ``--work-root`` matching ``--cell``
     and prints the aggregate operating characteristics (design §4.1) -- F per channel, coverage
     at 50/68/90/95, the score-zero test by class, the absolute-count audit table. Per rule 2 this
     PRINTS band outcomes for information only; it never writes a verdict file or a "PASS"/"FAIL"
     judgement of record -- that is S4's registration + the chair's own read of this output.

Run (smoke, matches the launch stamp's resource ceiling -- event-cap<=20, workers<=2):

  uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
      --n-universes 2 --N 20 --event-cap 20 --cell S \
      --h-values 0.725,0.73,0.735 --workers 2 --max-wall-s 560

  uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
      --score-only --cell S
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

# Module object (not just names out of it) -- needed so the B8.2.S2b draw-weight cache (below)
# can monkeypatch the bare module-global name ``catalogue_selected_host_draw_weights`` that
# ``MirrorUniverseGenerator.draw_realization``'s "catalogue_selected"/"catalogue_selected_2d"/
# "mixture_selected" branches look up at CALL time (a plain global-scope name lookup in
# correspondence_1d's own namespace, per Python's LEGB rule) -- this is an additive change to
# THIS driver's own caller-visible reuse strategy, not an edit to correspondence_1d.py.
# Module object (not just names) -- needed for the B8.2.S2c per-h precompute cache (below), the
# SAME bare-module-global-name monkeypatch technique applied to bayesian_statistics.py's own
# ``evaluate()`` (five ``precompute_*`` free functions + the ``SimulationDetectionProbability``
# constructor call, all looked up unqualified inside ``evaluate()``'s own module namespace at
# call time -- confirmed by reading bayesian_statistics.py:4656-4823; none of these calls are
# ``self.`` or module-qualified). No line of bayesian_statistics.py is edited.
import darksiren_emri.bayesian_inference.bayesian_statistics as bayesian_statistics  # noqa: E402
import darksiren_emri.validation.correspondence_1d as correspondence_1d  # noqa: E402
from darksiren_emri.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from darksiren_emri.dark_siren_injection import _redshift_population_weight  # noqa: E402
from darksiren_emri.emri_rate import R_eff_per_mbh  # noqa: E402
from darksiren_emri.galaxy_catalogue.handler import (  # noqa: E402
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
)
from darksiren_emri.galaxy_catalogue.pixel_completeness import (  # noqa: E402
    M_TH_CACHE_PATH,
    CompletenessModel,
)
from darksiren_emri.physical_relations import dist_to_redshift  # noqa: E402
from darksiren_emri.validation.correspondence_1d import (  # noqa: E402
    CRB_CSV_PATH,
    H_GRID_41,
    H_TRUE,
    INJECTION_POOL_DIR,
    REDUCED_CATALOGUE_PATH,
    CorrespondenceConfig,
    HostPool,
    MirrorUniverseGenerator,
    _host_pool_from_handler,
    _load_galaxy_catalog_handler,
    assert_resolved_production_flags,
    build_b0i_2d_selection_objects,
    catalogue_selected_host_draw_weights,
    combine_log_likelihood,
    compute_catalogue_class_weight_p_g,
    kernel_smeared_survival,
    run_mirror_seed_inprocess,
)

# ── provenance-pinned inputs (every number {value; source; date}, CLAUDE.md) ─

FLOOR_JSON_PATH = (
    REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/fanout1_20260829/b8_information_floor.json"
)
B3_POP_JSON_PATH = (
    REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/fanout1_20260829/b3_pop_prediction.json"
)
ADJUDICATOR_SOURCE = (
    "results/venue_transfer_20260811/adjudicate_venue_transfer.py "
    "(my_pit/my_post_sd/my_hpd_contains/my_ks_uniform/binom_bands/trapz_norm, "
    "copied verbatim below per design §1.2(a); row #99 reproduced the registered "
    "readout to <= 5.33e-15)"
)
CANDIDATE_COUNT_METHOD_SOURCE = (
    "results/campaign51_20260728/realistic_20260729/fanout1_20260829/b4_imp_stage1_forecast.py"
    ":146-169 (candidate_counts, 'log-line order == first-h-block CSV order for "
    "non-zero-host events'), adapted verbatim below"
)

_LOGGER = logging.getLogger("b8_cal_harness")

LAUNCH_STAMP = "launched under rows #255/#268 -- tree 2 node B8.2.S2"

# B3.1's registered z-bin edges (design §1.2(b) / §2.4), transcribed from B3_POP_JSON_PATH's
# own "registered_bin_edges" key -- checked against the live file at import time (below) so this
# constant cannot silently drift from its source.
B3_1_BIN_EDGES: tuple[float, ...] = (0.075, 0.392, 0.559, 0.659, 0.753, 1.018)


def _check_b3_1_bin_edges_against_source() -> None:
    with open(B3_POP_JSON_PATH) as fh:
        registered = tuple(json.load(fh)["registered_bin_edges"])
    if registered != B3_1_BIN_EDGES:
        raise ValueError(
            f"B3_1_BIN_EDGES {B3_1_BIN_EDGES} has drifted from "
            f"{B3_POP_JSON_PATH}'s own registered_bin_edges {registered} -- "
            "update the constant (append-only note, per CLAUDE.md's every-number provenance rule)"
        )


_check_b3_1_bin_edges_against_source()

_FLOOR_N_REF = 1588  # b8_information_floor.json's n_events (B8.1 route B, production N).


def _load_floor_json() -> dict[str, Any]:
    with open(FLOOR_JSON_PATH) as fh:
        return dict(json.load(fh))


def sigma_floor_for(channel: str, n_events: int) -> tuple[float, dict[str, Any]]:
    """Look up (and, off N=1588, analytically rescale) the B8.1 information floor.

    Args:
        channel: ``"no_bh"`` (1D) or ``"with_bh"`` (2D, "configuration of record" cell --
            the pinned catalogue's own ``BH_MASS_ERROR``, sigma_M=1.99, design §1.3).
        n_events: The realized event count the harness is scoring at.

    Returns:
        ``(sigma_h_floor, provenance)`` -- ``provenance`` names the exact JSON key, its
        {value, source, date}, and (if ``n_events != 1588``) the analytic
        ``sqrt(N_ref/n_events)`` rescaling applied (Fisher information ~ N for i.i.d. events;
        an approximation off the reference N, flagged as such -- NOT a re-measurement).
    """
    d = _load_floor_json()
    key_path: tuple[str, ...]
    if channel == "no_bh":
        key_path = ("oneD", "GLADE_photo", "closed_form", "sigma_h_floor")
    elif channel == "with_bh":
        key_path = (
            "twoD",
            "GLADE_photo",
            "total_predictive_0.55dex",
            "closed_form",
            "sigma_h_floor",
        )
    else:
        raise ValueError(f"unknown channel {channel!r}; expected 'no_bh'/'with_bh'")
    node: Any = d
    for k in key_path:
        node = node[k]
    value_at_ref = float(node)
    scale = (_FLOOR_N_REF / n_events) ** 0.5 if n_events != _FLOOR_N_REF else 1.0
    provenance = {
        "value_at_n_ref": value_at_ref,
        "n_ref": _FLOOR_N_REF,
        "key_path": list(key_path),
        "source": str(FLOOR_JSON_PATH),
        "date": "2026-08-29",
        "n_events_requested": n_events,
        "scale_applied_sqrt_n_ref_over_n": scale,
        "value_used": value_at_ref * scale,
        "note": (
            "exact at n_events==n_ref; otherwise an ANALYTIC i.i.d.-Fisher-information "
            "rescaling, not a re-measurement at this N"
        ),
    }
    return value_at_ref * scale, provenance


# ── verbatim-reused posterior primitives (design §1.2(a), ADJUDICATOR_SOURCE) ─
# Copied (not imported -- results/venue_transfer_20260811/adjudicate_venue_transfer.py is a
# script with top-level side-effect-free helper functions, but has no package __init__.py, so a
# cross-directory `import` would not resolve under mypy's per-file module search; copying keeps
# this file self-contained per the results/ scripts convention, e.g. kwq1_score.py). Bodies are
# UNCHANGED from the source; only type annotations were added (a mypy-compliance no-op).


def trapz_norm(h: npt.NDArray[np.float64], lnp: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalise exp(lnp) on grid h by trapezoid rule. Returns density. (ADJUDICATOR_SOURCE)"""
    lnp = np.asarray(lnp, dtype=np.float64)
    p = np.exp(lnp - lnp.max())
    dh = np.diff(h)
    z = float(np.sum(0.5 * (p[1:] + p[:-1]) * dh))
    result: npt.NDArray[np.float64] = p / z
    return result


def my_cdf(h: npt.NDArray[np.float64], post: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Cumulative trapezoid mass. (ADJUDICATOR_SOURCE)"""
    dh = np.diff(h)
    seg = 0.5 * (post[1:] + post[:-1]) * dh
    result: npt.NDArray[np.float64] = np.concatenate([[0.0], np.cumsum(seg)])
    return result


def my_pit(h: npt.NDArray[np.float64], post: npt.NDArray[np.float64], h_true: float) -> float:
    """PIT = integral_{h<=h_true} p(h) dh. (ADJUDICATOR_SOURCE)"""
    cum = my_cdf(h, post)
    return float(np.interp(h_true, h, cum))


def my_post_sd(h: npt.NDArray[np.float64], post: npt.NDArray[np.float64]) -> float:
    """Posterior SD by trapezoid moments. (ADJUDICATOR_SOURCE)"""
    dh = np.diff(h)
    m1 = float(np.sum(0.5 * (post[1:] * h[1:] + post[:-1] * h[:-1]) * dh))
    m2 = float(np.sum(0.5 * (post[1:] * h[1:] ** 2 + post[:-1] * h[:-1] ** 2) * dh))
    return float(np.sqrt(max(m2 - m1 * m1, 0.0)))


def my_hpd_contains(
    h: npt.NDArray[np.float64], post: npt.NDArray[np.float64], h_true: float, level: float
) -> bool:
    """HPD credible-region containment (density-threshold construction). (ADJUDICATOR_SOURCE)"""
    post = np.asarray(post, dtype=np.float64)
    w = np.gradient(h)
    mass = post * w
    order = np.argsort(post)[::-1]
    csum = np.cumsum(mass[order])
    k = int(np.searchsorted(csum, level))
    k = min(k, order.size - 1)
    thresh = float(post[order[k]])
    p_true = float(np.interp(h_true, h, post))
    return bool(p_true >= thresh)


def my_ks_uniform(vals: list[float]) -> float:
    """One-sample KS distance against Uniform(0,1). (ADJUDICATOR_SOURCE)"""
    x = np.sort(np.asarray(vals, dtype=np.float64))
    n = x.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1, dtype=np.float64)
    d_plus = float(np.max(i / n - x))
    d_minus = float(np.max(x - (i - 1) / n))
    return max(d_plus, d_minus)


def binom_bands(p: float, n: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """2-sigma/3-sigma normal-approximation bands on a binomial proportion. (ADJUDICATOR_SOURCE)"""
    s = (p * (1.0 - p) / n) ** 0.5
    return (p - 2 * s, p + 2 * s), (p - 3 * s, p + 3 * s)


# ── count-audit per-bin decomposition (design §1.2(b)) ───────────────────────
# NOT a new physics formula: both functions restrict an EXISTING production sum (computed
# globally by precompute_global_catalog_selection/precompute_phi_selection_integrals,
# bayesian_statistics.py) to z-sub-ranges, using the identical eligibility mask, weights, and
# survival table those functions read. Self-checked in build_generative_context() by asserting
# the per-bin sums reproduce the SAME global scalars compute_catalogue_class_weight_p_g reports.


def alpha_g_phi_per_bin(
    galaxy_catalog: GalaxyCatalogueHandler,
    detection_probability_obj: SimulationDetectionProbability,
    phi_survival_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    h_true: float,
    bin_edges: tuple[float, ...],
    z_max_cap: float,
) -> npt.NDArray[np.float64]:
    r"""Per-bin catalogue-leg selection sum: alpha_G^phi(bin) = sum_{g in bin} w_g S_bar_phi(z_g;h).

    Restricts, to each B3.1 z-bin, the EXACT sum
    ``precompute_global_catalog_selection(..., with_bh_mass=False, phi_survival_table=...)``
    computes over the FULL catalogue (``bayesian_statistics.py`` 'Sigma^phi(h) = sum_g w_g
    S_bar_phi(z_g;h)', same eligible-galaxy mask ``z_g < z_max(h) & finite(M_g) & M_g>0``, same
    ``w_g = R_eff_per_mbh(M_g)/(1+z_g)``, same table-interpolated ``S_bar_phi``).
    """
    catalog = galaxy_catalog.reduced_galaxy_catalog
    z_all = catalog[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64)
    m_all = catalog[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64)
    z_max = dist_to_redshift(detection_probability_obj.get_dl_max(h_true), h=h_true)
    z_max = min(z_max, z_max_cap)
    eligible = (z_all < z_max) & np.isfinite(m_all) & (m_all > 0.0)
    z_g = z_all[eligible]
    m_g = m_all[eligible]
    w_g = np.asarray(R_eff_per_mbh(m_g), dtype=np.float64) / (1.0 + z_g)
    z_grid, s_phi = phi_survival_table[h_true]
    s_bar = np.interp(z_g, z_grid, s_phi)
    contrib = w_g * s_bar
    edges = np.asarray(bin_edges, dtype=np.float64)
    out = np.zeros(len(edges) - 1, dtype=np.float64)
    bin_idx = np.searchsorted(edges, z_g, side="right") - 1
    for b in range(len(out)):
        out[b] = float(contrib[bin_idx == b].sum())
    return out


def beta_gbar_phi_per_bin(
    completeness: CompletenessModel,
    phi_survival_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    h_true: float,
    bin_edges: tuple[float, ...],
) -> npt.NDArray[np.float64]:
    r"""Per-bin completion-leg integral: beta_Gbar^phi(bin) = int_bin w_pop(1-fbar) S_bar_phi dz.

    Restricts the EXACT integrand ``precompute_phi_selection_integrals`` integrates over the
    full survival-table domain (``bayesian_statistics.py`` beta_Gbar^phi(h)) to each B3.1 z-bin,
    trapezoid on the table's own 1500-node grid masked to the bin (a grid-resolution
    approximation at the exact bin edges, not a re-derivation of the integrand).
    """
    z_grid, s_phi = phi_survival_table[h_true]
    p_pop = np.asarray(_redshift_population_weight(z_grid, h_true), dtype=np.float64)
    f_bar = np.clip(np.asarray(completeness.f_bar(z_grid, h_true), dtype=np.float64), 0.0, 1.0)
    integrand = (1.0 - f_bar) * s_phi * p_pop
    edges = np.asarray(bin_edges, dtype=np.float64)
    out = np.zeros(len(edges) - 1, dtype=np.float64)
    for b in range(len(out)):
        lo, hi = edges[b], edges[b + 1]
        mask = (z_grid >= lo) & (z_grid <= hi)
        if int(mask.sum()) < 2:
            continue
        out[b] = float(np.trapezoid(integrand[mask], z_grid[mask]))
    return out


# ── candidate-count log parser (CANDIDATE_COUNT_METHOD_SOURCE, adapted) ──────

_POSSIBLE_HOSTS_RE = re.compile(r"possible hosts found (\d+)/(\d+)\.\.\.")


def parse_candidate_counts(
    log_path: Path, diag_csv: Path
) -> tuple[dict[int, tuple[int, int]], str]:
    """Per-event (no_bh, with_bh) candidate-ball sizes from the evaluate() INFO log.

    Method: log-line order == first-h-block CSV order for non-zero-host events
    (CANDIDATE_COUNT_METHOD_SOURCE). Works identically whether the run was a single whole-grid
    ``evaluate()`` call or the two-call ``h_bounds``-pinned split (design's grid-split chunking):
    the candidate ball is h-list-independent (built once per call from the FIXED h_bounds
    window), so concatenating both calls' logs into one file (as this driver does) reproduces
    exactly the same block structure a single whole-grid call would have logged.

    Returns:
        ``({event_idx: (n_no_bh, n_with_bh)}, reason)`` -- ``reason`` is empty on success, else
        names why parsing was skipped (never raises; a caller treats this as best-effort).
    """
    if not log_path.is_file():
        return {}, "no log file"
    counts: list[tuple[int, int]] = []
    with open(log_path) as f:
        for line in f:
            if "possible hosts found" in line:
                m = _POSSIBLE_HOSTS_RE.search(line)
                if m:
                    counts.append((int(m.group(1)), int(m.group(2))))
    if not diag_csv.is_file():
        return {}, "no diagnostics csv"
    df = pd.read_csv(diag_csv)
    if df.empty:
        return {}, "empty diagnostics csv"
    h_sorted = sorted(df["h"].unique())
    first = df[df["h"] == h_sorted[0]].sort_values("event_idx").reset_index(drop=True)
    zero = (first["L_cat_no_bh"] == 0.0) & (first["L_cat_with_bh"] == 0.0)
    n_print = int((~zero).sum())
    if n_print == 0:
        return {int(e): (0, 0) for e in first["event_idx"].tolist()}, ""
    if len(counts) != n_print * len(h_sorted):
        return {}, f"ALIGNMENT FAIL: {len(counts)} log lines vs {n_print}x{len(h_sorted)}"
    block = counts[:n_print]
    printed_idx = first.loc[~zero, "event_idx"].tolist()
    out = {int(e): (0, 0) for e in first["event_idx"].tolist()}
    for e, c in zip(printed_idx, block, strict=True):
        out[int(e)] = c
    return out, ""


# ── runtime plumbing (stamps, affinity, generative context) ──────────────────


def git_stamp() -> dict[str, Any]:
    """A22 commit + dirty-state stamp (read-only git queries only; no git writes)."""

    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_paths": _run("status", "--porcelain").splitlines(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "launch_stamp": LAUNCH_STAMP,
    }


def pin_affinity(workers: int) -> dict[str, Any]:
    """Pin CPU affinity so ``evaluate()``'s ``max(1, affinity-2)`` sizing yields ``workers``.

    Mirrors the S1/verifier convention (taskset pin -> num_workers via affinity - 2); a no-op
    (recorded as such) on platforms without ``os.sched_setaffinity``.
    """
    if not hasattr(os, "sched_setaffinity"):
        return {"applied": False, "reason": "os.sched_setaffinity unavailable on this platform"}
    available = sorted(os.sched_getaffinity(0))
    n_cpus = max(1, workers) + 2
    chosen = set(available[:n_cpus]) if n_cpus <= len(available) else set(available)
    os.sched_setaffinity(0, chosen)
    return {"applied": True, "requested_workers": workers, "pinned_cpus": sorted(chosen)}


@dataclass
class GenerativeContext:
    """The estimator-derived objects the ``mixture_selected`` generator + count audit need.

    Built ONCE per script invocation (the ``functools.lru_cache`` decorators already on
    ``_load_galaxy_catalog_handler``/``build_b0i_2d_selection_objects`` mean a second call within
    the same process is free; this dataclass just avoids re-deriving ``host_pool``/``p_g_info``).
    """

    handler: GalaxyCatalogueHandler
    host_pool: HostPool
    completeness: CompletenessModel
    phi_survival_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
    detection_probability: SimulationDetectionProbability
    p_g_info: dict[str, float]
    n_pred_by_bin: npt.NDArray[np.float64]
    n_pred_self_check: dict[str, Any]


def build_generative_context(bin_edges: tuple[float, ...] = B3_1_BIN_EDGES) -> GenerativeContext:
    """Build the handler/completeness/survival-table/p_g objects (the dominant one-time cost)."""
    handler = _load_galaxy_catalog_handler(REDUCED_CATALOGUE_PATH)
    host_pool = _host_pool_from_handler(handler)
    completeness, phi_survival_table, detection_probability = build_b0i_2d_selection_objects(
        h_true=H_TRUE, injection_dir=INJECTION_POOL_DIR
    )
    p_g_info = compute_catalogue_class_weight_p_g(
        handler, h_true=H_TRUE, injection_dir=INJECTION_POOL_DIR
    )
    # RAW catalogue-leg per-bin sum: sum_{g in bin} w_g * S_bar_phi(z_g;h) -- the SAME raw
    # quantity precompute_global_catalog_selection(with_bh_mass=False, phi_survival_table=...)
    # sums globally as Sigma^phi(h) (compute_catalogue_class_weight_p_g's own "sigma_phi" key,
    # NOT its "alpha_G_phi" key -- path_a_mixture_objects's alpha_G^phi = r_Malm * beta_G^phi is
    # a DIFFERENT, Malmquist-rescaled quantity in beta_G^phi's continuous-integral units, per its
    # own docstring: alpha_G^phi = Sigma^4D/n_hat_w_phi = beta_G^phi * r_Malm; Sigma^phi is only
    # its *Monte-Carlo estimate* of beta_G^phi, "no calibration constant needed" by construction,
    # bayesian_statistics.py precompute_global_catalog_selection docstring).
    alpha_bins_raw = alpha_g_phi_per_bin(
        handler, detection_probability, phi_survival_table, H_TRUE, bin_edges, z_max_cap=1.5
    )
    beta_bins = beta_gbar_phi_per_bin(completeness, phi_survival_table, H_TRUE, bin_edges)
    sigma_phi_global = p_g_info["sigma_phi"]
    alpha_g_phi_global = p_g_info["alpha_G_phi"]
    # Rescale the raw per-bin shape into alpha_G^phi's units (the SAME r_Malm-type ratio
    # path_a_mixture_objects applies globally: alpha_G^phi/Sigma^phi), so
    # (alpha_bins_scaled + beta_bins) sits in beta_Gbar^phi's units and its total reproduces
    # D_tilde_phi's own alpha_G^phi term up to the same under/overflow tails
    # z_true_hist.n_below_lowest_edge/n_above_highest_edge already report per universe.
    rescale = alpha_g_phi_global / sigma_phi_global if sigma_phi_global > 0.0 else 0.0
    alpha_bins_scaled = alpha_bins_raw * rescale
    d_tilde_phi = p_g_info["D_tilde_phi"]
    n_pred_shape = (alpha_bins_scaled + beta_bins) / d_tilde_phi
    # Self-check (design §2.4 acceptance instrument, not a new physics claim): each per-bin RAW
    # decomposition must sum back (up to the bins' domain not fully tiling [0, z_max(h)], i.e.
    # the tails outside [min(bin_edges), max(bin_edges)]) to the SAME raw global scalar the
    # harness's own p_g computation reports, apples-to-apples in units.
    self_check = {
        "sigma_phi_binned_sum_raw": float(alpha_bins_raw.sum()),
        "sigma_phi_global": sigma_phi_global,
        "alpha_g_phi_rescale_factor": rescale,
        "alpha_g_phi_global": alpha_g_phi_global,
        "beta_gbar_phi_binned_sum": float(beta_bins.sum()),
        "beta_gbar_phi_global": p_g_info["beta_Gbar_phi"],
        "note": (
            "binned sums undercount the global scalar by the mass outside "
            f"[{bin_edges[0]}, {bin_edges[-1]}] (below the lowest / above the highest B3.1 "
            "edge) -- NOT a discrepancy, see z_true_hist's n_below/n_above_edge per universe"
        ),
    }
    return GenerativeContext(
        handler=handler,
        host_pool=host_pool,
        completeness=completeness,
        phi_survival_table=phi_survival_table,
        detection_probability=detection_probability,
        p_g_info=p_g_info,
        n_pred_by_bin=n_pred_shape,
        n_pred_self_check=self_check,
    )


# ── host-draw-weight cache (B8.2.S2b, 2026-08-30) ─────────────────────────────
#
# B8_2_S2_RECORD.md §5's cost finding: ``draw_realization``'s OWN wall time for
# ``host_mode="mixture_selected"`` is lower-bounded at >318s (isolated separately from context
# build and from ``evaluate()``), and the mechanism is identified: the catalogue-hosted branch
# calls ``catalogue_selected_host_draw_weights``, which runs ``kernel_smeared_survival`` over the
# ENTIRE ~20.8M-row pinned host pool -- a cost that is a pure function of ``(host_pool, h_true)``
# (§5's closing paragraph / §7 item 2), NOT of the realization seed or ``n_events``. Every
# universe this driver draws recomputes it from scratch even though, within one script
# invocation, ``ctx.host_pool``/``ctx.phi_survival_table``/``ctx.completeness`` are the SAME
# objects every time (``build_generative_context()`` is called once in ``main()``) -- and across
# separate invocations (e.g. one process per N-ladder point) the pinned catalogue + H_TRUE never
# change either. This cache memoizes the call: an in-process dict (free reuse across universes
# within one invocation) plus an on-disk ``.npz`` under ``--work-root`` (free reuse across
# invocations), keyed by a content hash of the pool's actual z/M arrays + h + the injection pool
# path + a source-hash of the two functions this depends on (so an edit to either one
# self-invalidates every existing cache entry -- no version constant to remember to bump).
#
# This is an ADDITIVE change to THIS DRIVER's own reuse-across-universes strategy (monkeypatching
# the bare module-global name ``correspondence_1d.catalogue_selected_host_draw_weights`` that
# ``draw_realization``'s host-mode branches look up at call time) -- it does not edit
# ``correspondence_1d.py``, does not change the mixture law, the RNG stream, or any band/
# statistic definition (design §8's bounded-scope rule): the cached path returns the IDENTICAL
# ``(normalized_weights, w_g, s_tilde_phi)`` arrays the uncached call would, so every downstream
# RNG consumer (``rng.choice(..., p=host_w)``, ``_draw_kernel_survival_redshifts``) sees the same
# floats either way -- proven bit-for-bit in B8_2_S2_RECORD.md §8's byte-identity check.

_DRAW_WEIGHT_CACHE_IN_PROCESS: dict[
    str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
] = {}
_DRAW_WEIGHT_CACHE_DIR: Path | None = None
_DRAW_WEIGHT_CACHE_ENABLED: bool = True
# Side-channel: run_one_universe reads this immediately after draw_realization() returns to
# attach {hit, compute_s, ...} to the checkpoint -- draw_realization's own return value carries
# no cache-provenance field (correspondence_1d.py is out of this stage's edit scope).
LAST_DRAW_WEIGHT_CACHE_INFO: dict[str, Any] = {"hit": "not_invoked_this_process_yet"}

_ORIGINAL_CATALOGUE_SELECTED_HOST_DRAW_WEIGHTS = catalogue_selected_host_draw_weights


def _reset_last_draw_weight_cache_info() -> None:
    """Call before each ``draw_realization`` so a no-catalogue-hosted-hosts draw (n_g==0, the
    cached function never invoked) cannot leave a PRIOR universe's cache-hit info attached."""
    global LAST_DRAW_WEIGHT_CACHE_INFO
    LAST_DRAW_WEIGHT_CACHE_INFO = {"hit": "not_invoked_this_draw"}


def _draw_weight_cache_key(pool: HostPool, h: float) -> str:
    """Content hash of the actual pool arrays + h + injection dir + a source fingerprint.

    Hashing the pool's OWN ``z``/``M`` arrays (not merely its identity or shape) means a
    genuinely different catalogue can never collide with a stale cache entry -- ``id(pool)``
    would not survive a fresh process (the on-disk leg) and a shape-only key would not catch a
    same-shape, different-content pool.
    """
    hasher = hashlib.sha256()
    hasher.update(str(REDUCED_CATALOGUE_PATH).encode())
    hasher.update(str(INJECTION_POOL_DIR).encode())
    hasher.update(np.ascontiguousarray(pool.z, dtype=np.float64).tobytes())
    if pool.M is not None:
        hasher.update(np.ascontiguousarray(pool.M, dtype=np.float64).tobytes())
    hasher.update(repr(float(h)).encode())
    hasher.update(inspect.getsource(_ORIGINAL_CATALOGUE_SELECTED_HOST_DRAW_WEIGHTS).encode())
    hasher.update(inspect.getsource(kernel_smeared_survival).encode())
    return hasher.hexdigest()[:32]


def configure_draw_weight_cache(work_root: Path, enabled: bool) -> None:
    """Point the process-global cache at ``work_root`` and enable/disable it.

    Must be called once, before the first ``draw_realization(host_mode="mixture_selected", ...)``
    of a script invocation -- ``main()`` calls this right after parsing args. ``enabled=False``
    (``--no-draw-weight-cache``) bypasses the cache entirely (recomputes every call, exactly the
    pre-B8.2.S2b behaviour) -- used only for the §8 byte-identity comparison against a cached run.
    """
    global _DRAW_WEIGHT_CACHE_DIR, _DRAW_WEIGHT_CACHE_ENABLED
    _DRAW_WEIGHT_CACHE_DIR = work_root / "draw_weight_cache"
    _DRAW_WEIGHT_CACHE_ENABLED = enabled
    if enabled:
        _DRAW_WEIGHT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cached_catalogue_selected_host_draw_weights(
    pool: HostPool,
    phi_survival_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    completeness: CompletenessModel,
    h: float = H_TRUE,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Drop-in replacement for ``catalogue_selected_host_draw_weights`` (same signature and
    return contract) -- monkeypatched onto ``correspondence_1d``'s module namespace so every
    ``draw_realization`` call site that looks up the bare name at call time (``"catalogue_
    selected"``/``"catalogue_selected_2d"``/``"mixture_selected"``) gets the cached path. This
    harness only ever exercises ``"mixture_selected"``, but the wrapper is correct for all three.
    """
    global LAST_DRAW_WEIGHT_CACHE_INFO
    if not _DRAW_WEIGHT_CACHE_ENABLED:
        t0 = time.time()
        result = _ORIGINAL_CATALOGUE_SELECTED_HOST_DRAW_WEIGHTS(
            pool, phi_survival_table, completeness, h=h
        )
        LAST_DRAW_WEIGHT_CACHE_INFO = {
            "cache_enabled": False,
            "hit": "disabled",
            "compute_s": time.time() - t0,
        }
        return result

    key = _draw_weight_cache_key(pool, h)
    cached = _DRAW_WEIGHT_CACHE_IN_PROCESS.get(key)
    if cached is not None:
        LAST_DRAW_WEIGHT_CACHE_INFO = {"cache_enabled": True, "hit": "in_process", "key": key}
        return cached

    if _DRAW_WEIGHT_CACHE_DIR is None:
        raise RuntimeError(
            "draw-weight cache used before configure_draw_weight_cache() was called -- "
            "main() must call it once, right after parsing args"
        )
    npz_path = _DRAW_WEIGHT_CACHE_DIR / f"draw_weights_{key}.npz"
    if npz_path.is_file():
        with np.load(npz_path) as z:
            result = (
                np.asarray(z["normalized"], dtype=np.float64),
                np.asarray(z["w_g"], dtype=np.float64),
                np.asarray(z["s_tilde_phi"], dtype=np.float64),
            )
        _DRAW_WEIGHT_CACHE_IN_PROCESS[key] = result
        LAST_DRAW_WEIGHT_CACHE_INFO = {
            "cache_enabled": True,
            "hit": "on_disk",
            "key": key,
            "path": str(npz_path),
        }
        return result

    t0 = time.time()
    result = _ORIGINAL_CATALOGUE_SELECTED_HOST_DRAW_WEIGHTS(
        pool, phi_survival_table, completeness, h=h
    )
    compute_s = time.time() - t0
    normalized, w_g, s_tilde_phi = result
    # Write-then-rename so a killed/interrupted process (this repo's own convention for
    # long-running local jobs) can never leave a reader-visible partial .npz.
    # np.savez APPENDS ".npz" to the filename if it doesn't already end in ".npz" -- a tmp name
    # ending in ".tmp" (e.g. "<key>.npz.tmp") therefore gets written as "<key>.npz.tmp.npz", not
    # "<key>.npz.tmp", so a plain tmp_path.replace(npz_path) then raises FileNotFoundError
    # (caught live in the first B8.2.S2b timing run). Use a tmp name that ALREADY ends in ".npz"
    # so numpy writes exactly that path, no silent extra suffix.
    tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez(tmp_path, normalized=normalized, w_g=w_g, s_tilde_phi=s_tilde_phi)
    tmp_path.replace(npz_path)
    _DRAW_WEIGHT_CACHE_IN_PROCESS[key] = result
    LAST_DRAW_WEIGHT_CACHE_INFO = {
        "cache_enabled": True,
        "hit": "miss",
        "key": key,
        "path": str(npz_path),
        "compute_s": compute_s,
    }
    return result


correspondence_1d.catalogue_selected_host_draw_weights = (
    _cached_catalogue_selected_host_draw_weights
)


# ── per-h catalogue-scale precompute cache (B8.2.S2c, 2026-08-31) ────────────
#
# Boundary finding (full writeup: B8_2_S2_RECORD.md §10). The N=106 ladder universe's
# call_0/call_1 cost (2632s/2750s, rows #255/#268) is dominated NOT by the per-event posterior
# loop but by five free functions ``evaluate()`` calls as LOCAL work on every single invocation,
# regardless of ``self`` state: ``precompute_completion_denominator`` (D(h)), ``precompute_
# missing_completion_denominator`` (beta_Gbar(h)), ``precompute_phi_marginal_survival``
# (S_bar_phi(z;h)), ``precompute_phi_selection_integrals`` (beta_G^phi/beta_Gbar^phi), and
# ``precompute_global_catalog_selection`` (Sigma_global(h)/Sigma^phi(h), the one that scans the
# full ~20.8M-row catalogue -- called THREE times per evaluate(), once per with_bh_mass branch
# plus once more under the phi convention). Every one of these functions' OWN docstring says so
# explicitly: precompute_global_catalog_selection -- "The sum is event-INDEPENDENT, so it is
# precomputed once per h like D(h)"; precompute_completion_denominator -- "D(h) is
# event-independent; compute once per h-value". None of the five takes ``events`` or
# ``self.cramer_rao_bounds`` as an argument (verified against their signatures).
#
# Reuse of a single ``BayesianStatistics`` instance across universes (the escape valve this
# stage's brief names) does NOT reach this cost: (a) ``self.cramer_rao_bounds`` is loaded ONCE in
# ``__init__`` and never reloaded inside ``evaluate()`` -- a reused instance would silently
# re-score a PRIOR universe's events unless the harness reimplemented __init__'s CSV-load logic
# from outside the class (fragile, out of this stage's edit scope); (b) even if (a) were solved,
# the five functions above are LOCAL variables inside ``evaluate()``'s body, recomputed
# unconditionally on every call -- reusing ``self`` buys zero savings for them. This boundary does
# not exist; it is not implemented.
#
# The boundary that DOES exist and IS implemented below: all five ``precompute_*`` functions, and
# the ``SimulationDetectionProbability`` constructor call that builds their shared
# ``detection_probability_obj`` argument, are looked up as BARE module-global names inside
# bayesian_statistics.py's own ``evaluate()`` method (the identical call-time LEGB lookup the
# B8.2.S2b draw-weight cache above already exploits for correspondence_1d.py). Monkeypatching
# these six names on the ``bayesian_statistics`` module object intercepts every call ``evaluate()``
# makes, from OUTSIDE bayesian_statistics.py -- no line of that file changes. The multiprocessing
# pool evaluate() spawns for the PER-EVENT posterior loop uses "forkserver"/"spawn" (bayesian_
# statistics.py:5198-5213, confirmed by reading the pool construction), not "fork": worker
# processes never call any of these six names themselves -- they receive the ALREADY-COMPUTED
# tables/objects via ``initargs`` -- so this driver-side monkeypatch (applied only in the main
# process, before the pool exists) cannot desync from what workers see; it is not a race with the
# worker-spawn mechanism.
#
# Cache key composition: since the five functions never receive ``events``/``cramer_rao_bounds``,
# their return value is a pure function of (h_values, the pinned catalogue, the pinned injection
# pool + P_det config, the frozen m_th completeness cache, and a handful of scalar/string flags)
# -- exactly the "legitimately reusable without changing any computed value" set the brief asks
# for. The key hashes CONTENT, not object identity, for the same reason the S2b draw-weight cache
# does (a fresh process must be able to hit an on-disk entry from an earlier one): the catalogue
# is fingerprinted by its own z/M arrays (:func:`_catalogue_fingerprint`), the completeness cache
# by the frozen ``.npy`` file's path+size+mtime (:func:`_completeness_fingerprint` -- the file is
# documented as frozen, "the SAME .npy file... byte-identical", pixel_completeness.py's own
# ``from_cache_or_build`` docstring), and ``detection_probability_obj`` by the constructor
# arguments used to build it (:func:`_cached_simulation_detection_probability` stamps that key
# onto the returned instance as ``_b8s2c_cache_fingerprint`` so downstream callers never need to
# guess at what determines its content). A source-hash of each wrapped function is folded into
# every key (mirrors the draw-weight cache: an edit to any of these six names in
# bayesian_statistics.py self-invalidates every existing cache entry, no version constant to
# remember to bump). ``--no-precompute-cache`` disables all six wrappers (falls through to the
# ORIGINAL function every call) for the §8/§10 byte-identity comparison.
#
# KNOWN LIMITATION (stated plainly, not glossed over): only the five ``precompute_*`` dict/array
# results are persisted ON DISK (small, trivially picklable). The ``SimulationDetectionProbability``
# instance itself (whose construction ALSO reloads+regrids the injection pool every evaluate()
# call) is cached IN-PROCESS ONLY -- pickling+restoring a live estimator object across process
# invocations (the ladder's checkpoint/resume granularity, main()'s own "re-running the same
# command resumes" design) was judged out of scope for this smoke-bounded stage; every NEW process
# invocation pays that one construction once (not once per universe), then reuses it in-process
# for every remaining universe that invocation scores, and its on-disk ``precompute_*`` cache
# entries hit immediately (same content-derived key, independent of object identity) even in a
# fresh process.

_PRECOMPUTE_CACHE_IN_PROCESS: dict[str, Any] = {}
_PRECOMPUTE_CACHE_DIR: Path | None = None
_PRECOMPUTE_CACHE_ENABLED: bool = True
_CATALOGUE_FINGERPRINT_BY_ID: dict[int, str] = {}
# Side-channel populated by every cached call this process makes since the last
# _reset_precompute_cache_info() -- attached to the checkpoint by run_one_universe, mirroring
# LAST_DRAW_WEIGHT_CACHE_INFO's role for the draw-weight cache above.
LAST_PRECOMPUTE_CACHE_INFO: dict[str, list[dict[str, Any]]] = {}


def configure_precompute_cache(work_root: Path, enabled: bool) -> None:
    """Point the process-global precompute cache at ``work_root`` and enable/disable it.

    Must be called once, before the first ``evaluate()`` call of a script invocation --
    ``main()`` calls this right after parsing args (alongside ``configure_draw_weight_cache``).

    ``work_root.resolve()`` (NOT the bare, possibly-relative ``work_root``) -- unlike the
    draw-weight cache above (only ever touched from ``draw_realization()``, which runs BEFORE
    ``run_mirror_seed_inprocess``'s internal ``os.chdir(work_root)``), this cache's wrapped
    functions run INSIDE ``bs.evaluate()``, i.e. AFTER that chdir. A relative ``_PRECOMPUTE_
    CACHE_DIR`` would then resolve against the wrong (per-universe, chdir'd) cwd -- caught live
    (``FileNotFoundError`` on the first smoke run) rather than shipped; see B8_2_S2_RECORD.md §10.
    """
    global _PRECOMPUTE_CACHE_DIR, _PRECOMPUTE_CACHE_ENABLED
    _PRECOMPUTE_CACHE_DIR = work_root.resolve() / "precompute_cache"
    _PRECOMPUTE_CACHE_ENABLED = enabled
    if enabled:
        _PRECOMPUTE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _reset_precompute_cache_info() -> None:
    """Call before each universe's ``evaluate()`` call(s) so a checkpoint only ever reports
    THIS universe's cache hits/misses, never a prior universe's leftover entries."""
    LAST_PRECOMPUTE_CACHE_INFO.clear()


def _record_precompute_cache_info(name: str, info: dict[str, Any]) -> None:
    LAST_PRECOMPUTE_CACHE_INFO.setdefault(name, []).append(info)


def _catalogue_fingerprint(galaxy_catalog: GalaxyCatalogueHandler) -> str:
    """Content hash of the catalogue's own z/M columns (the SAME two arrays
    :func:`precompute_global_catalog_selection` sums over) -- memoized by ``id()`` since the
    harness reuses ONE handler object (:data:`GenerativeContext.handler`) for the whole
    invocation, so this 20.8M-row hash is only ever paid once per process, not once per call.
    """
    obj_id = id(galaxy_catalog)
    cached = _CATALOGUE_FINGERPRINT_BY_ID.get(obj_id)
    if cached is not None:
        return cached
    catalog = galaxy_catalog.reduced_galaxy_catalog
    hasher = hashlib.sha256()
    hasher.update(str(REDUCED_CATALOGUE_PATH).encode())
    hasher.update(
        np.ascontiguousarray(
            catalog[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64)
        ).tobytes()
    )
    hasher.update(
        np.ascontiguousarray(
            catalog[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64)
        ).tobytes()
    )
    digest = hasher.hexdigest()[:32]
    _CATALOGUE_FINGERPRINT_BY_ID[obj_id] = digest
    return digest


def _completeness_fingerprint(completeness: Any) -> str:
    """Content proxy for the frozen m_th completeness cache: path+size+mtime of
    :data:`M_TH_CACHE_PATH` (``pixel_completeness.from_cache_or_build``'s own docstring: "the
    SAME .npy file is loaded byte-identically by injection and inference" -- a frozen artefact,
    not rebuilt within a campaign; a genuinely rebuilt file changes this fingerprint)."""
    try:
        st = os.stat(M_TH_CACHE_PATH)
        return f"{M_TH_CACHE_PATH}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return f"{M_TH_CACHE_PATH}:missing"


def _phi_table_fingerprint(
    phi_survival_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None,
) -> str:
    if not phi_survival_table:
        return "none"
    hasher = hashlib.sha256()
    for h in sorted(phi_survival_table):
        z_grid, s_phi = phi_survival_table[h]
        hasher.update(repr(float(h)).encode())
        hasher.update(np.ascontiguousarray(z_grid, dtype=np.float64).tobytes())
        hasher.update(np.ascontiguousarray(s_phi, dtype=np.float64).tobytes())
    return hasher.hexdigest()[:32]


def _detection_probability_fingerprint(detection_probability_obj: Any) -> str | None:
    """Read back the fingerprint :func:`_cached_simulation_detection_probability` stamped on the
    object at construction time. ``None`` means the object did not come from that cached factory
    (cache disabled, or some future call site this stage did not anticipate) -- every caller below
    treats ``None`` as "do not guess a key", falling through to the uncached original function for
    that one call rather than risk a wrong cache hit.
    """
    fp = getattr(detection_probability_obj, "_b8s2c_cache_fingerprint", None)
    return fp if isinstance(fp, str) else None


def _precompute_cache_lookup_or_compute(name: str, key: str, compute: Callable[[], Any]) -> Any:
    """Shared get-or-compute path (in-process dict, then on-disk pickle, then compute +
    write-then-rename) for every wrapper below -- ``compute`` is a zero-arg closure over the
    ORIGINAL (un-monkeypatched) function bound to its call-site arguments, so this helper never
    needs to know any wrapped function's own signature.
    """
    full_key = f"{name}_{key}"
    if not _PRECOMPUTE_CACHE_ENABLED:
        t0 = time.time()
        result = compute()
        _record_precompute_cache_info(name, {"hit": "disabled", "compute_s": time.time() - t0})
        return result
    cached = _PRECOMPUTE_CACHE_IN_PROCESS.get(full_key)
    if cached is not None:
        _record_precompute_cache_info(name, {"hit": "in_process", "key": full_key})
        return cached
    if _PRECOMPUTE_CACHE_DIR is None:
        raise RuntimeError(
            "precompute cache used before configure_precompute_cache() was called -- "
            "main() must call it once, right after parsing args"
        )
    pkl_path = _PRECOMPUTE_CACHE_DIR / f"{full_key}.pkl"
    if pkl_path.is_file():
        with open(pkl_path, "rb") as fh:
            result = pickle.load(fh)  # noqa: S301 -- this process's own trusted cache dir
        _PRECOMPUTE_CACHE_IN_PROCESS[full_key] = result
        _record_precompute_cache_info(
            name, {"hit": "on_disk", "key": full_key, "path": str(pkl_path)}
        )
        return result
    t0 = time.time()
    result = compute()
    compute_s = time.time() - t0
    # Write-then-rename (same crash-safety convention as the draw-weight cache's .npz above).
    tmp_path = pkl_path.with_name(pkl_path.name + ".tmp")
    with open(tmp_path, "wb") as fh:
        pickle.dump(result, fh)
    tmp_path.replace(pkl_path)
    _PRECOMPUTE_CACHE_IN_PROCESS[full_key] = result
    _record_precompute_cache_info(
        name, {"hit": "miss", "key": full_key, "path": str(pkl_path), "compute_s": compute_s}
    )
    return result


_ORIGINAL_SIMULATION_DETECTION_PROBABILITY = SimulationDetectionProbability


def _cached_simulation_detection_probability(*args: Any, **kwargs: Any) -> Any:
    """Drop-in replacement for ``SimulationDetectionProbability(...)`` -- monkeypatched onto
    ``bayesian_statistics``'s module namespace. Every call this harness's ``evaluate()`` invocation
    makes uses the SAME constructor arguments (same injection dir, SNR threshold, bin counts,
    estimator, z-resolved flags -- this driver never varies them); this wrapper reuses ONE
    instance across universes/calls instead of reloading + regridding the injection pool from
    scratch every time, and stamps the resulting instance with the fingerprint the ``precompute_*``
    wrappers below read back via :func:`_detection_probability_fingerprint`.
    """
    if not _PRECOMPUTE_CACHE_ENABLED:
        t0 = time.time()
        obj: Any = _ORIGINAL_SIMULATION_DETECTION_PROBABILITY(*args, **kwargs)
        # this driver's own cache tag, not a SimulationDetectionProbability attribute --
        # setattr (not obj.foo=...) so mypy does not check it against that class's declared
        # attributes (`obj` is deliberately `Any`-typed above for the same reason).
        setattr(obj, "_b8s2c_cache_fingerprint", None)  # noqa: B010
        _record_precompute_cache_info(
            "detection_probability_construct", {"hit": "disabled", "compute_s": time.time() - t0}
        )
        return obj

    hasher = hashlib.sha256()
    hasher.update(repr(args).encode())
    hasher.update(repr(sorted(kwargs.items(), key=lambda kv: kv[0])).encode())
    key = "detprob_" + hasher.hexdigest()[:32]
    cached = _PRECOMPUTE_CACHE_IN_PROCESS.get(key)
    if cached is not None:
        _record_precompute_cache_info(
            "detection_probability_construct", {"hit": "in_process", "key": key}
        )
        return cached
    t0 = time.time()
    obj = _ORIGINAL_SIMULATION_DETECTION_PROBABILITY(*args, **kwargs)
    setattr(obj, "_b8s2c_cache_fingerprint", key)  # noqa: B010
    _PRECOMPUTE_CACHE_IN_PROCESS[key] = obj
    _record_precompute_cache_info(
        "detection_probability_construct",
        {"hit": "miss", "key": key, "compute_s": time.time() - t0},
    )
    return obj


def _make_cached_precompute(
    original_func: Callable[..., Any],
    name: str,
    key_parts: Callable[[dict[str, Any]], list[str] | None],
) -> Callable[..., Any]:
    """Build a cached drop-in replacement for one ``bayesian_statistics.py`` free function.

    ``key_parts`` receives the call's fully-bound arguments (positional AND keyword, defaults
    applied via ``inspect.signature(...).bind(...).apply_defaults()``) as a ``{param_name: value}``
    dict, and returns the function-specific cache-key components (or ``None`` when a required
    fingerprint -- e.g. :func:`_detection_probability_fingerprint` -- is unavailable, in which case
    this wrapper falls through to ``original_func`` uncached rather than guess a key).
    """
    source_hash = hashlib.sha256(inspect.getsource(original_func).encode()).hexdigest()[:16]
    sig = inspect.signature(original_func)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _PRECOMPUTE_CACHE_ENABLED:
            # Recorded directly (not via _precompute_cache_lookup_or_compute) so every wrapped
            # function reports a "disabled" entry even when key_parts() would otherwise bail out
            # for lack of a fingerprint (e.g. _detection_probability_fingerprint returns None
            # whenever the cache is disabled, by design -- see that function's own docstring).
            t0 = time.time()
            result = original_func(*args, **kwargs)
            _record_precompute_cache_info(name, {"hit": "disabled", "compute_s": time.time() - t0})
            return result
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        parts = key_parts(dict(bound.arguments))
        if parts is None:
            return original_func(*args, **kwargs)
        key = hashlib.sha256(("|".join([*parts, source_hash])).encode()).hexdigest()[:32]
        return _precompute_cache_lookup_or_compute(
            name, key, lambda: original_func(*args, **kwargs)
        )

    wrapper.__name__ = f"cached_{original_func.__name__}"
    return wrapper


def _sorted_h_repr(h_values: Any) -> str:
    return repr(sorted(float(h) for h in h_values))


def _key_parts_completion_denominator(a: dict[str, Any]) -> list[str] | None:
    fp = _detection_probability_fingerprint(a["detection_probability_obj"])
    if fp is None:
        return None
    completeness = a.get("completeness")
    return [
        _sorted_h_repr(a["h_values"]),
        repr(float(a["Omega_m"])),
        repr(float(a["Omega_DE"])),
        repr(int(a["quad_n"])),
        repr(a["z_max_cap"]),
        fp,
        _completeness_fingerprint(completeness) if completeness is not None else "none",
    ]


def _key_parts_missing_completion_denominator(a: dict[str, Any]) -> list[str] | None:
    fp = _detection_probability_fingerprint(a["detection_probability_obj"])
    if fp is None:
        return None
    return [
        _sorted_h_repr(a["h_values"]),
        repr(int(a["quad_n"])),
        repr(a["z_max_cap"]),
        fp,
        _completeness_fingerprint(a["completeness"]),
    ]


def _key_parts_phi_marginal_survival(a: dict[str, Any]) -> list[str] | None:
    fp = _detection_probability_fingerprint(a["detection_probability_obj"])
    if fp is None:
        return None
    return [
        _sorted_h_repr(a["h_values"]),
        repr(a["z_max_cap"]),
        repr(int(a["n_z"])),
        repr(int(a["n_log10_M"])),
        fp,
    ]


def _key_parts_phi_selection_integrals(a: dict[str, Any]) -> list[str] | None:
    return [
        _sorted_h_repr(a["h_values"]),
        _phi_table_fingerprint(a["phi_survival_table"]),
        _completeness_fingerprint(a["completeness"]),
    ]


def _key_parts_global_catalog_selection(a: dict[str, Any]) -> list[str] | None:
    fp = _detection_probability_fingerprint(a["detection_probability_obj"])
    if fp is None:
        return None
    return [
        _sorted_h_repr(a["h_values"]),
        _catalogue_fingerprint(a["galaxy_catalog"]),
        fp,
        repr(bool(a["with_bh_mass"])),
        repr(a["z_max_cap"]),
        repr(bool(a["smear_sigma_z"])),
        _phi_table_fingerprint(a.get("phi_survival_table")),
        repr(a["sigma4d_mass_kernel"]),
        repr(a["eddington_m"]),
        repr(float(a["theta_b"])),
        repr(float(a["theta_s"])),
    ]


_ORIGINAL_PRECOMPUTE_COMPLETION_DENOMINATOR = bayesian_statistics.precompute_completion_denominator
_ORIGINAL_PRECOMPUTE_MISSING_COMPLETION_DENOMINATOR = (
    bayesian_statistics.precompute_missing_completion_denominator
)
_ORIGINAL_PRECOMPUTE_PHI_MARGINAL_SURVIVAL = bayesian_statistics.precompute_phi_marginal_survival
_ORIGINAL_PRECOMPUTE_PHI_SELECTION_INTEGRALS = (
    bayesian_statistics.precompute_phi_selection_integrals
)
_ORIGINAL_PRECOMPUTE_GLOBAL_CATALOG_SELECTION = (
    bayesian_statistics.precompute_global_catalog_selection
)

# setattr (not `bayesian_statistics.SimulationDetectionProbability = ...`) because mypy flags a
# plain assignment of a callable over a class name as "Cannot assign to a type" ([misc], not
# suppressible via `# type: ignore[assignment]`); setattr on the module object achieves the
# identical runtime effect (this is the module's own mutable __dict__) without that check.
setattr(  # noqa: B010
    bayesian_statistics, "SimulationDetectionProbability", _cached_simulation_detection_probability
)
bayesian_statistics.precompute_completion_denominator = _make_cached_precompute(
    _ORIGINAL_PRECOMPUTE_COMPLETION_DENOMINATOR,
    "precompute_completion_denominator",
    _key_parts_completion_denominator,
)
bayesian_statistics.precompute_missing_completion_denominator = _make_cached_precompute(
    _ORIGINAL_PRECOMPUTE_MISSING_COMPLETION_DENOMINATOR,
    "precompute_missing_completion_denominator",
    _key_parts_missing_completion_denominator,
)
bayesian_statistics.precompute_phi_marginal_survival = _make_cached_precompute(
    _ORIGINAL_PRECOMPUTE_PHI_MARGINAL_SURVIVAL,
    "precompute_phi_marginal_survival",
    _key_parts_phi_marginal_survival,
)
bayesian_statistics.precompute_phi_selection_integrals = _make_cached_precompute(
    _ORIGINAL_PRECOMPUTE_PHI_SELECTION_INTEGRALS,
    "precompute_phi_selection_integrals",
    _key_parts_phi_selection_integrals,
)
bayesian_statistics.precompute_global_catalog_selection = _make_cached_precompute(
    _ORIGINAL_PRECOMPUTE_GLOBAL_CATALOG_SELECTION,
    "precompute_global_catalog_selection",
    _key_parts_global_catalog_selection,
)


# ── per-universe drive + checkpoint ───────────────────────────────────────────


def checkpoint_path(work_root: Path, cell: str, seed: int) -> Path:
    return work_root / f"universe_seed{seed}_{cell}.json"


def run_status_path(work_root: Path, cell: str) -> Path:
    """Per-cell driver-invocation status file (row #288 S4 defect (c) repair).

    Overwritten (not append-only, unlike the per-universe checkpoints) at the end of every
    non-``--score-only`` invocation for this ``cell``, so ``score_only`` can report, per cell,
    whether the LATEST invocation stopped because it exhausted ``--n-universes`` (completion-
    limited) or because it hit ``--max-wall-s`` (wall-limited) -- see :func:`main`'s driver loop
    and B8_2_S3_PILOT_READOUT_RECORD.md caveat 4.2. This file records the FACT of the stop
    reason only; it does not define or apply a stop RULE (that is r-b82-s4 registration content,
    row #290 decisions-table row 3 scope note).
    """
    return work_root / f"_run_status_{cell}.json"


def gridsplit_marker_path(work_root: Path) -> Path:
    """Once-per-work-root marker for :func:`verify_grid_split_bit_identity` (B8.2.S2c item 1).

    Before this stage, ``--verify-split-once`` only tracked "once per PROCESS invocation" (a
    local ``verified_split_this_invocation`` flag in :func:`main`, reset to ``False`` every time
    the script starts) -- correct for a single long-lived run, but the N=106 ladder universe
    (rows #255/#268) paid the full-grid-evaluation-TWICE cost (~2x5,400s) because the
    ``--max-wall-s``-triggered resume re-invoked the script, which re-ran the check from scratch
    on the same work-root even though a PRIOR invocation had already verified the property there.
    A marker file makes "once" mean once per ``--work-root``, across resumes, matching the
    property's own nature (a fact about ``run_mirror_seed_inprocess``/``evaluate()``, not about a
    particular seed or process lifetime -- :func:`verify_grid_split_bit_identity`'s own docstring).
    """
    return work_root / "_gridsplit_check_verified.json"


def _channel_stats(df: pd.DataFrame, grid: npt.NDArray[np.float64], col: str) -> dict[str, Any]:
    """MAP/SD/PIT/HPD(50/68/90/95) for one channel's ``combined_*`` column."""
    d = df[np.isin(df["h"].to_numpy(dtype=np.float64), grid)]
    piv = d.pivot_table(index="event_idx", columns="h", values=col, aggfunc="first")
    piv = piv.reindex(columns=grid)
    vals = piv.to_numpy(dtype=np.float64)
    lnp = combine_log_likelihood(vals, "physics_floor")
    post = trapz_norm(grid, lnp)
    return {
        "ln_post": [float(x) for x in lnp],
        "h_grid": [float(x) for x in grid],
        "map_h": float(grid[int(np.argmax(lnp))]),
        "sd": my_post_sd(grid, post),
        "pit": my_pit(grid, post, H_TRUE),
        "hpd50": my_hpd_contains(grid, post, H_TRUE, 0.50),
        "hpd68": my_hpd_contains(grid, post, H_TRUE, 0.68),
        "hpd90": my_hpd_contains(grid, post, H_TRUE, 0.90),
        "hpd95": my_hpd_contains(grid, post, H_TRUE, 0.95),
        "n_events_scored": int(vals.shape[0]),
    }


def _score_at_truth_by_class(
    df: pd.DataFrame, events: pd.DataFrame, col: str, lo_h: float = 0.725, hi_h: float = 0.735
) -> dict[str, Any]:
    """Per-event secant score at H_TRUE (B4's per_event_scores), split by event_class."""
    piv = df.pivot_table(index="event_idx", columns="h", values=col, aggfunc="first")
    if lo_h not in piv.columns or hi_h not in piv.columns:
        return {"available": False, "reason": f"grid lacks {lo_h}/{hi_h}"}
    idx = piv.index.to_numpy()
    lo = piv[lo_h].to_numpy(dtype=np.float64)
    hi = piv[hi_h].to_numpy(dtype=np.float64)
    ok = (lo > 0.0) & (hi > 0.0)
    score = np.full(idx.shape[0], np.nan)
    score[ok] = (np.log(hi[ok]) - np.log(lo[ok])) / (hi_h - lo_h)
    classes = events.loc[idx, "event_class"].to_numpy()
    out: dict[str, Any] = {"available": True}
    for cls in ("catalogue_hosted", "dark"):
        vals = score[(classes == cls) & np.isfinite(score)]
        out[cls] = {
            "n": int(vals.size),
            "mean": float(vals.mean()) if vals.size else None,
            "sem": float(vals.std(ddof=1) / (vals.size**0.5)) if vals.size > 1 else None,
        }
    vals_all = score[np.isfinite(score)]
    out["all"] = {
        "n": int(vals_all.size),
        "mean": float(vals_all.mean()) if vals_all.size else None,
        "sem": float(vals_all.std(ddof=1) / (vals_all.size**0.5)) if vals_all.size > 1 else None,
    }
    return out


def _run_with_log_capture(
    universe_work: Path,
    events: pd.DataFrame,
    seed: int,
    handler: GalaxyCatalogueHandler,
    h_bounds: tuple[float, float],
    calls: list[tuple[float, ...]],
) -> tuple[Path, dict[str, float], dict[str, Any], Path]:
    """Run 1+ ``run_mirror_seed_inprocess`` calls into ONE work_root, log combined into one file.

    ``calls`` is a list of h-value tuples; each is one ``evaluate()`` call, all with the SAME
    explicit ``h_bounds`` ([P3-HGRID], design §3 item 3) so a multi-call split reproduces one
    whole-grid call bit-for-bit ([P3-HGRID]'s own guarantee, verified once per script invocation
    by :func:`verify_grid_split_bit_identity`). ``event_likelihoods.csv`` accumulates across
    calls (production's own append-mode writer); the log file likewise accumulates, which is what
    lets :func:`parse_candidate_counts` treat a split run exactly like a single whole-grid run.
    """
    universe_work.mkdir(parents=True, exist_ok=True)
    log_path = universe_work / "harness.log"
    resolved: dict[str, Any] = {}
    root_logger = logging.getLogger()
    prev_level = root_logger.level
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    root_logger.addHandler(fh)
    root_logger.setLevel(logging.INFO)
    elapsed: dict[str, float] = {}
    try:
        for i, h_values in enumerate(calls):
            diag_csv, dt = run_mirror_seed_inprocess(
                universe_work,
                events,
                seed,
                handler,
                h_values=h_values,
                h_bounds=h_bounds,
                resolved_flags_out=resolved,
            )
            elapsed[f"call_{i}"] = dt
    finally:
        root_logger.removeHandler(fh)
        fh.close()
        root_logger.setLevel(prev_level)
    return diag_csv, elapsed, resolved, log_path


def verify_grid_split_bit_identity(
    ctx: GenerativeContext,
    work_root: Path,
    seed: int,
    events: pd.DataFrame,
    h_values: tuple[float, ...],
) -> dict[str, Any]:
    """Close verifier must_fix 1 (B8_2_S1_VERIFIER_REPORT.md §3 item 1): the grid-split property.

    Runs ONE whole-grid ``evaluate()`` call and a 2-call split (first half / second half, BOTH
    passing ``h_bounds=(min(h_values), max(h_values))`` explicitly, per design §3's [P3-HGRID]
    requirement) on the SAME event set + seed, and diffs every non-identifier column. This is the
    live test the design lists as required S1 acceptance item (iv) and which neither S1 nor its
    verifier ran; it is run here, once per script invocation (not per universe -- the property is
    a fact about ``run_mirror_seed_inprocess``/``evaluate()``, not about a particular realization).
    """
    if len(h_values) < 2:
        return {"ran": False, "reason": "h_values has < 2 nodes; nothing to split"}
    h_bounds = (min(h_values), max(h_values))
    mid = len(h_values) // 2
    first_half = h_values[:mid]
    second_half = h_values[mid:]

    whole_root = work_root / f"_gridsplit_check_seed{seed}_whole"
    split_root = work_root / f"_gridsplit_check_seed{seed}_split"
    diag_whole, _, resolved_whole, _ = _run_with_log_capture(
        whole_root, events, seed, ctx.handler, h_bounds, [h_values]
    )
    diag_split, _, resolved_split, _ = _run_with_log_capture(
        split_root, events, seed, ctx.handler, h_bounds, [first_half, second_half]
    )
    df_whole = pd.read_csv(diag_whole).sort_values(["event_idx", "h"]).reset_index(drop=True)
    df_split = pd.read_csv(diag_split).sort_values(["event_idx", "h"]).reset_index(drop=True)
    id_cols = {"event_idx", "h"}
    compare_cols = [c for c in df_whole.columns if c not in id_cols]
    max_abs = 0.0
    max_rel = 0.0
    per_column: dict[str, float] = {}
    same_shape = df_whole.shape == df_split.shape
    if same_shape:
        for col in compare_cols:
            a = df_whole[col].to_numpy(dtype=np.float64)
            b = df_split[col].to_numpy(dtype=np.float64)
            diff = np.abs(a - b)
            per_column[col] = float(diff.max()) if diff.size else 0.0
            max_abs = max(max_abs, per_column[col])
            denom = np.maximum(np.abs(a), np.finfo(float).tiny)
            rel = float((diff / denom).max()) if diff.size else 0.0
            max_rel = max(max_rel, rel)
    return {
        "ran": True,
        "same_shape": same_shape,
        "n_rows_whole": int(df_whole.shape[0]),
        "n_rows_split": int(df_split.shape[0]),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "per_column_max_abs_diff": per_column,
        "bit_identical": bool(same_shape and max_abs == 0.0),
        "resolved_flags_whole": resolved_whole,
        "resolved_flags_split": resolved_split,
        "h_bounds": list(h_bounds),
        "first_half": list(first_half),
        "second_half": list(second_half),
    }


def run_one_universe(
    ctx: GenerativeContext,
    work_root: Path,
    seed: int,
    n_draw: int,
    event_cap: int | None,
    cell: str,
    h_values: tuple[float, ...],
    bin_edges: tuple[float, ...],
    verify_split: bool,
) -> dict[str, Any]:
    """Draw + score one mirror universe; returns the full checkpoint dict (not yet written)."""
    gw_scatter = cell == "S"
    gen = MirrorUniverseGenerator(
        CorrespondenceConfig(n_events=n_draw, crb_reference_csv=CRB_CSV_PATH)
    )
    _reset_last_draw_weight_cache_info()
    t_draw = time.time()
    events = gen.draw_realization(
        seed,
        host_pool=ctx.host_pool,
        host_mode="mixture_selected",
        completeness=ctx.completeness,
        phi_survival_table=ctx.phi_survival_table,
        class_weight_p_g=ctx.p_g_info["p_g"],
        gw_scatter=gw_scatter,
    )
    dt_draw = time.time() - t_draw
    draw_weight_cache_info = dict(LAST_DRAW_WEIGHT_CACHE_INFO)
    n_catalogue_hosted = int(events["n_catalogue_hosted"].iloc[0])
    n_realized_draw = int(len(events))
    if event_cap is not None and event_cap < len(events):
        events = events.head(event_cap).reset_index(drop=True)

    universe_work = work_root / f"seed{seed}_{cell}"
    h_bounds = (min(h_values), max(h_values))
    mid = max(1, len(h_values) // 2)
    calls = [h_values[:mid], h_values[mid:]] if len(h_values) > 1 else [h_values]
    calls = [c for c in calls if len(c) > 0]
    _reset_precompute_cache_info()
    diag_csv, elapsed_calls, resolved, log_path = _run_with_log_capture(
        universe_work, events, seed, ctx.handler, h_bounds, calls
    )
    precompute_cache_info = {k: list(v) for k, v in LAST_PRECOMPUTE_CACHE_INFO.items()}
    # draw_realization first (§5's isolated, dominant-when-cold cost), then the evaluate() call(s)
    # (call_0/call_1) -- reported as SEPARATE fields per B8_2_S2_RECORD.md §7 item 1's instruction,
    # not folded into one wall-clock total.
    elapsed = {"draw_realization": dt_draw, **elapsed_calls}
    assert_resolved_production_flags(resolved)

    df = pd.read_csv(diag_csv)
    grid = np.array(sorted(h_values), dtype=np.float64)
    no_bh = _channel_stats(df, grid, "combined_no_bh")
    with_bh = _channel_stats(df, grid, "combined_with_bh")
    score_no_bh = _score_at_truth_by_class(df, events, "combined_no_bh")
    score_with_bh = _score_at_truth_by_class(df, events, "combined_with_bh")

    cand_counts, cand_reason = parse_candidate_counts(log_path, diag_csv)
    n_cand_no_bh = [c[0] for c in cand_counts.values()]
    n_cand_with_bh = [c[1] for c in cand_counts.values()]

    z_true = events["z_true"].to_numpy(dtype=np.float64)
    event_class = events["event_class"].to_numpy()
    edges = np.asarray(bin_edges, dtype=np.float64)
    bin_idx = np.searchsorted(edges, z_true, side="right") - 1
    n_bins = len(edges) - 1
    z_hist = [int(np.sum(bin_idx == b)) for b in range(n_bins)]
    z_hist_cat = [
        int(np.sum((bin_idx == b) & (event_class == "catalogue_hosted"))) for b in range(n_bins)
    ]
    z_hist_dark = [int(np.sum((bin_idx == b) & (event_class == "dark"))) for b in range(n_bins)]
    n_below = int(np.sum(bin_idx < 0))
    n_above = int(np.sum(bin_idx >= n_bins))

    split_check = None
    if verify_split:
        split_check = verify_grid_split_bit_identity(ctx, work_root, seed, events, h_values)

    return {
        "schema_version": "b8_cal_harness_v1",
        "stamp": {**git_stamp(), "role": "B8.2.S2 universe driver"},
        "universe": {
            "seed": seed,
            "cell": cell,
            "gw_scatter": gw_scatter,
            "n_draw_requested": n_draw,
            "n_realized_draw": n_realized_draw,
            "n_scored": int(event_cap) if event_cap is not None else n_realized_draw,
            "n_catalogue_hosted": n_catalogue_hosted,
            "class_weight_p_g": ctx.p_g_info["p_g"],
            "draw_weight_cache": draw_weight_cache_info,
            "precompute_cache": precompute_cache_info,
        },
        "grid": {
            "h_values": list(h_values),
            "h_bounds": list(h_bounds),
            "calls": [list(c) for c in calls],
        },
        "elapsed_s": elapsed,
        "resolved_flags": resolved,
        "posterior": {"no_bh": no_bh, "with_bh": with_bh},
        "score_at_truth": {"no_bh": score_no_bh, "with_bh": score_with_bh},
        "z_true_hist": {
            "bin_edges": list(bin_edges),
            "counts": z_hist,
            "counts_catalogue_hosted": z_hist_cat,
            "counts_dark": z_hist_dark,
            "n_below_lowest_edge": n_below,
            "n_above_highest_edge": n_above,
        },
        "n_pred_by_bin": {
            "bin_edges": list(bin_edges),
            "n_pred_shape": [float(x) for x in ctx.n_pred_by_bin],
            "n_pred_scaled_to_n_draw": [float(x) * n_draw for x in ctx.n_pred_by_bin],
            "self_check": ctx.n_pred_self_check,
        },
        "candidate_census": {
            "log_parse_reason": cand_reason,
            "n_cand_no_bh": n_cand_no_bh,
            "n_cand_with_bh": n_cand_with_bh,
        },
        "grid_split_check": split_check,
    }


# ── score-only aggregator (design §8 S2 deliverable; rule 2: no verdict) ─────


class PopulationMixError(ValueError):
    """Raised by :func:`score_only` when a cell's checkpoints span >1 declared population.

    g-population lint (row #290 decisions-table row 3 / row #288 S4 defect (a)): the aggregator
    must refuse to silently pool rows whose N/population tag (``n_draw_requested``) differs --
    the row #288 contamination pooled 3 N-ladder timing seeds (N=106/400/1588) with 63 N=200
    pilot seeds into one n_universes=66 aggregate. Pass ``population=`` explicitly to select one
    population and exclude the rest (reported in the output's ``excluded_other_population``).
    """


def _population_tag(checkpoint: dict[str, Any]) -> int:
    """The population/N tag of one checkpoint: ``n_draw_requested`` (design §2.1 population unit)."""
    return int(checkpoint["universe"]["n_draw_requested"])


def score_only(work_root: Path, cell: str, population: int | None = None) -> dict[str, Any]:
    """Aggregate every checkpoint JSON under ``work_root`` for ``cell`` (design §4.1).

    ``population`` is the N/population tag (``n_draw_requested``) to aggregate. If omitted and
    the matched checkpoints span more than one population, this function REFUSES (raises
    :class:`PopulationMixError`) rather than silently pooling them -- the g-population lint
    (row #290 decisions-table row 3, repairing row #288 S4 defect (a)). If omitted and exactly
    one population is present, that population is used (the common single-population case is
    unaffected -- g-byte-id).
    """
    files = sorted(work_root.glob(f"universe_seed*_{cell}.json"))
    if not files:
        return {
            "n_universes": 0,
            "cell": cell,
            "population": population,
            "reason": f"no checkpoints found for cell={cell!r} under {work_root}",
        }
    all_checkpoints = [json.loads(f.read_text()) for f in files]
    populations_present = sorted({_population_tag(c) for c in all_checkpoints})

    if population is None:
        if len(populations_present) > 1:
            counts = {
                p: sum(1 for c in all_checkpoints if _population_tag(c) == p)
                for p in populations_present
            }
            raise PopulationMixError(
                f"cell={cell!r} under {work_root} spans {len(populations_present)} populations "
                f"(n_draw_requested -> count: {counts}); pass population=<one of "
                f"{populations_present}> explicitly -- refusing to pool mixed-N rows "
                "(g-population lint, row #288 S4 defect (a))"
            )
        population = populations_present[0] if populations_present else None

    checkpoints = [c for c in all_checkpoints if _population_tag(c) == population]
    excluded = [
        {"file": str(f), "n_draw_requested": _population_tag(c)}
        for f, c in zip(files, all_checkpoints, strict=True)
        if _population_tag(c) != population
    ]
    n_u = len(checkpoints)
    if n_u == 0:
        return {
            "n_universes": 0,
            "cell": cell,
            "population": population,
            "reason": (
                f"no checkpoints for cell={cell!r} population={population!r} under {work_root} "
                f"(populations present: {populations_present})"
            ),
        }

    out: dict[str, Any] = {
        "n_universes": n_u,
        "cell": cell,
        "population": population,
        "populations_present_before_filter": populations_present,
        "excluded_other_population": excluded,
        "files": [
            str(f)
            for f, c in zip(files, all_checkpoints, strict=True)
            if _population_tag(c) == population
        ],
    }
    for channel, key in (("no_bh", "no_bh"), ("with_bh", "with_bh")):
        sds = [c["posterior"][key]["sd"] for c in checkpoints]
        pits = [c["posterior"][key]["pit"] for c in checkpoints]
        maps = [c["posterior"][key]["map_h"] for c in checkpoints]
        n_scored = [c["posterior"][key]["n_events_scored"] for c in checkpoints]
        median_n = float(np.median(n_scored)) if n_scored else 0.0
        sigma_h_harness = float(np.median(sds))
        sigma_floor, floor_prov = sigma_floor_for(channel, max(1, int(round(median_n))))
        f_dilution = sigma_h_harness / sigma_floor if sigma_floor > 0 else float("nan")
        ks_d = my_ks_uniform(pits)
        cov: dict[str, Any] = {}
        for level, key_hpd in ((0.50, "hpd50"), (0.68, "hpd68"), (0.90, "hpd90"), (0.95, "hpd95")):
            hits = sum(1 for c in checkpoints if c["posterior"][key][key_hpd])
            frac = hits / n_u
            band2, band3 = binom_bands(level, n_u)
            cov[key_hpd] = {
                "level": level,
                "hits": hits,
                "n": n_u,
                "fraction": frac,
                "in_2sigma_band": band2[0] <= frac <= band2[1],
                "band_2sigma": list(band2),
            }
        mean_map_bias = float(np.mean(maps)) - H_TRUE
        sem_map = float(np.std(maps, ddof=1) / (n_u**0.5)) if n_u > 1 else float("nan")
        z_map = (
            mean_map_bias / sem_map
            if sem_map and np.isfinite(sem_map) and sem_map > 0
            else float("nan")
        )

        # Score-zero test by class: pooled across ALL events, ALL universes (design §4.1: "mean
        # of ∂_h ln p_i over N x n_U events"). Only per-universe SUMMARY stats (mean_i, sem_i,
        # n_i) are checkpointed (not raw per-event scores, to keep checkpoints small), so this
        # combines independent per-universe estimates via the exact N-weighted pooled mean
        # (identical to pooling the raw per-event data: sum(n_i*mean_i)/sum(n_i)) and the
        # closed-form pooled SEM of that weighted sum, Var = sum((n_i/N_tot)^2 * sem_i^2)
        # (independent-block combination; universes with n_i==1, hence no sample sem_i, are
        # excluded from the SEM/Z leg but still enter the point-estimate mean).
        pooled_n: dict[str, list[float]] = {"catalogue_hosted": [], "dark": [], "all": []}
        pooled_mean: dict[str, list[float]] = {"catalogue_hosted": [], "dark": [], "all": []}
        pooled_sem: dict[str, list[float]] = {"catalogue_hosted": [], "dark": [], "all": []}
        for c in checkpoints:
            s = c["score_at_truth"][key]
            if not s.get("available"):
                continue
            for cls in ("catalogue_hosted", "dark", "all"):
                m = s[cls]["mean"]
                n = s[cls]["n"]
                if m is not None and n:
                    pooled_n[cls].append(float(n))
                    pooled_mean[cls].append(float(m))
                    pooled_sem[cls].append(s[cls]["sem"])
        score_zero: dict[str, Any] = {}
        for cls in ("catalogue_hosted", "dark", "all"):
            ns, means, sems = pooled_n[cls], pooled_mean[cls], pooled_sem[cls]
            if not ns:
                score_zero[cls] = {"n_universes_with_data": 0}
                continue
            n_tot = sum(ns)
            weighted_mean = sum(ni * mi for ni, mi in zip(ns, means, strict=True)) / n_tot
            usable = [(ni, se) for ni, se in zip(ns, sems, strict=True) if se is not None]
            if usable:
                var_pooled = sum((ni / n_tot) ** 2 * se**2 for ni, se in usable)
                sem_pooled = float(var_pooled**0.5)
            else:
                sem_pooled = float("nan")
            z = (
                weighted_mean / sem_pooled
                if sem_pooled and np.isfinite(sem_pooled) and sem_pooled > 0
                else float("nan")
            )
            score_zero[cls] = {
                "n_universes_with_data": len(ns),
                "n_events_pooled": n_tot,
                "mean_pooled": weighted_mean,
                "sem_pooled": sem_pooled,
                "z": z,
                "pass_abs_z_le_3": bool(np.isfinite(z) and abs(z) <= 3.0),
            }

        out[channel] = {
            "sigma_h_harness_median_sd": sigma_h_harness,
            "sigma_floor": sigma_floor,
            "sigma_floor_provenance": floor_prov,
            "F_dilution": f_dilution,
            "pit_ks_d": ks_d,
            "pit_ks_band_informational": 0.134,  # design §4.1, n_U=100 exact critical value
            "coverage": cov,
            "mean_map_minus_h_true": mean_map_bias,
            "sem_map": sem_map,
            "z_map": z_map,
            "score_zero_test_by_class": score_zero,
        }

    # Absolute-count audit (design §1.2(b), item 1: harness-universe instrument test).
    n_bins = len(B3_1_BIN_EDGES) - 1
    totals = [0] * n_bins
    totals_cat = [0] * n_bins
    totals_dark = [0] * n_bins
    n_draw_total = 0
    n_pred_shape = checkpoints[0]["n_pred_by_bin"]["n_pred_shape"]
    self_check = checkpoints[0]["n_pred_by_bin"]["self_check"]
    for c in checkpoints:
        h = c["z_true_hist"]
        for b in range(n_bins):
            totals[b] += h["counts"][b]
            totals_cat[b] += h["counts_catalogue_hosted"][b]
            totals_dark[b] += h["counts_dark"][b]
        n_draw_total += c["universe"]["n_realized_draw"]
    count_audit = []
    for b in range(n_bins):
        n_pred_bin = n_pred_shape[b] * n_draw_total
        n_real_bin = totals[b]
        poisson_se = n_pred_bin**0.5 if n_pred_bin > 0 else float("nan")
        z = (
            (n_real_bin - n_pred_bin) / poisson_se
            if poisson_se and np.isfinite(poisson_se) and poisson_se > 0
            else float("nan")
        )
        count_audit.append(
            {
                "bin_lo": B3_1_BIN_EDGES[b],
                "bin_hi": B3_1_BIN_EDGES[b + 1],
                "n_real": n_real_bin,
                "n_real_catalogue_hosted": totals_cat[b],
                "n_real_dark": totals_dark[b],
                "n_pred": n_pred_bin,
                "poisson_se": poisson_se,
                "z": z,
                "in_3sigma_informational": bool(np.isfinite(z) and abs(z) <= 3.0),
            }
        )
    out["count_audit"] = {
        "n_draw_total": n_draw_total,
        "n_pred_self_check": self_check,
        "per_bin": count_audit,
    }

    # Row #288 S4 defect (c): report wall-limited-vs-completion-limited status explicitly, per
    # cell, from the driver's own run_status sidecar (written by main(), not invented here). No
    # stop RULE is applied -- that is r-b82-s4 registration content (row #290 scope note).
    status_file = run_status_path(work_root, cell)
    if status_file.is_file():
        status = json.loads(status_file.read_text())
        out["run_status"] = {
            "available": True,
            "stopped_reason": status.get("stopped_reason"),
            "wall_limited": status.get("stopped_reason") == "wall_limited",
            "n_universes_requested_this_invocation": status.get(
                "n_universes_requested_this_invocation"
            ),
            "n_done_this_invocation": status.get("n_done_this_invocation"),
            "n_checkpoints_total_under_work_root": status.get(
                "n_checkpoints_total_under_work_root"
            ),
            "max_wall_s": status.get("max_wall_s"),
            "wall_elapsed_s_this_invocation": status.get("wall_elapsed_s_this_invocation"),
            "source_file": str(status_file),
        }
    else:
        out["run_status"] = {
            "available": False,
            "reason": (
                f"no {status_file.name} sidecar found under {work_root} -- either the driver "
                "was never invoked for this cell under this work-root, or these checkpoints "
                "predate the row #288 S4 defect (c) repair"
            ),
        }
    return out


def score_ratio_t_over_s(work_root: Path, population: int | None = None) -> dict[str, Any]:
    """Cell-T / cell-S ratio of ``sigma_h_harness_median_sd``, per channel (row #288 S4 defect
    (b): B8_2_HARNESS_DESIGN_20260829.md line 233 registers this ratio as the S4 input that lets
    the production width comparison (design §4 width branch) be read on the matching (truth-
    centred vs scattered) convention. Cell T carries NO coverage/PIT claim (design §2.3: its PIT
    is degenerate by construction) -- only the SD ratio is registered here; this function does
    not compute or imply a coverage verdict for cell T.
    """
    s = score_only(work_root, "S", population=population)
    t = score_only(work_root, "T", population=population)
    out: dict[str, Any] = {"population": population, "cell_s": s, "cell_t": t, "ratio": {}}
    if s.get("n_universes", 0) == 0 or t.get("n_universes", 0) == 0:
        out["ratio"]["reason"] = (
            f"cannot form T/S ratio: n_universes S={s.get('n_universes', 0)} "
            f"T={t.get('n_universes', 0)} (need > 0 in both)"
        )
        return out
    for channel in ("no_bh", "with_bh"):
        sd_s = s[channel]["sigma_h_harness_median_sd"]
        sd_t = t[channel]["sigma_h_harness_median_sd"]
        out["ratio"][channel] = {
            "sigma_h_harness_median_sd_S": sd_s,
            "sigma_h_harness_median_sd_T": sd_t,
            "T_over_S": sd_t / sd_s if sd_s else float("nan"),
        }
    return out


def print_score_only_report(result: dict[str, Any]) -> None:
    """Print band outcomes for information ONLY -- no verdict is written (design rule 2)."""
    print("=" * 78)
    print(f"B8.2 [CAL] harness -- score-only aggregate (INFORMATIONAL, no verdict; {LAUNCH_STAMP})")
    print(
        f"n_universes = {result.get('n_universes')}  cell = {result.get('cell')}  "
        f"population = {result.get('population')}"
    )
    if result.get("excluded_other_population"):
        print(
            f"  ({len(result['excluded_other_population'])} checkpoint(s) EXCLUDED as a "
            f"different population -- g-population lint, row #288 S4 defect (a))"
        )
    if result.get("n_universes", 0) == 0:
        print(result.get("reason"))
        return
    rs = result.get("run_status")
    if rs is not None:
        if rs.get("available"):
            print(
                f"  run_status: stopped_reason={rs.get('stopped_reason')} "
                f"wall_limited={rs.get('wall_limited')} "
                f"n_done_this_invocation={rs.get('n_done_this_invocation')}/"
                f"{rs.get('n_universes_requested_this_invocation')}"
            )
        else:
            print(f"  run_status: UNAVAILABLE ({rs.get('reason')})")
    for channel in ("no_bh", "with_bh"):
        c = result[channel]
        print("-" * 78)
        print(f"channel = {channel}")
        print(f"  sigma_h,harness (median SD)  = {c['sigma_h_harness_median_sd']:.6g}")
        print(f"  sigma_h,floor (B8.1)         = {c['sigma_floor']:.6g}")
        print(f"  F = SD/floor                 = {c['F_dilution']:.4g}")
        print(
            f"  PIT-KS D                     = {c['pit_ks_d']:.4g}  (informational band: <= 0.134 at n_U=100)"
        )
        for lvl, cov in c["coverage"].items():
            print(
                f"  coverage {lvl} = {cov['fraction']:.3f} ({cov['hits']}/{cov['n']}), "
                f"2sigma band {cov['band_2sigma']}, in_band={cov['in_2sigma_band']}"
            )
        print(f"  mean(MAP) - h_true = {c['mean_map_minus_h_true']:.4g}, Z = {c['z_map']:.3g}")
        for cls, sc in c["score_zero_test_by_class"].items():
            if sc.get("n_universes_with_data"):
                print(
                    f"  score-zero[{cls}]: Z = {sc['z']:.3g}, pass(|Z|<=3) = {sc['pass_abs_z_le_3']}"
                )
    print("-" * 78)
    print("absolute-count audit (n_pred vs n_real, harness-universe instrument test):")
    for row in result["count_audit"]["per_bin"]:
        print(
            f"  z in ({row['bin_lo']:.3f}, {row['bin_hi']:.3f}]: "
            f"n_real={row['n_real']} n_pred={row['n_pred']:.2f} Z={row['z']:.3g} "
            f"in_3sigma={row['in_3sigma_informational']}"
        )
    print("=" * 78)
    print("NOTE: the above are band OUTCOMES for the chair's/S4's own read -- this script does")
    print("NOT emit a PASS/FAIL verdict (design §8 S2 acceptance, rule 2).")


# ── CLI ────────────────────────────────────────────────────────────────────


def _parse_h_values(spec: str) -> tuple[float, ...]:
    return tuple(sorted(float(x) for x in spec.split(",") if x.strip()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--work-root", type=Path, default=THIS_DIR / "b8_cal_harness_work")
    parser.add_argument("--n-universes", type=int, default=2)
    parser.add_argument("--N", dest="n_draw", type=int, default=200)
    parser.add_argument("--event-cap", type=int, default=None)
    parser.add_argument("--cell", choices=["S", "T"], default="S")
    parser.add_argument("--h-values", type=str, default=",".join(str(h) for h in H_GRID_41))
    parser.add_argument("--seed-block", type=int, default=900200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-wall-s", type=float, default=560.0)
    parser.add_argument(
        "--verify-split-once",
        action="store_true",
        default=True,
        help="run the grid-split bit-identity check once (on the first NEW universe this "
        "invocation scores) -- default on; pass --no-verify-split-once to skip",
    )
    parser.add_argument("--no-verify-split-once", dest="verify_split_once", action="store_false")
    parser.add_argument(
        "--force-gridsplit-check",
        action="store_true",
        default=False,
        help="run the grid-split bit-identity check even if a PRIOR invocation already wrote "
        "the once-per-work-root marker (B8.2.S2c item 1, gridsplit_marker_path()) under "
        "--work-root -- use to re-verify after touching run_mirror_seed_inprocess/evaluate()",
    )
    parser.add_argument(
        "--no-draw-weight-cache",
        dest="draw_weight_cache",
        action="store_false",
        default=True,
        help="disable the catalogue_selected_host_draw_weights cache (B8.2.S2b) -- recompute "
        "the >318s-dominant host-draw weights fresh every universe; use only for the §8 "
        "byte-identity comparison against a cached run (B8_2_S2_RECORD.md §8)",
    )
    parser.add_argument(
        "--no-precompute-cache",
        dest="precompute_cache",
        action="store_false",
        default=True,
        help="disable the B8.2.S2c per-h precompute cache (SimulationDetectionProbability + the "
        "five precompute_* functions) -- recompute every one fresh every evaluate() call; use "
        "only for the §10 byte-identity comparison against a cached run",
    )
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="N/population tag (n_draw_requested) to aggregate in --score-only mode; required "
        "if the matched checkpoints span more than one population (g-population lint, row #288 "
        "S4 defect (a)) -- omit when exactly one population is present under --work-root.",
    )
    parser.add_argument(
        "--score-only-ratio-t-s",
        action="store_true",
        help="in --score-only mode, also compute+print the cell-T / cell-S SD ratio (row #288 "
        "S4 defect (b), the T0/T-vs-S control read registered by "
        "B8_2_HARNESS_DESIGN_20260829.md line 233) instead of a single-cell report.",
    )
    args = parser.parse_args()

    args.work_root.mkdir(parents=True, exist_ok=True)
    configure_draw_weight_cache(args.work_root, enabled=args.draw_weight_cache)
    configure_precompute_cache(args.work_root, enabled=args.precompute_cache)

    if args.score_only:
        if args.score_only_ratio_t_s:
            ratio_result = score_ratio_t_over_s(args.work_root, population=args.population)
            print_score_only_report(ratio_result["cell_s"])
            print_score_only_report(ratio_result["cell_t"])
            print("=" * 78)
            print("T / S sigma_h,harness (median SD) ratio (design line 233 control read):")
            for channel, r in ratio_result["ratio"].items():
                if channel == "reason":
                    print(f"  {ratio_result['ratio']['reason']}")
                    break
                print(
                    f"  {channel}: S={r['sigma_h_harness_median_sd_S']:.6g} "
                    f"T={r['sigma_h_harness_median_sd_T']:.6g} T/S={r['T_over_S']:.4g}"
                )
            return 0
        try:
            result = score_only(args.work_root, args.cell, population=args.population)
        except PopulationMixError as exc:
            print(f"g-population lint REFUSAL: {exc}")
            return 1
        print_score_only_report(result)
        return 0

    h_values = _parse_h_values(args.h_values)
    affinity_info = pin_affinity(args.workers)
    _LOGGER.info("affinity: %s", affinity_info)

    t_start = time.time()
    ctx = build_generative_context()
    _LOGGER.info(
        "generative context built in %.1fs; p_g=%.6g", time.time() - t_start, ctx.p_g_info["p_g"]
    )
    print(f"[{LAUNCH_STAMP}] generative context built in {time.time() - t_start:.1f}s")
    print(f"p_g = {ctx.p_g_info['p_g']:.7g} (D_tilde_phi={ctx.p_g_info['D_tilde_phi']:.7g})")
    print(f"n_pred self-check: {ctx.n_pred_self_check}")

    marker_path = gridsplit_marker_path(args.work_root)
    verified_split_this_invocation = False
    if args.verify_split_once and marker_path.is_file() and not args.force_gridsplit_check:
        verified_split_this_invocation = True
        print(
            f"grid-split bit-identity check: marker {marker_path} already present (verified "
            "in a prior invocation under this --work-root) -- skipping; pass "
            "--force-gridsplit-check to re-run"
        )
    n_done = 0
    stopped_reason = "exhausted_n_universes"  # overwritten below if the wall-limit fires first
    for i in range(args.n_universes):
        if time.time() - t_start > args.max_wall_s:
            stopped_reason = "wall_limited"
            print(
                f"--max-wall-s ({args.max_wall_s}s) reached after {n_done} universe(s) this "
                "invocation; re-run the same command to resume (checkpoints already written "
                "are skipped)."
            )
            break
        seed = args.seed_block + i
        ckpt_file = checkpoint_path(args.work_root, args.cell, seed)
        if ckpt_file.is_file():
            print(f"seed {seed} cell {args.cell}: checkpoint already exists, skipping")
            continue
        do_split_check = args.verify_split_once and not verified_split_this_invocation
        t_u = time.time()
        checkpoint = run_one_universe(
            ctx,
            args.work_root,
            seed,
            args.n_draw,
            args.event_cap,
            args.cell,
            h_values,
            B3_1_BIN_EDGES,
            verify_split=do_split_check,
        )
        if do_split_check:
            verified_split_this_invocation = True
            sc = checkpoint["grid_split_check"]
            print(f"grid-split bit-identity check: bit_identical={sc.get('bit_identical')}")
            marker_path.write_text(
                json.dumps(
                    {
                        "verified_at_seed": seed,
                        "cell": args.cell,
                        "bit_identical": sc.get("bit_identical"),
                        "max_abs_diff": sc.get("max_abs_diff"),
                        "stamp": git_stamp(),
                        "note": (
                            "once-per-work-root marker (B8.2.S2c item 1) -- delete this file or "
                            "pass --force-gridsplit-check to re-run the check"
                        ),
                    },
                    indent=1,
                )
            )
        ckpt_file.write_text(json.dumps(checkpoint, indent=1))
        n_done += 1
        print(
            f"seed {seed} cell {args.cell}: done in {time.time() - t_u:.1f}s -> {ckpt_file} "
            f"(n_scored no_bh={checkpoint['posterior']['no_bh']['n_events_scored']}, "
            f"n_catalogue_hosted={checkpoint['universe']['n_catalogue_hosted']})"
        )
        print(
            f"  elapsed_s: draw_realization={checkpoint['elapsed_s'].get('draw_realization'):.1f}s "
            f"(cache: {checkpoint['universe']['draw_weight_cache'].get('hit')}), "
            + ", ".join(
                f"{k}={v:.1f}s"
                for k, v in checkpoint["elapsed_s"].items()
                if k != "draw_realization"
            )
        )
        pc_hits = {
            name: [call.get("hit") for call in calls]
            for name, calls in checkpoint["universe"]["precompute_cache"].items()
        }
        print(f"  precompute_cache hits: {pc_hits}")
    wall_elapsed = time.time() - t_start
    print(f"total wall this invocation: {wall_elapsed:.1f}s; {n_done} universe(s) scored")

    n_checkpoints_total = len(list(args.work_root.glob(f"universe_seed*_{args.cell}.json")))
    run_status_path(args.work_root, args.cell).write_text(
        json.dumps(
            {
                "cell": args.cell,
                "stamp": git_stamp(),
                "seed_block": args.seed_block,
                "n_universes_requested_this_invocation": args.n_universes,
                "n_done_this_invocation": n_done,
                "n_checkpoints_total_under_work_root": n_checkpoints_total,
                "max_wall_s": args.max_wall_s,
                "wall_elapsed_s_this_invocation": wall_elapsed,
                "stopped_reason": stopped_reason,
                "note": (
                    "FACT only, not a stop RULE (row #290 decisions-table row 3 scope note: "
                    "the stop rule itself is r-b82-s4 registration content). Overwritten by "
                    "every non-score-only invocation for this cell under this --work-root."
                ),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
