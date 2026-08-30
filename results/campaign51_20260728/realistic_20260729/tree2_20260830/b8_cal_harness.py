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
import re
import subprocess
import sys
import time
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
from darksiren_emri.galaxy_catalogue.pixel_completeness import CompletenessModel  # noqa: E402
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


# ── per-universe drive + checkpoint ───────────────────────────────────────────


def checkpoint_path(work_root: Path, cell: str, seed: int) -> Path:
    return work_root / f"universe_seed{seed}_{cell}.json"


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
    diag_csv, elapsed_calls, resolved, log_path = _run_with_log_capture(
        universe_work, events, seed, ctx.handler, h_bounds, calls
    )
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


def score_only(work_root: Path, cell: str) -> dict[str, Any]:
    """Aggregate every checkpoint JSON under ``work_root`` for ``cell`` (design §4.1)."""
    files = sorted(work_root.glob(f"universe_seed*_{cell}.json"))
    if not files:
        return {
            "n_universes": 0,
            "cell": cell,
            "reason": f"no checkpoints found for cell={cell!r} under {work_root}",
        }
    checkpoints = [json.loads(f.read_text()) for f in files]
    n_u = len(checkpoints)

    out: dict[str, Any] = {"n_universes": n_u, "cell": cell, "files": [str(f) for f in files]}
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
    return out


def print_score_only_report(result: dict[str, Any]) -> None:
    """Print band outcomes for information ONLY -- no verdict is written (design rule 2)."""
    print("=" * 78)
    print(f"B8.2 [CAL] harness -- score-only aggregate (INFORMATIONAL, no verdict; {LAUNCH_STAMP})")
    print(f"n_universes = {result.get('n_universes')}  cell = {result.get('cell')}")
    if result.get("n_universes", 0) == 0:
        print(result.get("reason"))
        return
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
        "--no-draw-weight-cache",
        dest="draw_weight_cache",
        action="store_false",
        default=True,
        help="disable the catalogue_selected_host_draw_weights cache (B8.2.S2b) -- recompute "
        "the >318s-dominant host-draw weights fresh every universe; use only for the §8 "
        "byte-identity comparison against a cached run (B8_2_S2_RECORD.md §8)",
    )
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    args.work_root.mkdir(parents=True, exist_ok=True)
    configure_draw_weight_cache(args.work_root, enabled=args.draw_weight_cache)

    if args.score_only:
        result = score_only(args.work_root, args.cell)
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

    verified_split_this_invocation = False
    n_done = 0
    for i in range(args.n_universes):
        if time.time() - t_start > args.max_wall_s:
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
    print(f"total wall this invocation: {time.time() - t_start:.1f}s; {n_done} universe(s) scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
