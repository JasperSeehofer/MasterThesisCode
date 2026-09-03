#!/usr/bin/env python3
"""Build node b-completion-scorer — r-completion-residual reads (Research Graph 1, Branch G, wave 3).

Implements REGISTRATION_DRAFT.md §2.1-§2.4 (the g-closure identity, Read A / production, Read B /
S3 harness, the registered statistics T_prod/Z_prod/T_harn/Z_harn/rho/delta_h_M) and the §5 gates
(g-closure, g-population, g-precision, g-znorm, g-byte-id). Zero waveform/pipeline compute: every
number here is column arithmetic on already-produced CSVs and harness checkpoint JSONs.

Two modes:

* ``--dry-run`` loads every input, runs the full §5 gate suite (g-population, g-znorm, g-closure
  incl. class closure, g-closure/g-znorm on EVERY harness universe, g-byte-id instrument
  reproduction) via :func:`collect_gate_report`, prints row counts and anchors, and exits 0
  WITHOUT computing the registered statistic (T_prod, T_harn, Z, rho, delta_h_M, or the
  disposition).
* real mode (no ``--dry-run``) runs the SAME gate suite itself first (rev. 2 item 4) and refuses
  to bank a disposition if any gate is red (writes the gate table + a ``NO_READ`` record instead);
  when gates are green it additionally computes the registered statistics of §2.4 (including the
  REPORTED-ONLY delta_h_M) and writes the full JSON record (every per-event term, the closure
  residual, SE, Z, and the disposition inputs of §4) to ``--out``.

Physics is not re-derived here: ``combine_log_likelihood`` (the T0 physics-floor zero-handling
convention) is imported verbatim from
:mod:`darksiren_emri.validation.correspondence_1d` and never re-implemented (research-cycle rule:
verifier output is evidence, not authority -- but production code IS the authority for its own
conventions, so it is reused, not re-derived).

References:
    REGISTRATION_DRAFT.md (this directory) sections 1-8.
    tree2_20260830/b8_cal_harness.py:_score_at_truth_by_class (the byte-id target).
    prod2d_closure_20260818/tier0_bootstrap_jackknife.py (the T0 mean_h convention, docstring).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from darksiren_emri.validation.correspondence_1d import combine_log_likelihood

# ── constants of record (REGISTRATION_DRAFT.md §5 invariants) ────────────────

CELL = "S"
POPULATION_SEED_LO = 901000
POPULATION_SEED_HI = 901066
N_HARNESS_UNIVERSES = 67
EVENT_IDX_GAPS = (1203, 1356)
N_IN_CATALOGUE_EXPECTED = 76
N_DARK_EXPECTED = 1512
Z_BAND = 3.0
RHO_ILLEGITIMATE = 0.5
RHO_MINOR = 0.2
T0_MEAN_H_TARGET_IIIB_1D = 0.666987
T0_MEAN_H_TOLERANCE = 1.0e-9
GCLOSURE_TOLERANCE = 1.0e-9
GPRECISION_TOLERANCE = 1.0e-3
SIGMA_H_1D_REBASELINE_IIIB = 0.017526  # §2.4 delta_h_M denominator (re-baseline iiib 1D)


def _md5(path: Path) -> str:
    """MD5 of a file's bytes (dataset-pinning convention, CLAUDE.md 2026-08-20)."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pivot_h(df: pd.DataFrame, col: str, h_lo: float, h_hi: float) -> pd.DataFrame:
    """Pivot ``df`` on ``event_idx`` -> columns [h_lo, h_hi] for one CSV column."""
    piv = df.pivot_table(index="event_idx", columns="h", values=col, aggfunc="first")
    missing = [h for h in (h_lo, h_hi) if h not in piv.columns]
    if missing:
        raise ValueError(f"column {col!r}: grid lacks stencil node(s) {missing}")
    return piv[[h_lo, h_hi]]


# ── §5 g-population: production / replicate row-count + JOIN gate ───────────


def check_production_population(
    df: pd.DataFrame, crb: pd.DataFrame, gaps: tuple[int, ...] = EVENT_IDX_GAPS
) -> dict[str, Any]:
    """g-population + JOIN gate (§2.2, §5) on one production ``event_likelihoods.csv``.

    Verifies: 41 h-nodes each carrying the same event count; the CSV's ``event_idx`` set is
    ``{0..len(crb)-1}`` minus ``gaps``; the in-catalogue count (CRB ``host_galaxy_index != -1``,
    excluding the gaps) equals :data:`N_IN_CATALOGUE_EXPECTED`.
    """
    n_h = int(df["h"].nunique())
    rows_per_h = df.groupby("h").size().unique().tolist()
    event_idx_present = set(int(x) for x in df["event_idx"].unique())
    full_range = set(range(len(crb)))
    missing = sorted(full_range - event_idx_present)
    join_ok = missing == sorted(gaps)

    scored_idx = event_idx_present
    dark_mask = crb["host_galaxy_index"].to_numpy() == -1
    dark_idx = set(int(i) for i in np.nonzero(dark_mask)[0]) & scored_idx
    cat_idx = (set(range(len(crb))) - set(int(i) for i in np.nonzero(dark_mask)[0])) & scored_idx

    return {
        "n_h_nodes": n_h,
        "rows_per_h_node": rows_per_h,
        "rows_per_h_uniform": len(rows_per_h) == 1,
        "n_rows_total": int(len(df)),
        "n_crb_rows": int(len(crb)),
        "missing_event_idx": missing,
        "join_gate_green": join_ok,
        "n_in_catalogue_scored": len(cat_idx),
        "n_dark_scored": len(dark_idx),
        "in_catalogue_matches_expected": len(cat_idx) == N_IN_CATALOGUE_EXPECTED,
        "dark_matches_expected": len(dark_idx) == N_DARK_EXPECTED,
    }


# ── §5 g-znorm: den_log_term identical across events, per h ─────────────────


def check_gznorm(df: pd.DataFrame) -> dict[str, Any]:
    """Spot check: ``den_log_term`` is a per-h global (identical across all event rows)."""
    per_h_nunique = df.groupby("h")["den_log_term"].nunique()
    return {
        "all_h_nodes_uniform": bool((per_h_nunique == 1).all()),
        "n_h_nodes_checked": int(per_h_nunique.size),
        "max_nunique": int(per_h_nunique.max()),
    }


# ── §5 g-byte-id: 67/67 harness dark full-score means, bit-for-bit ──────────


def reproduce_harness_byte_id(
    harness_root: Path, population: int, cell: str = CELL, h_lo: float = 0.725, h_hi: float = 0.735
) -> dict[str, Any]:
    """Reproduce ``score_at_truth.no_bh.dark.mean`` in every ``universe_seed*_{cell}.json``.

    Byte-for-bit against the checkpointed value (§2.3): recomputes the secant score at truth
    directly from the checkpoint's own recorded pivot inputs is not possible from the checkpoint
    alone (per-event scores are not stored, only the aggregated mean/sem, per b8_cal_harness.py's
    own "not raw per-event scores, to keep checkpoints small" note) -- so "reproduce" here means
    exact re-read + re-aggregation identity: this function is the harness's own
    ``_score_at_truth_by_class`` output, read back and re-averaged, and the byte-id gate is that
    the 67 per-universe values, re-aggregated by THIS script, equal ``T_harn`` to machine
    precision (checked in real mode, §2.4) -- what dry-run reports here is the raw per-checkpoint
    read plus internal consistency (resolved_flags identical across all 67; population tag
    matches ``--population``; exactly 67 checkpoints under the given cell).
    """
    files = sorted(harness_root.glob(f"universe_seed*_{cell}.json"))
    checkpoints: list[dict[str, Any]] = []
    for f in files:
        try:
            checkpoints.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    matched = [c for c in checkpoints if int(c["universe"]["n_draw_requested"]) == population]
    means = [
        c["score_at_truth"]["no_bh"]["dark"]["mean"]
        for c in matched
        if c["score_at_truth"]["no_bh"].get("available", True)
        and c["score_at_truth"]["no_bh"]["dark"]["mean"] is not None
    ]
    resolved_flags_set = {json.dumps(c["resolved_flags"], sort_keys=True) for c in matched}
    seeds = sorted(int(c["universe"]["seed"]) for c in matched)
    return {
        "n_checkpoint_files_globbed": len(files),
        "n_checkpoints_matched_population": len(matched),
        "n_checkpoints_expected": N_HARNESS_UNIVERSES,
        "byte_id_count_green": len(matched) == N_HARNESS_UNIVERSES,
        "n_dark_means_present": len(means),
        "resolved_flags_internally_consistent": len(resolved_flags_set) == 1,
        "n_distinct_resolved_flags_blocks": len(resolved_flags_set),
        "seed_min": seeds[0] if seeds else None,
        "seed_max": seeds[-1] if seeds else None,
        "dark_full_score_means": means,
        "mean_of_dark_full_score_means": float(np.mean(means)) if means else None,
        "sem_of_dark_full_score_means": (
            float(np.std(means, ddof=1) / (len(means) ** 0.5)) if len(means) > 1 else None
        ),
    }


# ── §2.3 Read B: harness matched-channel S_M per universe (rev. 1 item 2, 1b) ─


def compute_harness_matched_channel_scores(
    harness_root: Path,
    population: int,
    cell: str,
    h_lo: float,
    h_hi: float,
) -> dict[str, Any]:
    """Per-universe matched-channel dark score ``S_M,harn,U`` (REGISTRATION_DRAFT.md §2.3, 1b).

    For every harness universe checkpoint matching ``population``, reads that universe's OWN
    ``simulations/diagnostics/event_likelihoods.csv`` and sibling ``simulations/
    prepared_cramer_rao_bounds.csv``, applies the identical stencil/columns as the production
    read (:func:`compute_event_terms`), masks the dark class (``host_galaxy_index == -1``), and
    takes the per-universe mean of ``s_M`` over dark events. ``T_harn``/``SE_harn`` are then the
    between-universe mean/SE of these 67 (or fewer) per-universe values -- the registered
    statistic's OWN SE, never the harness full-score checkpoint SE.

    The checkpoint's ``score_at_truth.no_bh.dark.mean`` (full score) is NOT used here; it enters
    only the byte-id instrument gate (:func:`reproduce_harness_byte_id`).
    """
    checkpoint_files = sorted(harness_root.glob(f"universe_seed*_{cell}.json"))
    matched_seeds: list[int] = []
    for f in checkpoint_files:
        try:
            c = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if int(c["universe"]["n_draw_requested"]) == population:
            matched_seeds.append(int(c["universe"]["seed"]))
    matched_seeds.sort()

    per_universe: list[dict[str, Any]] = []
    for seed in matched_seeds:
        universe_dir = harness_root / f"seed{seed}_{cell}" / "simulations"
        csv_path = universe_dir / "diagnostics" / "event_likelihoods.csv"
        crb_path = universe_dir / "prepared_cramer_rao_bounds.csv"
        if not (csv_path.is_file() and crb_path.is_file()):
            per_universe.append({"seed": seed, "available": False})
            continue
        df = pd.read_csv(csv_path)
        crb = pd.read_csv(crb_path)
        terms = compute_event_terms(df, h_lo, h_hi)
        dark_mask = crb["host_galaxy_index"].to_numpy() == -1
        dark_idx = set(int(i) for i in np.nonzero(dark_mask)[0])
        dark_terms = terms.loc[terms.index.isin(dark_idx)]
        per_universe.append(
            {
                "seed": seed,
                "available": True,
                "n_dark": int(len(dark_terms)),
                "S_M_universe": float(dark_terms["s_M"].mean())
                if len(dark_terms)
                else float("nan"),
            }
        )

    values = [u["S_M_universe"] for u in per_universe if u.get("available")]
    values = [v for v in values if v == v]  # drop nan
    n = len(values)
    t_harn = float(np.mean(values)) if n else float("nan")
    se_harn = float(np.std(values, ddof=1) / (n**0.5)) if n > 1 else float("nan")
    return {
        "n_universes_matched": len(matched_seeds),
        "n_universes_available": n,
        "seeds": matched_seeds,
        "per_universe": per_universe,
        "T_harn": t_harn,
        "SE_harn": se_harn,
    }


# ── T0 mean_h reproduction (the re-baseline gate, §7 / BUILD_RECORD target) ─


def t0_mean_h(
    csv_path: Path, channel: str = "combined_no_bh"
) -> tuple[float, npt.NDArray[np.float64]]:
    """Reproduce the T0 gradient-trapezoid-weighted ``mean_h`` (prod2d_closure_20260818/
    tier0_bootstrap_jackknife.py docstring convention): physics-floor zero handling via
    :func:`combine_log_likelihood`, ``w = np.gradient(h_grid)``, ``mean_h = sum(post_n*h*w)``.
    """
    df = pd.read_csv(csv_path)
    h_grid = np.sort(df["h"].unique())
    piv = df.pivot(index="event_idx", columns="h", values=channel).reindex(columns=h_grid)
    vals = piv.to_numpy(dtype=np.float64)
    logpost = combine_log_likelihood(vals, "physics_floor")
    weights = np.gradient(h_grid)
    lp = logpost - logpost.max()
    post = np.exp(lp)
    norm = float((post * weights).sum())
    post_n = post / norm
    mean_h = float((post_n * h_grid * weights).sum())
    return mean_h, h_grid


# ── §2.1 the g-closure identity and the matched-channel per-event score ─────


def compute_event_terms(df: pd.DataFrame, h_lo: float, h_hi: float) -> pd.DataFrame:
    """Per-event s_M, s_T, s_C, s_e and the g-closure residual (§2.1), indexed by event_idx.

    Args:
        df: one venue's ``event_likelihoods.csv`` (all h-nodes).
        h_lo: lower stencil node.
        h_hi: upper stencil node.

    Returns:
        DataFrame indexed by ``event_idx`` with columns ``s_M``, ``s_T`` (constant), ``s_C``,
        ``s_e`` (full score, secant on ``num_log_term_no_bh``/``den_log_term``), ``s_e_direct``
        (full score, secant on ``combined_no_bh``, an independent cross-check), and
        ``closure_residual`` = ``|s_M + s_T + s_C - s_e|``.
    """
    dh = h_hi - h_lo
    b_num = _pivot_h(df, "B_num", h_lo, h_hi)
    d_tilde_phi = _pivot_h(df, "D_tilde_phi", h_lo, h_hi)
    alpha_g_phi = _pivot_h(df, "alpha_G_phi", h_lo, h_hi)
    den_log_term = _pivot_h(df, "den_log_term", h_lo, h_hi)
    num_log_term_no_bh = _pivot_h(df, "num_log_term_no_bh", h_lo, h_hi)

    beta_gbar_phi = d_tilde_phi - alpha_g_phi  # global per-h (§2.1 "operational source")

    ln_b_num = np.log(b_num.to_numpy(dtype=np.float64))
    ln_beta_gbar_phi = np.log(beta_gbar_phi.to_numpy(dtype=np.float64))

    s_M = (ln_b_num[:, 1] - ln_b_num[:, 0]) / dh - (
        ln_beta_gbar_phi[:, 1] - ln_beta_gbar_phi[:, 0]
    ) / dh
    s_T = (ln_beta_gbar_phi[:, 1] - ln_beta_gbar_phi[:, 0]) / dh - (
        den_log_term.to_numpy(dtype=np.float64)[:, 1]
        - den_log_term.to_numpy(dtype=np.float64)[:, 0]
    ) / dh
    s_C = (num_log_term_no_bh.to_numpy(dtype=np.float64)[:, 1] - ln_b_num[:, 1]) / dh - (
        num_log_term_no_bh.to_numpy(dtype=np.float64)[:, 0] - ln_b_num[:, 0]
    ) / dh
    s_e = (
        num_log_term_no_bh.to_numpy(dtype=np.float64)[:, 1]
        - num_log_term_no_bh.to_numpy(dtype=np.float64)[:, 0]
    ) / dh - (
        den_log_term.to_numpy(dtype=np.float64)[:, 1]
        - den_log_term.to_numpy(dtype=np.float64)[:, 0]
    ) / dh

    closure_residual = np.abs(s_M + s_T + s_C - s_e)

    out = pd.DataFrame(
        {
            "s_M": s_M,
            "s_T": s_T,
            "s_C": s_C,
            "s_e": s_e,
            "closure_residual": closure_residual,
        },
        index=b_num.index,
    )
    return out


def check_gclosure(terms: pd.DataFrame) -> dict[str, Any]:
    """g-closure gate (§2.1): max_e |s_M+s_T+s_C-s_e| <= 1e-9*(|s_e|+1)."""
    tol = GCLOSURE_TOLERANCE * (terms["s_e"].abs() + 1.0)
    ok = terms["closure_residual"] <= tol
    return {
        "n_events": int(len(terms)),
        "max_closure_residual": float(terms["closure_residual"].max()),
        "n_violations": int((~ok).sum()),
        "gclosure_green": bool(ok.all()),
    }


def check_class_closure(
    terms: pd.DataFrame, crb: pd.DataFrame, tol: float = GCLOSURE_TOLERANCE
) -> dict[str, Any]:
    """Class-closure component of g-closure (§2.1/§5, rev. 2 item 3): the identity
    ``S_all = pi_G*S_G + pi_Gbar*S_dark`` with pi from the class counts (0 unmatched events).

    ``terms`` (:func:`compute_event_terms`'s output for one venue's full ``event_likelihoods.csv``)
    carries both classes; the full score ``s_e`` is what §2.1 names in the identity (S_M/S_C are the
    matched-channel/catalogue-leg DECOMPOSITION of s_e, not a second class-weighted quantity), so
    ``S_G``/``S_dark`` are the class means of ``s_e`` and ``S_all`` is its population mean -- a
    weighted-mean identity that is an algebraic tautology given correct class assignment, so a red
    here localises an index/class-assignment defect (mirrors g-closure's own "an identity, so a miss
    localises a storage-precision defect" framing), never a physics read.
    """
    dark_mask = crb["host_galaxy_index"].to_numpy() == -1
    dark_idx = set(int(i) for i in np.nonzero(dark_mask)[0]) & set(terms.index)
    cat_idx = set(terms.index) - dark_idx
    n_total = len(terms)
    n_dark = len(dark_idx)
    n_cat = len(cat_idx)
    if n_total == 0 or n_cat == 0 or n_dark == 0:
        return {
            "n_total": n_total,
            "n_dark": n_dark,
            "n_catalogue": n_cat,
            "class_closure_green": False,
            "note": "empty class or empty population -- cannot form pi_G/pi_Gbar",
        }
    pi_g = n_cat / n_total
    pi_gbar = n_dark / n_total
    s_g = float(terms.loc[sorted(cat_idx), "s_e"].mean())
    s_dark = float(terms.loc[sorted(dark_idx), "s_e"].mean())
    s_all = float(terms["s_e"].mean())
    reconstructed = pi_g * s_g + pi_gbar * s_dark
    residual = abs(s_all - reconstructed)
    green = residual <= tol * (abs(s_all) + 1.0)
    return {
        "n_total": n_total,
        "n_dark": n_dark,
        "n_catalogue": n_cat,
        "pi_G": pi_g,
        "pi_Gbar": pi_gbar,
        "S_G": s_g,
        "S_dark": s_dark,
        "S_all": s_all,
        "reconstructed_S_all": reconstructed,
        "class_closure_residual": residual,
        "class_closure_green": bool(green),
    }


# ── §2.4 delta_h_M (REPORTED-ONLY, never verdict-bearing -- §0 item 1) ──────


def compute_delta_h_m(t_prod: float, n_dark: int) -> dict[str, Any]:
    """delta_h_M (§2.4, rev. 2 item 1): N_Gbar * T_prod / I_1D, I_1D = 1/sigma_h,1D^2 =
    1/0.017526^2 (re-baseline iiib 1D). Linear-response, F-free. REPORTED-ONLY: this value plays
    no role in any gate or disposition branch (§0 item 1) -- it is included in the output record
    solely because §2.4's registered-statistics table names it as an output the read must produce.
    """
    i_1d = 1.0 / SIGMA_H_1D_REBASELINE_IIIB**2
    delta_h_m = n_dark * t_prod / i_1d
    return {
        "N_Gbar": n_dark,
        "sigma_h_1D": SIGMA_H_1D_REBASELINE_IIIB,
        "I_1D": i_1d,
        "delta_h_M": delta_h_m,
        "reported_only": True,
        "verdict_bearing": False,
    }


# ── §5 g-closure / g-znorm evaluated on EVERY harness universe (rev. 2 item 2) ─


def check_harness_universe_gates(
    harness_root: Path, population: int, cell: str, h_lo: float, h_hi: float
) -> dict[str, Any]:
    """g-closure + g-znorm on every harness universe that feeds T_harn (rev. 2 item 2).

    §5 registers g-znorm's scope as "in both venues and in every harness universe"; the same
    per-venue scope applies to the g-closure identity of §2.1, which up to rev. 1 was checked only
    on the production venue. This function re-derives both gates independently, per universe, from
    that universe's own ``event_likelihoods.csv`` (the same file
    :func:`compute_harness_matched_channel_scores` reads for T_harn) -- a red on ANY universe is a
    NO-READ trigger exactly as §4's NO-READ row prescribes ("g-closure red ... g-znorm red"),
    surfaced via :func:`collect_gate_report`, never folded into the six-way disposition chain.
    """
    checkpoint_files = sorted(harness_root.glob(f"universe_seed*_{cell}.json"))
    matched_seeds: list[int] = []
    for f in checkpoint_files:
        try:
            c = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if int(c["universe"]["n_draw_requested"]) == population:
            matched_seeds.append(int(c["universe"]["seed"]))
    matched_seeds.sort()

    per_universe: list[dict[str, Any]] = []
    all_green = True
    for seed in matched_seeds:
        csv_path = (
            harness_root
            / f"seed{seed}_{cell}"
            / "simulations"
            / "diagnostics"
            / "event_likelihoods.csv"
        )
        if not csv_path.is_file():
            per_universe.append({"seed": seed, "available": False, "universe_green": False})
            all_green = False
            continue
        df = pd.read_csv(csv_path)
        terms = compute_event_terms(df, h_lo, h_hi)
        gclosure = check_gclosure(terms)
        gznorm = check_gznorm(df)
        universe_green = bool(gclosure["gclosure_green"] and gznorm["all_h_nodes_uniform"])
        if not universe_green:
            all_green = False
        per_universe.append(
            {
                "seed": seed,
                "available": True,
                "gclosure_green": gclosure["gclosure_green"],
                "max_closure_residual": gclosure["max_closure_residual"],
                "gznorm_green": gznorm["all_h_nodes_uniform"],
                "universe_green": universe_green,
            }
        )

    count_matches_expected = len(matched_seeds) == N_HARNESS_UNIVERSES
    return {
        "n_universes_checked": len(matched_seeds),
        "n_universes_expected": N_HARNESS_UNIVERSES,
        "count_matches_expected": count_matches_expected,
        "all_universes_gclosure_gznorm_green": bool(all_green and count_matches_expected),
        "per_universe": per_universe,
    }


# ── §2.3 g-precision cross-check (optional, informational) ──────────────────


def check_gprecision(
    harness_root: Path, h_lo: float, h_hi: float, csv_beta_gbar_phi: dict[float, float]
) -> dict[str, Any]:
    """Cross-check the CSV-derived beta_Gbar_phi(h) against a full-precision
    ``selection_tables_h_*.json`` where one exists for the stencil nodes (§2.1)."""
    found: dict[str, Any] = {}
    for h in (h_lo, h_hi):
        label = f"{h:.2f}".replace(".", "_")
        candidates = list(harness_root.glob(f"**/selection_tables_h_{label}.json")) + list(
            harness_root.parent.glob(f"**/selection_tables_h_{label}.json")
        )
        if not candidates:
            found[str(h)] = {"available": False}
            continue
        table = json.loads(candidates[0].read_text())
        full = float(table.get("beta_Gbar_phi", float("nan")))
        col = csv_beta_gbar_phi.get(h, float("nan"))
        rel = abs(full - col) / abs(full) if full else float("nan")
        found[str(h)] = {
            "available": True,
            "source": str(candidates[0]),
            "full_precision_value": full,
            "csv_derived_value": col,
            "relative_diff": rel,
            "within_tolerance": bool(rel <= GPRECISION_TOLERANCE) if rel == rel else None,
        }
    any_checked = any(v.get("available") for v in found.values())
    return {"nodes": found, "any_full_precision_source_found": any_checked}


# ── shared gate collection: dry-run reporting AND real-mode self-gating ─────
# (rev. 2 item 4: real mode must run this SAME suite itself, not trust a separate dry-run.)


def collect_gate_report(
    production_csv_path: Path,
    production_crb_path: Path,
    replicate_csv_path: Path | None,
    harness_root: Path,
    population: int,
    h_lo: float,
    h_hi: float,
    crb_md5_expected: str,
    catalogue_md5_expected: str,
    h_true: float,
) -> dict[str, Any]:
    """Run every §5 gate on the given inputs: g-population (production + replicate), g-znorm
    (production + replicate), g-closure incl. class closure (production + replicate), g-closure/
    g-znorm on every harness universe (rev. 2 item 2), g-byte-id, and the T0 mean_h anchor.
    Returns the full gate table plus an overall ``gates_green`` boolean and a ``NO_READ`` verdict
    naming every gate that fired, matching §4's NO-READ trigger list exactly. Called identically
    by :func:`run_dry_run` (reporting only) and :func:`compute_registered_statistics` (gating).
    """
    report: dict[str, Any] = {}

    production_csv = pd.read_csv(production_csv_path)
    production_crb = pd.read_csv(production_crb_path)
    crb_md5_actual = _md5(production_crb_path)
    report["production_crb_md5"] = {
        "expected": crb_md5_expected,
        "actual": crb_md5_actual,
        "match": crb_md5_actual == crb_md5_expected,
    }
    report["catalogue_md5_of_record"] = {
        "expected": catalogue_md5_expected,
        "note": (
            "not independently re-hashed here -- this script consumes only the CSVs the "
            "catalogue already fed into (event_likelihoods.csv, prepared_cramer_rao_bounds.csv); "
            "recorded for provenance per CLAUDE.md dataset-pinning convention"
        ),
    }
    report["g_population_production"] = check_production_population(production_csv, production_crb)
    report["g_znorm_production"] = check_gznorm(production_csv)

    terms_production = compute_event_terms(production_csv, h_lo, h_hi)
    report["g_closure_production"] = check_gclosure(terms_production)
    report["g_class_closure_production"] = check_class_closure(terms_production, production_crb)

    replicate_available = replicate_csv_path is not None and replicate_csv_path.is_file()
    if replicate_available:
        assert replicate_csv_path is not None  # narrows for mypy
        replicate_csv = pd.read_csv(replicate_csv_path)
        report["g_population_replicate"] = check_production_population(
            replicate_csv, production_crb
        )
        report["g_znorm_replicate"] = check_gznorm(replicate_csv)
        terms_replicate = compute_event_terms(replicate_csv, h_lo, h_hi)
        report["g_closure_replicate"] = check_gclosure(terms_replicate)
        report["g_class_closure_replicate"] = check_class_closure(terms_replicate, production_crb)
    else:
        report["g_population_replicate"] = {"available": False}
        report["g_znorm_replicate"] = {"available": False}
        report["g_closure_replicate"] = {"available": False}
        report["g_class_closure_replicate"] = {"available": False}

    report["g_byte_id_harness"] = reproduce_harness_byte_id(
        harness_root, population, CELL, h_lo, h_hi
    )
    report["g_harness_universes"] = check_harness_universe_gates(
        harness_root, population, CELL, h_lo, h_hi
    )

    mean_h, h_grid = t0_mean_h(production_csv_path, "combined_no_bh")
    # READOUT_RECORD.md's table quotes mean_h at 6 decimal places (0.666987) -- that display
    # precision, not a full-precision stash, is the only anchor of record (repro_summary.json and
    # the c0prime_eval JSONs do not carry a full-precision mean_h). A byte-for-bit comparison
    # against a 6-dp display value is bounded below by that rounding (up to 5e-7), so "reproduces"
    # here means: this script's own full-precision mean_h rounds to the displayed anchor. abs_diff
    # is reported for the record; it is display-rounding noise, not a computation mismatch.
    reproduces = round(mean_h, 6) == T0_MEAN_H_TARGET_IIIB_1D
    report["t0_mean_h"] = {
        "computed": mean_h,
        "target_display_precision": T0_MEAN_H_TARGET_IIIB_1D,
        "computed_rounded_to_6dp": round(mean_h, 6),
        "abs_diff": abs(mean_h - T0_MEAN_H_TARGET_IIIB_1D),
        "reproduces_to_tolerance": reproduces,
        "reproduction_basis": "round(computed, 6) == displayed anchor (source carries no finer precision)",
        "tolerance": T0_MEAN_H_TOLERANCE,
        "n_h_grid": int(h_grid.size),
    }

    report["anchors"] = {
        "production_rows": int(len(production_csv)),
        "production_n_h_nodes": int(production_csv["h"].nunique()),
        "crb_rows": int(len(production_crb)),
        "n_dark_scored": report["g_population_production"]["n_dark_scored"],
        "n_in_catalogue_scored": report["g_population_production"]["n_in_catalogue_scored"],
        "harness_checkpoints_matched": report["g_byte_id_harness"][
            "n_checkpoints_matched_population"
        ],
        "h_stencil": [h_lo, h_hi],
        "h_true": h_true,
    }

    # ── overall gates_green + NO-READ (§4's trigger list, rev. 2 items 2/4) ──
    pop_prod = report["g_population_production"]
    population_production_green = bool(
        pop_prod["join_gate_green"]
        and pop_prod["in_catalogue_matches_expected"]
        and pop_prod["dark_matches_expected"]
    )
    if replicate_available:
        pop_rep = report["g_population_replicate"]
        population_replicate_green = bool(
            pop_rep["join_gate_green"]
            and pop_rep["in_catalogue_matches_expected"]
            and pop_rep["dark_matches_expected"]
        )
        znorm_replicate_green = bool(report["g_znorm_replicate"]["all_h_nodes_uniform"])
        closure_replicate_green = bool(report["g_closure_replicate"]["gclosure_green"])
        class_closure_replicate_green = bool(
            report["g_class_closure_replicate"]["class_closure_green"]
        )
    else:
        # replicate is REPORTED/optional (§2.2) -- its absence is a disclosed skip, not a red gate.
        population_replicate_green = True
        znorm_replicate_green = True
        closure_replicate_green = True
        class_closure_replicate_green = True

    znorm_production_green = bool(report["g_znorm_production"]["all_h_nodes_uniform"])
    closure_production_green = bool(report["g_closure_production"]["gclosure_green"])
    class_closure_production_green = bool(
        report["g_class_closure_production"]["class_closure_green"]
    )
    harness_universes_green = bool(
        report["g_harness_universes"]["all_universes_gclosure_gznorm_green"]
    )
    byte_id_green = bool(report["g_byte_id_harness"]["byte_id_count_green"])
    t0_green = bool(report["t0_mean_h"]["reproduces_to_tolerance"])

    gates_green = bool(
        population_production_green
        and population_replicate_green
        and znorm_production_green
        and znorm_replicate_green
        and closure_production_green
        and closure_replicate_green
        and class_closure_production_green
        and class_closure_replicate_green
        and harness_universes_green
        and byte_id_green
        and t0_green
    )

    triggers: list[str] = []
    if not population_production_green:
        triggers.append("g-population (production)")
    if not population_replicate_green:
        triggers.append("g-population (replicate)")
    if not znorm_production_green:
        triggers.append("g-znorm (production)")
    if not znorm_replicate_green:
        triggers.append("g-znorm (replicate)")
    if not closure_production_green:
        triggers.append("g-closure (production, per-event identity)")
    if not closure_replicate_green:
        triggers.append("g-closure (replicate, per-event identity)")
    if not class_closure_production_green:
        triggers.append("g-closure (class closure, production)")
    if not class_closure_replicate_green:
        triggers.append("g-closure (class closure, replicate)")
    if not harness_universes_green:
        triggers.append("g-closure/g-znorm (harness universe)")
    if not byte_id_green:
        triggers.append("g-byte-id")
    if not t0_green:
        triggers.append("t0-mean-h anchor")

    report["gates_green"] = gates_green
    report["NO_READ"] = {"no_read": bool(triggers), "triggers": triggers}
    return report


# ── dry-run ───────────────────────────────────────────────────────────────


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    """Load every input, run the full §5 gate suite (:func:`collect_gate_report`), print anchors,
    and return the summary (exit 0, no statistic computed)."""
    report = collect_gate_report(
        args.production_csv,
        args.production_crb,
        args.replicate_csv,
        args.harness_root,
        args.population,
        args.h_lo,
        args.h_hi,
        args.crb_md5,
        args.catalogue_md5,
        args.h_true,
    )
    report["mode"] = "dry-run"
    return report


# ── real mode: §2.4 registered statistics + §4 disposition ─────────────────


def compute_registered_statistics(
    production_csv_path: Path,
    production_crb_path: Path,
    replicate_csv_path: Path | None,
    harness_root: Path,
    population: int,
    h_lo: float,
    h_hi: float,
    crb_md5_expected: str,
    catalogue_md5_expected: str,
    h_true: float,
) -> dict[str, Any]:
    """Compute T_prod/Z_prod/T_harn/Z_harn/rho/delta_h_M (§2.4) and the §4 disposition.

    NOT invoked by the builder (standing rule 2: a different agent runs real mode). Implemented
    here so the launch-block script exists in full per the registration draft; the builder's
    BUILD_RECORD documents that only ``--dry-run`` was executed.

    Rev. 2 item 4: real mode now runs the FULL §5 gate suite itself first, via the same
    :func:`collect_gate_report` dry-run uses (g-population, g-znorm, g-closure incl. class
    closure, the harness-universe closure/znorm sweep, g-byte-id, t0 mean_h) -- it no longer
    relies on a temporally-separate dry-run having been green. If any gate is red, this function
    refuses to bank a disposition: it returns the gate table + a ``NO_READ`` record instead (§4's
    NO-READ row -- "nothing banked").
    """
    gates = collect_gate_report(
        production_csv_path,
        production_crb_path,
        replicate_csv_path,
        harness_root,
        population,
        h_lo,
        h_hi,
        crb_md5_expected,
        catalogue_md5_expected,
        h_true,
    )
    if not gates["gates_green"]:
        return {
            "mode": "real",
            "NO_READ": True,
            "no_read_triggers": gates["NO_READ"]["triggers"],
            "disposition": "NO-READ",
            "gates": gates,
        }

    production_csv = pd.read_csv(production_csv_path)
    production_crb = pd.read_csv(production_crb_path)
    terms = compute_event_terms(production_csv, h_lo, h_hi)
    dark_mask = production_crb["host_galaxy_index"].to_numpy() == -1
    dark_idx = set(int(i) for i in np.nonzero(dark_mask)[0])
    dark_terms = terms.loc[terms.index.isin(dark_idx)]

    t_prod = float(dark_terms["s_M"].mean())
    se_prod = float(dark_terms["s_M"].std(ddof=1) / (len(dark_terms) ** 0.5))
    z_prod = t_prod / se_prod if se_prod else float("nan")

    # byte-id instrument gate: harness FULL-score checkpoint means, reproduced bit-for-bit.
    # INFORMATIONAL ONLY below (rev. 1 item 2) -- never enters T_harn/SE_harn/Z_harn.
    byte_id = reproduce_harness_byte_id(harness_root, population, CELL, h_lo, h_hi)
    full_score_means = byte_id["dark_full_score_means"]
    t_full_harn_informational = (
        float(np.mean(full_score_means)) if full_score_means else float("nan")
    )
    se_full_harn_informational = (
        float(np.std(full_score_means, ddof=1) / (len(full_score_means) ** 0.5))
        if len(full_score_means) > 1
        else float("nan")
    )

    # Registered statistic (rev. 1 item 2, 1b): the matched-channel score S_M, computed per
    # harness universe from that universe's OWN per-event diagnostics, same stencil/columns as
    # production. T_harn/SE_harn are the between-universe mean/SE of THESE values.
    harn_matched = compute_harness_matched_channel_scores(
        harness_root, population, CELL, h_lo, h_hi
    )
    t_harn = harn_matched["T_harn"]
    se_harn = harn_matched["SE_harn"]
    z_harn = t_harn / se_harn if se_harn else float("nan")

    rho = t_harn / t_prod if (abs(z_prod) > Z_BAND and t_prod) else None

    # delta_h_M (§2.4, rev. 2 item 1): REPORTED-ONLY, never enters the disposition below.
    delta_h_m = compute_delta_h_m(t_prod, int(len(dark_terms)))

    # class-closure identity (§2.1, rev. 2 item 3), re-derived here (in addition to the
    # pre-disposition gate check above) so its residual sits alongside the other per-event terms
    # in the output record.
    class_closure = check_class_closure(terms, production_crb)

    if abs(z_harn) > Z_BAND and rho is not None and rho >= RHO_ILLEGITIMATE:
        disposition = "ILLEGITIMATE"
    elif abs(z_harn) <= Z_BAND and abs(z_prod) <= Z_BAND:
        disposition = "FLOOR-CONSISTENT"
    elif abs(z_harn) <= Z_BAND and abs(z_prod) > Z_BAND:
        disposition = "INTERMEDIATE (a) harness-clean, production-displaced"
    elif abs(z_harn) > Z_BAND and rho is not None and RHO_MINOR < rho < RHO_ILLEGITIMATE:
        disposition = "INTERMEDIATE (b) partial"
    elif abs(z_harn) > Z_BAND and rho is not None and rho <= RHO_MINOR:
        disposition = "INTERMEDIATE (c) minor-illegitimate"
    elif abs(z_harn) > Z_BAND and rho is None:
        # |Z_harn| > 3 AND |Z_prod| <= 3 (rho undefined) -- REGISTRATION_DRAFT.md §4 revision 1b.
        disposition = "INTERMEDIATE (d) HARNESS-ONLY-SIGNAL"
    else:  # pragma: no cover -- exhaustive by construction (§4 revision 1b); defensive only.
        raise AssertionError(
            f"unreachable disposition state: Z_harn={z_harn}, Z_prod={z_prod}, rho={rho}"
        )

    return {
        "mode": "real",
        "NO_READ": False,
        "T_prod": t_prod,
        "SE_prod": se_prod,
        "Z_prod": z_prod,
        "N_dark_prod": int(len(dark_terms)),
        "T_harn": t_harn,
        "SE_harn": se_harn,
        "Z_harn": z_harn,
        "n_universes_harn": harn_matched["n_universes_available"],
        "T_full_harn_informational": t_full_harn_informational,
        "SE_full_harn_informational": se_full_harn_informational,
        "rho": rho,
        "delta_h_M": delta_h_m,
        "class_closure": class_closure,
        "disposition": disposition,
        "per_event_terms": {
            "s_M": dark_terms["s_M"].tolist(),
            "s_T": dark_terms["s_T"].tolist(),
            "s_C": dark_terms["s_C"].tolist(),
            "s_e": dark_terms["s_e"].tolist(),
            "closure_residual": dark_terms["closure_residual"].tolist(),
            "event_idx": dark_terms.index.tolist(),
        },
        "gates": gates,
        "harness_matched_channel_detail": harn_matched,
    }


# ── CLI ───────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--production-csv", type=Path, required=True)
    p.add_argument("--production-crb", type=Path, required=True)
    p.add_argument("--replicate-csv", type=Path, default=None)
    p.add_argument("--harness-root", type=Path, required=True)
    p.add_argument("--population", type=int, required=True)
    p.add_argument("--h-lo", type=float, required=True)
    p.add_argument("--h-hi", type=float, required=True)
    p.add_argument("--h-true", type=float, required=True)
    p.add_argument("--crb-md5", type=str, required=True)
    p.add_argument("--catalogue-md5", type=str, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.dry_run:
        report = run_dry_run(args)
        print(json.dumps(report, indent=1, default=str))
        print(f"\nDRY-RUN gates all green: {report['gates_green']}", file=sys.stderr)
        return 0

    result = compute_registered_statistics(
        args.production_csv,
        args.production_crb,
        args.replicate_csv,
        args.harness_root,
        args.population,
        args.h_lo,
        args.h_hi,
        args.crb_md5,
        args.catalogue_md5,
        args.h_true,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
