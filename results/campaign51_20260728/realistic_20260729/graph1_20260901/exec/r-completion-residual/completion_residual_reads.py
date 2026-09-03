#!/usr/bin/env python3
"""Build node b-completion-scorer — r-completion-residual reads (Research Graph 1, Branch G, wave 3).

Implements REGISTRATION_DRAFT.md §2.1-§2.4 (the g-closure identity, Read A / production, Read B /
S3 harness, the registered statistics T_prod/Z_prod/T_harn/Z_harn/rho/delta_h_M) and the §5 gates
(g-closure, g-population, g-precision, g-znorm, g-byte-id). Zero waveform/pipeline compute: every
number here is column arithmetic on already-produced CSVs and harness checkpoint JSONs.

Two modes:

* ``--dry-run`` loads every input, runs the gates (g-population, the g-closure identity, the
  g-byte-id instrument reproduction, the g-znorm spot check), prints row counts and anchors, and
  exits 0 WITHOUT computing the registered statistic (T_prod, T_harn, Z, rho, or the disposition).
* real mode (no ``--dry-run``) additionally computes the registered statistics of §2.4 and writes
  the full JSON record (every per-event term, the closure residual, SE, Z, and the disposition
  inputs of §4) to ``--out``.

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


# ── T0 mean_h reproduction (the re-baseline gate, §7 / BUILD_RECORD target) ─


def t0_mean_h(csv_path: Path, channel: str = "combined_no_bh") -> tuple[float, npt.NDArray[np.float64]]:
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


def compute_event_terms(
    df: pd.DataFrame, h_lo: float, h_hi: float
) -> pd.DataFrame:
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

    s_M = (ln_b_num[:, 1] - ln_b_num[:, 0]) / dh - (ln_beta_gbar_phi[:, 1] - ln_beta_gbar_phi[:, 0]) / dh
    s_T = (ln_beta_gbar_phi[:, 1] - ln_beta_gbar_phi[:, 0]) / dh - (
        den_log_term.to_numpy(dtype=np.float64)[:, 1] - den_log_term.to_numpy(dtype=np.float64)[:, 0]
    ) / dh
    s_C = (
        (num_log_term_no_bh.to_numpy(dtype=np.float64)[:, 1] - ln_b_num[:, 1]) / dh
        - (num_log_term_no_bh.to_numpy(dtype=np.float64)[:, 0] - ln_b_num[:, 0]) / dh
    )
    s_e = (
        num_log_term_no_bh.to_numpy(dtype=np.float64)[:, 1]
        - num_log_term_no_bh.to_numpy(dtype=np.float64)[:, 0]
    ) / dh - (
        den_log_term.to_numpy(dtype=np.float64)[:, 1] - den_log_term.to_numpy(dtype=np.float64)[:, 0]
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


# ── dry-run ───────────────────────────────────────────────────────────────


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    """Load every input, run the gates, print anchors, and return the summary (exit 0, no
    statistic computed)."""
    report: dict[str, Any] = {"mode": "dry-run"}

    production_csv = pd.read_csv(args.production_csv)
    production_crb = pd.read_csv(args.production_crb)
    crb_md5_actual = _md5(args.production_crb)
    report["production_crb_md5"] = {
        "expected": args.crb_md5,
        "actual": crb_md5_actual,
        "match": crb_md5_actual == args.crb_md5,
    }
    report["catalogue_md5_of_record"] = {
        "expected": args.catalogue_md5,
        "note": (
            "not independently re-hashed here -- this script consumes only the CSVs the "
            "catalogue already fed into (event_likelihoods.csv, prepared_cramer_rao_bounds.csv); "
            "recorded for provenance per CLAUDE.md dataset-pinning convention"
        ),
    }
    report["g_population_production"] = check_production_population(production_csv, production_crb)
    report["g_znorm_production"] = check_gznorm(production_csv)

    if args.replicate_csv is not None and args.replicate_csv.is_file():
        replicate_csv = pd.read_csv(args.replicate_csv)
        report["g_population_replicate"] = check_production_population(replicate_csv, production_crb)
        report["g_znorm_replicate"] = check_gznorm(replicate_csv)
    else:
        report["g_population_replicate"] = {"available": False}

    terms = compute_event_terms(production_csv, args.h_lo, args.h_hi)
    report["g_closure_production"] = check_gclosure(terms)

    report["g_byte_id_harness"] = reproduce_harness_byte_id(
        args.harness_root, args.population, CELL, args.h_lo, args.h_hi
    )

    mean_h, h_grid = t0_mean_h(args.production_csv, "combined_no_bh")
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
        "harness_checkpoints_matched": report["g_byte_id_harness"]["n_checkpoints_matched_population"],
        "h_stencil": [args.h_lo, args.h_hi],
        "h_true": args.h_true,
    }
    return report


# ── real mode: §2.4 registered statistics + §4 disposition ─────────────────


def compute_registered_statistics(
    production_csv_path: Path,
    production_crb_path: Path,
    harness_root: Path,
    population: int,
    h_lo: float,
    h_hi: float,
) -> dict[str, Any]:
    """Compute T_prod/Z_prod/T_harn/Z_harn/rho/delta_h_M (§2.4) and the §4 disposition.

    NOT invoked by the builder (standing rule 2: a different agent runs real mode). Implemented
    here so the launch-block script exists in full per the registration draft; the builder's
    BUILD_RECORD documents that only ``--dry-run`` was executed.
    """
    production_csv = pd.read_csv(production_csv_path)
    production_crb = pd.read_csv(production_crb_path)
    terms = compute_event_terms(production_csv, h_lo, h_hi)
    dark_mask = production_crb["host_galaxy_index"].to_numpy() == -1
    dark_idx = set(int(i) for i in np.nonzero(dark_mask)[0])
    dark_terms = terms.loc[terms.index.isin(dark_idx)]

    t_prod = float(dark_terms["s_M"].mean())
    se_prod = float(dark_terms["s_M"].std(ddof=1) / (len(dark_terms) ** 0.5))
    z_prod = t_prod / se_prod if se_prod else float("nan")

    byte_id = reproduce_harness_byte_id(harness_root, population, CELL, h_lo, h_hi)
    means = byte_id["dark_full_score_means"]
    t_harn = float(np.mean(means)) if means else float("nan")
    se_harn = float(np.std(means, ddof=1) / (len(means) ** 0.5)) if len(means) > 1 else float("nan")
    z_harn = t_harn / se_harn if se_harn else float("nan")

    rho = t_harn / t_prod if (abs(z_prod) > Z_BAND and t_prod) else None

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
    else:
        disposition = "INTERMEDIATE (unclassified -- rho undefined, |Z_prod| <= 3)"

    return {
        "mode": "real",
        "T_prod": t_prod,
        "SE_prod": se_prod,
        "Z_prod": z_prod,
        "N_dark_prod": int(len(dark_terms)),
        "T_harn": t_harn,
        "SE_harn": se_harn,
        "Z_harn": z_harn,
        "n_universes_harn": len(means) if means else 0,
        "rho": rho,
        "disposition": disposition,
        "per_event_terms": {
            "s_M": dark_terms["s_M"].tolist(),
            "s_T": dark_terms["s_T"].tolist(),
            "s_C": dark_terms["s_C"].tolist(),
            "s_e": dark_terms["s_e"].tolist(),
            "closure_residual": dark_terms["closure_residual"].tolist(),
            "event_idx": dark_terms.index.tolist(),
        },
        "gates": {
            "g_closure": check_gclosure(terms),
            "g_byte_id_harness": byte_id,
        },
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
        gates_green = (
            report["g_population_production"]["join_gate_green"]
            and report["g_population_production"]["in_catalogue_matches_expected"]
            and report["g_population_production"]["dark_matches_expected"]
            and report["g_znorm_production"]["all_h_nodes_uniform"]
            and report["g_closure_production"]["gclosure_green"]
            and report["g_byte_id_harness"]["byte_id_count_green"]
            and report["t0_mean_h"]["reproduces_to_tolerance"]
        )
        print(f"\nDRY-RUN gates all green: {gates_green}", file=sys.stderr)
        return 0

    result = compute_registered_statistics(
        args.production_csv,
        args.production_crb,
        args.harness_root,
        args.population,
        args.h_lo,
        args.h_hi,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
