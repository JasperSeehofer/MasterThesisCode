"""Pre-check O2 (PREREGISTRATION_SELFGEN_CONTROL.md section 9): L_cat_no_bh == 0.

The pure-completion decomposition of the 12 banked B-SEL seeds, at zero compute.
Per event and h-node the harness assembles (bayesian_statistics.py:5248-5255,
re-derived in correspondence_1d.py:1439-1500):

    combined_no_bh = (beta_G_phi * L_cat_no_bh + B_num) / D_tilde_phi,
    beta_G_phi = alpha_G_phi / r_Malm.

Setting ``L_cat_no_bh == 0`` therefore gives the exact pure-completion per-event
likelihood by subtraction:

    combined_pure = combined_no_bh - beta_G_phi * L_cat_no_bh / D_tilde_phi.

Scored with the fully-corrected combine of record (ledger row #146):
``combine_log_likelihood(zero_handling="physics_floor")`` +
``moment_weights(convention="trapezoid")`` on ``H_GRID_41``.

DECISION STATISTIC AND BANDS (registered pre-data; see the PRE-CHECK O2 BAND
REGISTRATION block appended below the pre-registration's freeze line):

    delta_bias = mean_12(mean_h_pure) - mean_12(mean_h_full)

This is a DETERMINISTIC PAIRED recomputation on fixed banked data: the sampling
variance of the paired difference is exactly zero, so per amendment A15 (and its
recorded A-7 counter-example) NO statistical band is applied. The bands are
materiality thresholds referenced to the downstream decision -- whether C-SG's
design must change before 51-69 CPU-h are spent:

    |delta_bias| >= 0.0110  ->  IMPOSTOR-SUBSTANTIAL  (>=10% of the -0.1083;
                                the "pure completion carries it" language of
                                rows #137/#140 needs revisiting AND C-SG's
                                design must change)
    |delta_bias| >= 0.0023  ->  IMPOSTOR-MATERIAL     (at or above C-SG's own
                                best 15-seed resolution SE = 0.009/sqrt(15);
                                C-SG's design must change before it runs)
    |delta_bias| <  0.0023  ->  IMPOSTOR-IMMATERIAL   (the section-5 generator-
                                model mismatch is quantified and below C-SG's
                                resolution; C-SG proceeds unchanged)

Validity gates, all can-fail, all scored before delta_bias is read:

    GATE I  -- identity: the assembled RHS reproduces the banked combined_no_bh
               column to <= 1e-9 relative, every (event, h) cell, all 12 seeds.
    GATE F  -- full-arm reproduction: the recomputed full-arm fleet bias equals
               row #146's -0.1083 to <= 5e-5, and the in-scorer moment code
               matches compute_seed_statistics() to <= 1e-12 per seed.
    GATE P  -- provenance: reproduces section 0 item 5's measured impostor-share
               stats on bsel_seed900101 at h = 0.73 (128/174 active; median
               6e-4, p90 0.057, p99 0.647, max 0.821, mean 0.034).

REPORTED-ONLY secondaries: per-seed pure-arm map_h/sigma_h/r_low/c68, the
per-event score at truth (central difference at 0.725/0.735) full vs pure, the
count of events excluded by physics_floor in the pure arm (all-zero rows), and
the 3 unbanked bsel CSV directories (seeds 900113-900115) which are DISCLOSED
and not scored (no banked JSON -> no pairing provenance; GATE R-1 convention).

Design-time sightings disclosed: the identity was sighted on one row
(seed900101, event 0, h = 0.5) and GATE P's targets were measured at
registration time (prereg section 0 item 5). delta_bias itself has not been
computed before this scorer's commit.

Usage:
    uv run python results/prod2d_closure_20260818/decompose_impostor_leg.py
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from darksiren_emri.validation.correspondence_1d import (
    H_GRID_41,
    H_TRUE,
    R_LOW_THRESHOLD,
    _hpd_contains,
    combine_log_likelihood,
    compute_seed_statistics,
    moment_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARMS_DIR = REPO_ROOT / "results/prod2d_closure_20260818/correspondence_arms"
CSV_ROOT = REPO_ROOT / "results/prod2d_closure_20260818/arm_event_likelihoods"

BIAS_OF_RECORD = -0.1083  # ledger row #146, fully corrected, N=12
# GATE AMENDMENT 1 (see prereg): alpha_G_phi/r_Malm/D_tilde_phi are stored at 7
# significant figures (bayesian_statistics.py:4365 `_seven_sf`); per-column max
# quantization error 4.9e-7, three columns -> 2e-6 bound. Observed 5.0-5.5e-7.
GATE_I_TOL = 2.0e-6
GATE_F_FLEET_TOL = 5.0e-5
GATE_F_MOMENT_TOL = 1.0e-12
BAND_SUBSTANTIAL = 0.0110
BAND_MATERIAL = 0.0023

# Section 0 item 5 targets (bsel_seed900101, h = 0.73) and tolerances, under the
# REGISTRATION-TIME VERIFIER'S convention (GATE AMENDMENT 1): share =
# alpha_G_phi*L_cat/(alpha_G_phi*L_cat + B_num) -- no 1/r_Malm -- with quantiles
# over ACTIVE events (L_cat > 0) only. This is a provenance assertion; the
# assembly-true beta-convention shares are reported separately as descriptive
# numbers of record.
GATE_P_TARGETS: dict[str, tuple[float, float]] = {
    "n_active": (128.0, 0.0),
    "n_events": (174.0, 0.0),
    "share_median": (6.0e-4, 5.0e-5),
    "share_p90": (0.057, 0.001),
    "share_p99": (0.647, 0.001),
    "share_max": (0.821, 0.001),
    "share_mean": (0.034, 0.001),
    "n_above_half": (2.0, 0.0),
}


def csv_for(seed: int) -> Path:
    return (
        CSV_ROOT
        / f"bsel_seed{seed}"
        / f"seed{seed}"
        / "simulations/diagnostics/event_likelihoods.csv"
    )


def load_matrices(
    df: pd.DataFrame,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, int]:
    """Pivot to (event, h) matrices of combined_no_bh (full) and pure, on H_GRID_41.

    Returns:
        (full, pure, gate_i_max_rel, n_events). ``pure`` is clipped at 0 from
        below only after the identity gate is computed, so a negative residual
        beyond float error would fail GATE I rather than be hidden.
    """
    grid = np.array(H_GRID_41, dtype=np.float64)
    df = df[np.isin(df["h"].to_numpy(dtype=np.float64), grid)].copy()
    beta = df["alpha_G_phi"].to_numpy(np.float64) / df["r_Malm"].to_numpy(np.float64)
    df["catalogue_term"] = (
        beta * df["L_cat_no_bh"].to_numpy(np.float64) / df["D_tilde_phi"].to_numpy(np.float64)
    )
    df["completion_term"] = df["B_num"].to_numpy(np.float64) / df["D_tilde_phi"].to_numpy(
        np.float64
    )

    def piv(col: str) -> npt.NDArray[np.float64]:
        return (
            df.pivot_table(index="event_idx", columns="h", values=col, aggfunc="first")
            .reindex(columns=grid)
            .to_numpy(dtype=np.float64)
        )

    full = piv("combined_no_bh")
    cat = piv("catalogue_term")
    comp = piv("completion_term")

    recon = cat + comp
    scale = np.maximum(np.abs(full), np.finfo(float).tiny)
    gate_i_max_rel = float(np.nanmax(np.abs(recon - full) / scale))

    pure = full - cat
    pure = np.clip(pure, 0.0, None)
    return full, pure, gate_i_max_rel, int(full.shape[0])


def moments(vals: npt.NDArray[np.float64]) -> dict[str, Any]:
    """Corrected-convention posterior moments (row #146) from a likelihood matrix."""
    grid = np.array(H_GRID_41, dtype=np.float64)
    sum_log_l = combine_log_likelihood(vals, "physics_floor")
    if not np.isfinite(sum_log_l).any():
        return {"mean_h": None, "map_h": None, "sigma_h": None, "c68": None, "r_low": None}
    weights = moment_weights(grid, "trapezoid")
    lp = sum_log_l - sum_log_l.max()
    post = np.exp(lp)
    norm = float((post * weights).sum())
    post_n = post / norm if norm > 0 else post
    mean_h = float((post_n * grid * weights).sum())
    var = float((post_n * (grid - mean_h) ** 2 * weights).sum())
    map_h = float(grid[int(np.argmax(sum_log_l))])
    target = int(np.nonzero(np.isclose(grid, H_TRUE))[0][0])
    n_excluded = int((~(vals > 0.0).any(axis=1)).sum())
    return {
        "mean_h": mean_h,
        "map_h": map_h,
        "sigma_h": float(np.sqrt(max(var, 0.0))),
        "c68": _hpd_contains(post_n, weights, target, 0.68),
        "r_low": map_h <= R_LOW_THRESHOLD,
        "n_excluded_physics_floor": n_excluded,
    }


def score_at_truth(vals: npt.NDArray[np.float64]) -> dict[str, Any]:
    """REPORTED-ONLY per-event score at truth: central difference ln L over 0.725/0.735."""
    grid = np.array(H_GRID_41, dtype=np.float64)
    i_lo = int(np.nonzero(np.isclose(grid, 0.725))[0][0])
    i_hi = int(np.nonzero(np.isclose(grid, 0.735))[0][0])
    lo, hi = vals[:, i_lo], vals[:, i_hi]
    ok = (lo > 0.0) & (hi > 0.0)
    if not ok.any():
        return {"mean_score": None, "n_used": 0, "n_skipped": int(vals.shape[0])}
    s = (np.log(hi[ok]) - np.log(lo[ok])) / (0.735 - 0.725)
    return {
        "mean_score": float(s.mean()),
        "sem_score": float(s.std(ddof=1) / np.sqrt(s.size)) if s.size > 1 else None,
        "n_used": int(ok.sum()),
        "n_skipped": int((~ok).sum()),
    }


def _share_stats(share: npt.NDArray[np.float64]) -> dict[str, float]:
    return {
        "share_median": float(np.median(share)),
        "share_p90": float(np.percentile(share, 90)),
        "share_p99": float(np.percentile(share, 99)),
        "share_max": float(share.max()),
        "share_mean": float(share.mean()),
        "n_above_half": float((share > 0.5).sum()),
    }


def gate_p(df: pd.DataFrame) -> dict[str, Any]:
    """Provenance: impostor-share stats on bsel_seed900101 at h = 0.73.

    GATE AMENDMENT 1: asserts the section 0 item 5 numbers under the
    REGISTRATION-TIME VERIFIER'S convention (alpha, no 1/r_Malm, active events
    only) as a provenance check, and reports the assembly-true beta-convention
    shares (verified by GATES I+F against the banked combined column) as the
    descriptive numbers of record.
    """
    at = df[np.isclose(df["h"].to_numpy(np.float64), 0.73)]
    alpha = at["alpha_G_phi"].to_numpy(np.float64)
    lcat = at["L_cat_no_bh"].to_numpy(np.float64)
    b_num = at["B_num"].to_numpy(np.float64)
    r_malm = at["r_Malm"].to_numpy(np.float64)
    active = lcat > 0.0

    share_verifier = (alpha * lcat) / (alpha * lcat + b_num)
    share_beta = (alpha / r_malm * lcat) / (alpha / r_malm * lcat + b_num)

    measured: dict[str, float] = {
        "n_active": float(active.sum()),
        "n_events": float(at.shape[0]),
        **_share_stats(share_verifier[active]),
    }
    rows = {
        k: {"measured": measured[k], "target": t, "tol": tol, "pass": abs(measured[k] - t) <= tol}
        for k, (t, tol) in GATE_P_TARGETS.items()
    }
    return {
        "convention_note": (
            "targets asserted under the registration-time verifier's convention "
            "(alpha*L/(alpha*L+B), active events only); beta convention is the "
            "assembly-true share"
        ),
        "rows": rows,
        "beta_convention_active_events": _share_stats(share_beta[active]),
        "beta_convention_all_events": _share_stats(share_beta),
        "pass": all(r["pass"] for r in rows.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "results/prod2d_closure_20260818/decompose_impostor_leg_output.json"
        ),
    )
    args = ap.parse_args()

    banked = sorted(
        (json.loads(p.read_text()) for p in ARMS_DIR.glob("bsel_seed*.json")),
        key=lambda r: int(r["seed"]),
    )
    seeds = [int(r["seed"]) for r in banked]

    unbanked = sorted(
        d.name
        for d in CSV_ROOT.glob("bsel_seed*")
        if int(d.name.removeprefix("bsel_seed")) not in seeds
    )

    per_seed: list[dict[str, Any]] = []
    gate_i_rows: list[dict[str, Any]] = []
    gate_f_rows: list[dict[str, Any]] = []
    gate_p_result: dict[str, Any] | None = None

    for rec in banked:
        seed = int(rec["seed"])
        df = pd.read_csv(csv_for(seed))
        full_vals, pure_vals, gi, n_events = load_matrices(df)
        paired = n_events == int(rec["n_eff"])
        gate_i_rows.append({"seed": seed, "max_rel": gi, "pass": gi <= GATE_I_TOL})

        m_full = moments(full_vals)
        m_pure = moments(pure_vals)

        ref = compute_seed_statistics(csv_for(seed), seed)
        d_moment = abs(m_full["mean_h"] - ref.mean_h)
        gate_f_rows.append(
            {
                "seed": seed,
                "delta_vs_compute_seed_statistics": d_moment,
                "pass": d_moment <= GATE_F_MOMENT_TOL,
            }
        )

        if seed == 900101:
            gate_p_result = gate_p(df)

        per_seed.append(
            {
                "seed": seed,
                "paired": paired,
                "n_events": n_events,
                "full": m_full,
                "pure": m_pure,
                "delta_mean_h": m_pure["mean_h"] - m_full["mean_h"],
                "score_at_truth_full": score_at_truth(full_vals),
                "score_at_truth_pure": score_at_truth(pure_vals),
            }
        )

    full_means = np.array([r["full"]["mean_h"] for r in per_seed], dtype=np.float64)
    pure_means = np.array([r["pure"]["mean_h"] for r in per_seed], dtype=np.float64)
    bias_full = float(full_means.mean() - H_TRUE)
    bias_pure = float(pure_means.mean() - H_TRUE)
    delta_bias = bias_pure - bias_full

    gate_i_pass = all(r["pass"] for r in gate_i_rows)
    fleet_dev = abs(bias_full - BIAS_OF_RECORD)
    gate_f_pass = fleet_dev <= GATE_F_FLEET_TOL and all(r["pass"] for r in gate_f_rows)
    assert gate_p_result is not None
    gates_pass = gate_i_pass and gate_f_pass and bool(gate_p_result["pass"])

    if not gates_pass:
        band = "GATES-FAILED -- delta_bias MAY NOT BE READ"
    elif abs(delta_bias) >= BAND_SUBSTANTIAL:
        band = "IMPOSTOR-SUBSTANTIAL"
    elif abs(delta_bias) >= BAND_MATERIAL:
        band = "IMPOSTOR-MATERIAL"
    else:
        band = "IMPOSTOR-IMMATERIAL"

    out = {
        "registered_in": "PREREGISTRATION_SELFGEN_CONTROL.md section 9 (pre-check O2)",
        "gate_i_identity": {"tol": GATE_I_TOL, "rows": gate_i_rows, "pass": gate_i_pass},
        "gate_f_full_arm": {
            "recomputed_fleet_bias": bias_full,
            "bias_of_record": BIAS_OF_RECORD,
            "fleet_dev": fleet_dev,
            "fleet_tol": GATE_F_FLEET_TOL,
            "moment_rows": gate_f_rows,
            "pass": gate_f_pass,
        },
        "gate_p_provenance": gate_p_result,
        "unbanked_csv_dirs_disclosed": unbanked,
        "n_seeds": len(per_seed),
        "bias_full": bias_full,
        "bias_pure": bias_pure,
        "delta_bias": delta_bias,
        "delta_bias_frac_of_record": delta_bias / abs(BIAS_OF_RECORD),
        "bands": {"substantial": BAND_SUBSTANTIAL, "material": BAND_MATERIAL},
        "band_fired": band,
        "per_seed": per_seed,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                "gate_i": gate_i_pass,
                "gate_f": gate_f_pass,
                "gate_p": bool(gate_p_result["pass"]),
                "bias_full": bias_full,
                "bias_pure": bias_pure,
                "delta_bias": delta_bias,
                "band_fired": band,
                "out": args.out,
            },
            indent=2,
        )
    )
    return 0 if gates_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
