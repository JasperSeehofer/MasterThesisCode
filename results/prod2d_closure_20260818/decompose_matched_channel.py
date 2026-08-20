"""Pre-check O3 (registered 2026-08-21, below the C-SG prereg freeze line).

Free read 2: the MATCHED-CHANNEL decomposition of the 12 banked B-SEL seeds.

Derivation (registered with the band block in PREREGISTRATION_SELFGEN_CONTROL.md):
the estimator's per-event mixture normalization splits as

    D_tilde_phi = alpha_G_phi + beta_Gbar_phi     (bayesian_statistics.py:2427)

with ``alpha_G_phi`` the catalogue-sector and ``beta_Gbar_phi`` the dark-sector
normalization. B-SEL draws events CONDITIONED on being dark-detected
(z ~ w_pop*(1-f_bar)*S_bar_phi), so the model-matched conditional likelihood
for its draw is the completion numerator over the DARK-sector normalization:

    L_matched = B_num / beta_Gbar_phi = B_num / (D_tilde_phi - alpha_G_phi).

If the completion leg is internally self-consistent and the draw matches the
model's dark-detected density, E[d_h ln L_matched] = 0 at truth. The pure
channel of pre-check O2 (B_num / D_tilde_phi) differs from the matched channel
by the EVENT-INDEPENDENT tilt ln(D_tilde/beta_Gbar)(h) = -ln(1 - w_tilde_G(h)),
amplified by n_events in the seed posterior -- the registered candidate owner
of O2's pure-channel residual (-0.0291).

DECISION STATISTIC: bias_matched = mean_12(mean_h_matched) - 0.73 under the
row #146 corrected combine on H_GRID_41.

BANDS (materiality, same scale + A15 no-statistical-band statement as O2;
deterministic paired read on fixed banked data):

    |bias_matched| <  0.0023  -> MATCHED-CONSISTENT    (completion leg self-
                                 consistent at C-SG resolution on B-SEL data;
                                 C-SG v3 confirms with a clean generator)
    |bias_matched| in [0.0023, 0.0110) -> MATCHED-SMALL (residual at C-SG
                                 resolution scale; attributable a priori to
                                 B-SEL's known generator-side caveats (f_k-vs-
                                 f_bar pixel mismatch, donor rows, sigma draw);
                                 C-SG v3 adjudicates)
    |bias_matched| >= 0.0110  -> MATCHED-INCONSISTENT  (the completion leg
                                 itself carries a substantial defect signal;
                                 rows #137/#140 partially reinstated in the
                                 matched channel)

Validity gates (can-fail, scored before bias_matched is read):

    GATE T -- h-only-ness: within every (seed, h), alpha_G_phi and D_tilde_phi
              are constant across events to <= 2e-6 relative (their storage
              precision; O2 GATE AMENDMENT 1), and beta_Gbar = D_tilde - alpha
              > 0 at every node.
    GATE F2 -- the full-channel fleet bias reproduces -0.1083 to <= 5e-5
              (same machinery as O2's GATE F).

REPORTED-ONLY: the analytic tilt d/dh[ln(D_tilde/beta_Gbar)] at h=0.73 per
seed; the identity bias_pure vs bias_matched gap against the tilt prediction;
per-seed matched-channel map/sigma_h/c68/r_low and physics-floor exclusions;
and the b0-arm CATALOGUE-matched corroboration (L_cat-sector conditional
(alpha_G_phi/r_Malm)*L_cat_no_bh / alpha_G_phi = L_cat_no_bh/r_Malm, on the
25 banked b0 seeds) -- EXPLORATORY, convention not independently verified,
carries no verdict.

Usage:
    uv run python results/prod2d_closure_20260818/decompose_matched_channel.py
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
    moment_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARMS_DIR = REPO_ROOT / "results/prod2d_closure_20260818/correspondence_arms"
CSV_ROOT = REPO_ROOT / "results/prod2d_closure_20260818/arm_event_likelihoods"

BIAS_OF_RECORD = -0.1083
GATE_T_TOL = 2.0e-6
GATE_F2_FLEET_TOL = 5.0e-5
BAND_SUBSTANTIAL = 0.0110
BAND_MATERIAL = 0.0023


def csv_for(arm: str, seed: int) -> Path:
    return (
        CSV_ROOT
        / f"{arm}_seed{seed}"
        / f"seed{seed}"
        / "simulations/diagnostics/event_likelihoods.csv"
    )


def moments(vals: npt.NDArray[np.float64]) -> dict[str, Any]:
    """Corrected-convention (row #146) posterior moments from a likelihood matrix."""
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
    return {
        "mean_h": mean_h,
        "map_h": map_h,
        "sigma_h": float(np.sqrt(max(var, 0.0))),
        "c68": _hpd_contains(post_n, weights, target, 0.68),
        "r_low": map_h <= R_LOW_THRESHOLD,
        "n_excluded_physics_floor": int((~(vals > 0.0).any(axis=1)).sum()),
    }


def pivots(df: pd.DataFrame, cols: list[str]) -> dict[str, npt.NDArray[np.float64]]:
    grid = np.array(H_GRID_41, dtype=np.float64)
    df = df[np.isin(df["h"].to_numpy(dtype=np.float64), grid)]
    out: dict[str, npt.NDArray[np.float64]] = {}
    for c in cols:
        out[c] = (
            df.pivot_table(index="event_idx", columns="h", values=c, aggfunc="first")
            .reindex(columns=grid)
            .to_numpy(dtype=np.float64)
        )
    return out


def h_only_check(mat: npt.NDArray[np.float64]) -> float:
    """Max relative spread across events, per h-node."""
    lo = np.nanmin(mat, axis=0)
    hi = np.nanmax(mat, axis=0)
    scale = np.maximum(np.abs(lo), np.finfo(float).tiny)
    return float(np.max((hi - lo) / scale))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "results/prod2d_closure_20260818/decompose_matched_channel_output.json"
        ),
    )
    args = ap.parse_args()
    grid = np.array(H_GRID_41, dtype=np.float64)
    i_lo = int(np.nonzero(np.isclose(grid, 0.725))[0][0])
    i_hi = int(np.nonzero(np.isclose(grid, 0.735))[0][0])

    bsel = sorted(
        (json.loads(p.read_text()) for p in ARMS_DIR.glob("bsel_seed*.json")),
        key=lambda r: int(r["seed"]),
    )

    per_seed: list[dict[str, Any]] = []
    gate_t_rows: list[dict[str, Any]] = []
    full_means: list[float] = []
    for rec in bsel:
        seed = int(rec["seed"])
        df = pd.read_csv(csv_for("bsel", seed))
        m = pivots(df, ["combined_no_bh", "B_num", "alpha_G_phi", "D_tilde_phi"])
        alpha, dtil = m["alpha_G_phi"], m["D_tilde_phi"]
        beta_gbar = dtil - alpha
        spread = max(h_only_check(alpha), h_only_check(dtil))
        gate_t_rows.append(
            {
                "seed": seed,
                "max_rel_spread": spread,
                "min_beta_gbar": float(np.nanmin(beta_gbar)),
                "pass": spread <= GATE_T_TOL and float(np.nanmin(beta_gbar)) > 0.0,
            }
        )
        matched = m["B_num"] / beta_gbar
        m_full = moments(m["combined_no_bh"])
        m_matched = moments(matched)
        full_means.append(m_full["mean_h"])

        # analytic tilt: event-independent, take node-wise median across events
        with np.errstate(divide="ignore", invalid="ignore"):
            tilt = np.nanmedian(np.log(dtil / beta_gbar), axis=0)
        tilt_slope = float((tilt[i_hi] - tilt[i_lo]) / (grid[i_hi] - grid[i_lo]))
        per_seed.append(
            {
                "seed": seed,
                "n_events": int(m["B_num"].shape[0]),
                "full_mean_h": m_full["mean_h"],
                "matched": m_matched,
                "tilt_dlnDoverBeta_dh_at_073": tilt_slope,
                "tilt_times_n": tilt_slope * m["B_num"].shape[0],
            }
        )

    matched_means = np.array([r["matched"]["mean_h"] for r in per_seed], dtype=np.float64)
    bias_full = float(np.mean(full_means) - H_TRUE)
    bias_matched = float(matched_means.mean() - H_TRUE)
    sd_matched = float(matched_means.std(ddof=1))

    gate_t_pass = all(r["pass"] for r in gate_t_rows)
    gate_f2_pass = abs(bias_full - BIAS_OF_RECORD) <= GATE_F2_FLEET_TOL
    gates_pass = gate_t_pass and gate_f2_pass

    if not gates_pass:
        band = "GATES-FAILED -- bias_matched MAY NOT BE READ"
    elif abs(bias_matched) >= BAND_SUBSTANTIAL:
        band = "MATCHED-INCONSISTENT"
    elif abs(bias_matched) >= BAND_MATERIAL:
        band = "MATCHED-SMALL"
    else:
        band = "MATCHED-CONSISTENT"

    # REPORTED-ONLY b0 catalogue-sector corroboration (exploratory, no verdict).
    b0 = sorted(
        (json.loads(p.read_text()) for p in ARMS_DIR.glob("b0_seed*.json")),
        key=lambda r: int(r["seed"]),
    )
    b0_rows: list[dict[str, Any]] = []
    for rec in b0:
        seed = int(rec["seed"])
        path = csv_for("b0", seed)
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        m = pivots(df, ["L_cat_no_bh", "r_Malm"])
        cat_matched = m["L_cat_no_bh"] / m["r_Malm"]
        b0_rows.append({"seed": seed, "cat_matched": moments(cat_matched)})
    b0_means = [r["cat_matched"]["mean_h"] for r in b0_rows if r["cat_matched"]["mean_h"]]
    b0_bias = float(np.mean(b0_means) - H_TRUE) if b0_means else None

    out = {
        "registered_in": "PREREGISTRATION_SELFGEN_CONTROL.md pre-check O3 (2026-08-21)",
        "gate_t_h_only": {"tol": GATE_T_TOL, "rows": gate_t_rows, "pass": gate_t_pass},
        "gate_f2_full_arm": {
            "recomputed_fleet_bias": bias_full,
            "bias_of_record": BIAS_OF_RECORD,
            "pass": gate_f2_pass,
        },
        "n_seeds": len(per_seed),
        "bias_matched": bias_matched,
        "sd_matched": sd_matched,
        "sem_matched": sd_matched / float(np.sqrt(len(per_seed))),
        "bands": {"substantial": BAND_SUBSTANTIAL, "material": BAND_MATERIAL},
        "band_fired": band,
        "per_seed": per_seed,
        "b0_catalogue_matched_EXPLORATORY": {"n": len(b0_rows), "bias": b0_bias, "rows": b0_rows},
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                "gate_t": gate_t_pass,
                "gate_f2": gate_f2_pass,
                "bias_matched": bias_matched,
                "sem_matched": out["sem_matched"],
                "band_fired": band,
                "b0_cat_matched_bias_EXPLORATORY": b0_bias,
                "out": args.out,
            },
            indent=2,
        )
    )
    return 0 if gates_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
