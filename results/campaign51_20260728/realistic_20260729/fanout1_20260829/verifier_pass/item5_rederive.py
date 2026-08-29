#!/usr/bin/env python3
"""Item 5 (B4.1 [IMP]) independent re-derivation — END-OF-FAN-OUT VERIFIER PASS.

Written fresh by the assigned verifier (no code copied from
b4_imp_stage1_forecast.py's per-event scoring machinery; only the pipeline's
own registered combine/moment functions are reused, since those ARE the
production code under test, not the record's restatement of a number).

Re-derives, straight from the raw event_likelihoods.csv files on disk:
  1. The FT-arm (fused completion, twin/phi catalogue leg) 12-seed fleet
     bias_full, bias_pure, and delta = pure - full, on H_GRID_41.
  2. The FC-arm equivalent.
  3. The O2 (bsel, off/coded) delta, compared to the banked
     decompose_impostor_leg_output.json value the record claims 4e-17 agreement with.
  4. The assembly-identity gate (cat_term + comp_term == combined_no_bh) on
     every arm, independently.
  5. The production HEAD (iiib) ASSUMPTION-JOIN in-catalogue fraction check
     (76/1588).

No numbers are taken from b4_imp_stage1_forecast.json; every input is a raw
CSV/JSON path cited in the record.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from darksiren_emri.validation.correspondence_1d import (
    H_GRID_41,
    H_TRUE,
    combine_log_likelihood,
    moment_weights,
)

REPO_ROOT = Path("/home/jasper/Repositories/darksiren-emri")
BSEL_ROOT = REPO_ROOT / "results/prod2d_closure_20260818/arm_event_likelihoods"
P3_WORK = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/p3_work"
SEEDS = [900101 + i for i in range(12)]
GRID = np.array(H_GRID_41, dtype=np.float64)


def diag_path(arm: str, seed: int) -> Path:
    if arm == "bsel":
        return BSEL_ROOT / f"bsel_seed{seed}" / f"seed{seed}" / "simulations/diagnostics/event_likelihoods.csv"
    return P3_WORK / f"{arm}_{seed}_work" / f"seed{seed}" / "simulations/diagnostics/event_likelihoods.csv"


def mean_h_from_matrix(vals: np.ndarray) -> float:
    """vals: (n_events, n_nodes) linear per-event likelihoods on GRID."""
    sum_log_l = combine_log_likelihood(vals, "physics_floor")
    weights = moment_weights(GRID, "trapezoid")
    lp = sum_log_l - sum_log_l.max()
    post = np.exp(lp)
    norm = float((post * weights).sum())
    post_n = post / norm
    return float((post_n * GRID * weights).sum())


def arm_fleet(arm: str) -> dict:
    deltas = []
    fulls = []
    pures = []
    gate_max = 0.0
    n_seeds = 0
    for seed in SEEDS:
        p = diag_path(arm, seed)
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        d = df[np.isin(df["h"].to_numpy(np.float64), GRID)].copy()
        beta = d["alpha_G_phi"].to_numpy(np.float64) / d["r_Malm"].to_numpy(np.float64)
        cat_term = beta * d["L_cat_no_bh"].to_numpy(np.float64) / d["D_tilde_phi"].to_numpy(np.float64)
        comp_term = d["B_num"].to_numpy(np.float64) / d["D_tilde_phi"].to_numpy(np.float64)
        d = d.assign(cat_term=cat_term, comp_term=comp_term)

        full = (
            d.pivot_table(index="event_idx", columns="h", values="combined_no_bh", aggfunc="first")
            .reindex(columns=GRID)
            .to_numpy(dtype=np.float64)
        )
        cat = (
            d.pivot_table(index="event_idx", columns="h", values="cat_term", aggfunc="first")
            .reindex(columns=GRID)
            .to_numpy(dtype=np.float64)
        )
        comp = (
            d.pivot_table(index="event_idx", columns="h", values="comp_term", aggfunc="first")
            .reindex(columns=GRID)
            .to_numpy(dtype=np.float64)
        )
        scale = np.maximum(np.abs(full), np.finfo(float).tiny)
        gate_i = float(np.nanmax(np.abs(cat + comp - full) / scale))
        gate_max = max(gate_max, gate_i)

        pure = np.clip(full - cat, 0.0, None)

        mean_h_full = mean_h_from_matrix(full)
        mean_h_pure = mean_h_from_matrix(pure)
        bias_full = mean_h_full - H_TRUE
        bias_pure = mean_h_pure - H_TRUE
        fulls.append(bias_full)
        pures.append(bias_pure)
        deltas.append(bias_pure - bias_full)
        n_seeds += 1

    deltas_arr = np.array(deltas)
    return {
        "n_seeds": n_seeds,
        "bias_full_mean": float(np.mean(fulls)),
        "bias_pure_mean": float(np.mean(pures)),
        "delta_mean": float(np.mean(deltas_arr)),
        "delta_sd": float(np.std(deltas_arr, ddof=1)),
        "delta_sem": float(np.std(deltas_arr, ddof=1) / np.sqrt(n_seeds)),
        "n_positive": int(np.sum(deltas_arr > 0)),
        "gate_i_max_rel": gate_max,
        "per_seed_delta": deltas,
    }


def main() -> None:
    results = {}
    for arm in ("bsel", "fc", "ft"):
        results[arm] = arm_fleet(arm)
        r = results[arm]
        print(
            f"[{arm}] n_seeds={r['n_seeds']} bias_full={r['bias_full_mean']:.5f} "
            f"bias_pure={r['bias_pure_mean']:.5f} delta={r['delta_mean']:.5f} "
            f"+/- {r['delta_sem']:.5f} (SD {r['delta_sd']:.5f}) n_pos={r['n_positive']}/{r['n_seeds']} "
            f"gate_i={r['gate_i_max_rel']:.3e}"
        )

    # --- O2 of-record comparison ---
    o2_path = REPO_ROOT / "results/prod2d_closure_20260818/decompose_impostor_leg_output.json"
    o2 = json.loads(o2_path.read_text())
    print("\nO2 of-record JSON top-level keys:", list(o2.keys()))
    # find the delta_bias-like field
    for k, v in o2.items():
        if isinstance(v, (int, float)) and "delta" in k.lower():
            print(f"  o2[{k!r}] = {v!r}")
    bsel_delta = results["bsel"]["delta_mean"]
    print(f"  independently-rederived bsel delta = {bsel_delta!r}")

    # --- record's claimed decisive numbers, for direct diff ---
    claimed = {
        "ft_bias_full": -0.08444034043042647,
        "ft_bias_pure": 0.03830456905484958,
        "ft_delta_mean": 0.12274490948527605,
        "ft_delta_sem": 0.0077368129010078075,
        "ft_delta_sd": 0.026801106066399762,
        "fc_bias_full": -0.11351,  # from record table (rounded)
        "fc_bias_pure": 0.03830,
        "fc_delta_mean": 0.15181,
        "fc_delta_sem": 0.01071,
    }
    print("\n--- diff vs record's claimed decisive numbers ---")
    for arm, prefix in (("ft", "ft"), ("fc", "fc")):
        r = results[arm]
        for field, claimed_key in (
            ("bias_full_mean", f"{prefix}_bias_full"),
            ("bias_pure_mean", f"{prefix}_bias_pure"),
            ("delta_mean", f"{prefix}_delta_mean"),
        ):
            mine = r[field]
            theirs = claimed[claimed_key]
            print(f"  {claimed_key}: mine={mine:.6f} claimed={theirs:.6f} diff={mine - theirs:.2e}")

    # --- ASSUMPTION-JOIN in-catalogue fraction check ---
    crb_seed61000 = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/headreadout_20260827/seed61000/prepared_cramer_rao_bounds.csv"
    print("\n--- ASSUMPTION-JOIN check ---")
    if crb_seed61000.is_file():
        crb = pd.read_csv(crb_seed61000)
        n_rows = len(crb)
        if "in_catalog" in crb.columns:
            n_in_cat = int(crb["in_catalog"].sum())
            print(f"  seed61000 CRB rows = {n_rows}, in_catalog sum = {n_in_cat}, "
                  f"fraction = {n_in_cat}/{n_rows} = {n_in_cat / n_rows:.6f} "
                  f"(claimed 76/1588 = {76 / 1588:.6f})")
        else:
            print(f"  seed61000 CRB rows = {n_rows}; no 'in_catalog' column -- cannot check directly")
    else:
        print(f"  NOT FOUND: {crb_seed61000}")

    out_path = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/fanout1_20260829/verifier_pass/item5_rederive_output.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
