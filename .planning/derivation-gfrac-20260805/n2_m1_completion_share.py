"""N-2 measurement M1: the 1D completion leg's share of the 1D mixture.

Spec (N2_SELECTION_NUMERATOR_DERIVATION_20260805.md.DRAFT section 6.4, M1):
"the 1D mixture's completion-vs-catalogue share per event is *not* measured
(only the 2D's, 95%, section 6.6). It is a free read of event_likelihoods.csv
+ the w_G/beta^phi columns and it rescales P-2's band directly."

Method (path-(A), FIXB_PATHA_PACKAGE.md section 3.2, the ONLY branch active
in the post-fix runs — verified alpha_G_phi/D_tilde_phi/r_Malm are constant
across events at fixed h in both CSVs):

    combined_no_bh_i(h) = (beta_G_phi(h) * L_cat_no_bh_i(h) + B_num_phi_i(h))
                           / D_tilde_phi(h)

    D_tilde_phi(h)      = alpha_G_phi(h) + beta_Gbar_phi(h)          [in CSV: D_tilde_phi, alpha_G_phi]
    beta_Gbar_phi(h)    = D_tilde_phi(h) - alpha_G_phi(h)             (derived)
    beta_G_phi(h)       = alpha_G_phi(h) / r_Malm(h)                  (derived; alpha_G_phi = beta_G_phi * r_Malm)
    B_num_phi_i(h)      = L_comp_i(h) * beta_Gbar_phi(h)              (derived; L_comp := B_num/beta_Gbar
                                                                         is convention-free per code comment
                                                                         at bayesian_statistics.py:4322-4323,
                                                                         so B_num_phi = B_num*beta_Gbar_phi/beta_Gbar
                                                                         = L_comp * beta_Gbar_phi, beta_Gbar cancels)

1D completion share_i(h) = B_num_phi_i(h) / (beta_G_phi(h)*L_cat_no_bh_i(h) + B_num_phi_i(h))
                          = B_num_phi_i(h) / (D_tilde_phi(h) * combined_no_bh_i(h))

The second form is used as the primary read (fewer derived quantities); the
first form (catalog term C_i explicit) is used as a cross-check by
reconstructing combined_no_bh from C_i + B_num_phi_i and comparing to the CSV
column directly (sanity gate — if this does not close to high precision, the
derivation above is wrong and the share numbers below must not be trusted).

Then rescales the derivation draft's naive 1D tilt forecast (n2_sphi_tilt.py,
+207.0 nats/h, which implicitly assumed the completion term is the ENTIRE 1D
mixture for every event, i.e. share_i == 1) to the EXACT correction implied by
the measured per-event share:

    numerator_corr,i(h)   = C_i(h) + B_num_phi_i(h) * S_bar_phi_i(h)
    numerator_uncorr,i(h) = C_i(h) + B_num_phi_i(h)                     (= current implementation)
    numerator_corr,i / numerator_uncorr,i = 1 - share_i(h) * (1 - S_bar_phi_i(h))

    => rescaled per-event ln-tilt_i(h) = ln(1 - share_i(h)*(1 - S_bar_phi_i(h)))

This is EXACT (not a linearization) given share_i(h) and S_bar_phi_i(h); it
reduces to the naive ln(S_bar_phi_i) when share_i == 1.

Read-only. No source modified. No run launched.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")

from master_thesis_code.bayesian_inference.bayesian_statistics import (  # noqa: E402
    precompute_phi_marginal_survival,
)
from master_thesis_code.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from master_thesis_code.physical_relations import dist_to_redshift  # noqa: E402

POOL = (
    "/home/jasper/Repositories/MasterThesisCode/results/campaign51_20260728/"
    "realistic_20260729/gate_b_20260730/injection_pool_mix200k_20260728"
)
BASE = "/home/jasper/Repositories/MasterThesisCode/results/run_20260804_postfix"
VENUES = ["iiib", "joint_r1"]
H_TARGET = 0.73

sdp = SimulationDetectionProbability(
    injection_data_dir=POOL,
    snr_threshold=20.0,
    expected_z_max=1.5,
    allow_shallow_pool=True,
)
print("pool loaded", flush=True)


def quantiles(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "min": float(x.min()),
        "q05": float(np.percentile(x, 5)),
        "q25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "q75": float(np.percentile(x, 75)),
        "q95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }


results: dict[str, dict] = {}

for venue in VENUES:
    print(f"=== venue {venue} ===", flush=True)
    csv_path = f"{BASE}/{venue}/diagnostics/event_likelihoods.csv"
    crb_path = f"{BASE}/{venue}/diagnostics/prepared_cramer_rao_bounds.csv"
    df = pd.read_csv(csv_path)
    crb = pd.read_csv(crb_path)

    h_grid_csv = np.array(sorted(df["h"].unique()), dtype=float)
    print(f"{venue}: {df['event_idx'].nunique()} events, {h_grid_csv.size} h points,"
          f" CRB {len(crb)} rows", flush=True)

    # --- per-h population scalars (verify constant across events at fixed h) ---
    pop = df.groupby("h")[["alpha_G_phi", "D_tilde_phi", "r_Malm", "w_G"]].agg(
        ["mean", "std"]
    )
    max_std_frac = float(
        (pop.xs("std", axis=1, level=1) / pop.xs("mean", axis=1, level=1).abs())
        .replace([np.inf, -np.inf], np.nan)
        .max()
        .max()
    )
    print(f"{venue}: population-scalar constancy check, max rel-std across events "
          f"at fixed h = {max_std_frac:.3e} (should be ~0)", flush=True)

    alpha_G_phi_h = pop.xs("mean", axis=1, level=1)["alpha_G_phi"]
    D_tilde_phi_h = pop.xs("mean", axis=1, level=1)["D_tilde_phi"]
    r_Malm_h = pop.xs("mean", axis=1, level=1)["r_Malm"]
    beta_Gbar_phi_h = D_tilde_phi_h - alpha_G_phi_h
    beta_G_phi_h = alpha_G_phi_h / r_Malm_h

    # --- per-event, per-h share + sanity reconstruction ---
    df = df.merge(
        beta_Gbar_phi_h.rename("beta_Gbar_phi").reset_index(), on="h", how="left"
    )
    df = df.merge(beta_G_phi_h.rename("beta_G_phi").reset_index(), on="h", how="left")
    df = df.merge(D_tilde_phi_h.rename("D_tilde_phi_join").reset_index(), on="h", how="left")

    df["B_num_phi"] = df["L_comp"] * df["beta_Gbar_phi"]
    df["C_i"] = df["beta_G_phi"] * df["L_cat_no_bh"]
    denom = df["C_i"] + df["B_num_phi"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["share_1D"] = np.where(denom > 0, df["B_num_phi"] / denom, np.nan)

    combined_recon = np.where(
        df["D_tilde_phi_join"] > 0, denom / df["D_tilde_phi_join"], np.nan
    )
    finite = np.isfinite(combined_recon) & np.isfinite(df["combined_no_bh"].to_numpy())
    rel_err = np.abs(
        combined_recon[finite] - df["combined_no_bh"].to_numpy()[finite]
    ) / np.maximum(np.abs(df["combined_no_bh"].to_numpy()[finite]), 1e-300)
    print(
        f"{venue}: sanity reconstruction of combined_no_bh — max rel err "
        f"{rel_err.max():.3e}, median {np.median(rel_err):.3e} over {finite.sum()} rows",
        flush=True,
    )

    n_both_zero = int((denom == 0).sum())
    n_total = len(df)

    # --- share stats at h=0.73 ---
    row73 = df[np.isclose(df["h"], H_TARGET)]
    share73 = row73["share_1D"].to_numpy(dtype=float)
    stats73 = quantiles(share73)
    n_dominant73 = int((share73 > 0.99).sum())
    n_zero73 = int((row73["B_num_phi"].to_numpy() <= 0).sum())

    # --- h-dependence of share (grouped mean/median per h) ---
    share_by_h = df.groupby("h")["share_1D"].agg(["mean", "median", "count"])

    # --- exact rescaled tilt: needs S_bar_phi_i(h) at the CSV's own h grid ---
    tab = precompute_phi_marginal_survival([float(h) for h in h_grid_csv], sdp)
    print(f"{venue}: S_bar_phi table built on {h_grid_csv.size} h points", flush=True)

    dl_by_idx = crb["luminosity_distance"].to_numpy(dtype=float)  # row index == event_idx
    n_crb = dl_by_idx.size

    event_idx_all = df["event_idx"].to_numpy(dtype=int)
    valid_idx_mask = event_idx_all < n_crb
    n_out_of_range = int((~valid_idx_mask).sum())

    lnS = np.full(len(df), np.nan)
    for h in h_grid_csv:
        zg, Sg = tab[float(h)]
        m = np.isclose(df["h"].to_numpy(), h) & valid_idx_mask
        idx = event_idx_all[m]
        dl = dl_by_idx[idx]
        zs = np.array([float(dist_to_redshift(d, h=float(h))) for d in dl])
        s = np.interp(zs, zg, Sg, left=Sg[0], right=Sg[-1])
        lnS[m] = np.log(np.clip(s, 1e-300, None))

    df["ln_S_bar_phi"] = lnS
    S_bar_phi = np.exp(df["ln_S_bar_phi"].to_numpy())
    share = df["share_1D"].to_numpy()
    with np.errstate(invalid="ignore"):
        ratio = 1.0 - share * (1.0 - S_bar_phi)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["delta_ln_rescaled"] = np.where(ratio > 0, np.log(ratio), np.nan)
    df["delta_ln_naive"] = df["ln_S_bar_phi"]  # naive = full-share assumption

    per_h_sum_rescaled = df.groupby("h")["delta_ln_rescaled"].sum()
    per_h_sum_naive = df.groupby("h")["delta_ln_naive"].sum()
    per_h_n_valid = df.groupby("h")["delta_ln_rescaled"].apply(lambda s: int(s.notna().sum()))

    h_lo, h_hi = h_grid_csv.min(), h_grid_csv.max()
    chord_rescaled = float(
        (per_h_sum_rescaled.loc[h_hi] - per_h_sum_rescaled.loc[h_lo]) / (h_hi - h_lo)
    )
    chord_naive = float(
        (per_h_sum_naive.loc[h_hi] - per_h_sum_naive.loc[h_lo]) / (h_hi - h_lo)
    )

    # central-difference slope at h=0.73 (need neighbouring grid points)
    h_sorted = np.array(sorted(h_grid_csv))
    i73 = int(np.argmin(np.abs(h_sorted - H_TARGET)))
    if 0 < i73 < h_sorted.size - 1:
        h_m, h_p = h_sorted[i73 - 1], h_sorted[i73 + 1]
        slope_rescaled_73 = float(
            (per_h_sum_rescaled.loc[h_p] - per_h_sum_rescaled.loc[h_m]) / (h_p - h_m)
        )
        slope_naive_73 = float(
            (per_h_sum_naive.loc[h_p] - per_h_sum_naive.loc[h_m]) / (h_p - h_m)
        )
    else:
        slope_rescaled_73 = float("nan")
        slope_naive_73 = float("nan")

    results[venue] = {
        "n_events_csv": int(df["event_idx"].nunique()),
        "n_crb_rows": int(n_crb),
        "n_out_of_range_event_idx": n_out_of_range,
        "population_scalar_max_rel_std_across_events": max_std_frac,
        "combined_no_bh_reconstruction_max_rel_err": float(rel_err.max()),
        "combined_no_bh_reconstruction_median_rel_err": float(np.median(rel_err)),
        "share_1D_at_h073": stats73,
        "n_events_share_gt_099_at_h073": n_dominant73,
        "n_events_B_num_phi_zero_at_h073": n_zero73,
        "n_events_denom_zero_any_h": n_both_zero,
        "n_rows_total": n_total,
        "share_1D_by_h": {
            str(h): {
                "mean": float(share_by_h.loc[h, "mean"]),
                "median": float(share_by_h.loc[h, "median"]),
                "count": int(share_by_h.loc[h, "count"]),
            }
            for h in h_grid_csv
        },
        "rescaled_tilt": {
            "sum_delta_ln_rescaled_nats_at_h073": float(
                per_h_sum_rescaled.loc[h_sorted[i73]]
            ),
            "sum_delta_ln_naive_nats_at_h073": float(per_h_sum_naive.loc[h_sorted[i73]]),
            "n_valid_events_at_h073": int(per_h_n_valid.loc[h_sorted[i73]]),
            "central_diff_slope_rescaled_nats_per_h_at_073": slope_rescaled_73,
            "central_diff_slope_naive_nats_per_h_at_073": slope_naive_73,
            "chord_slope_rescaled_nats_per_h_full_grid": chord_rescaled,
            "chord_slope_naive_nats_per_h_full_grid": chord_naive,
            "h_grid_bounds": [float(h_lo), float(h_hi)],
        },
    }

print(json.dumps(results, indent=2))
with open(
    "/home/jasper/Repositories/MasterThesisCode/.planning/derivation-gfrac-20260805/"
    "n2_m1_completion_share_results.json",
    "w",
) as f:
    json.dump(results, f, indent=2)
