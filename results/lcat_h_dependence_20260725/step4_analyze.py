"""Step 4 — factor attribution, P1/P2/P3 tests, and summary JSON.

Consumes decomposition_results.json (step 3), global_sums.json (step 2), the
shipped diagnostics CSV (all 1461 host-found events), and the shipped
combined_posterior.json (D(h)) + diagnostics w_G(h) for beta_G(h).

All slopes are least-squares d ln X / dh over the 41-value grid (units: ln-e per
unit h; multiply by 0.26 for the full-grid span).
"""

import json

import numpy as np
import pandas as pd

OUT = "/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725"
RUN = "/home/jasper/Repositories/MasterThesisCode/results/campaign_phase2_runs/run_20260719_seed1000_exp40"

res = json.load(open(f"{OUT}/decomposition_results.json"))
glob_sums = {float(k): v for k, v in json.load(open(f"{OUT}/global_sums.json"))["no_bh"].items()}
h = np.asarray(res["h_grid"])
G = np.asarray([glob_sums[x] for x in h])


def slope(y):
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        return float("nan")
    return float(np.polyfit(h, np.log(y), 1)[0])


# ---- shipped global structure: D(h), w_G(h), beta_G(h) ----
comb = json.load(open(f"{RUN}/simulations/posteriors/combined_posterior.json"))
D_h = {
    round(float(hv), 4): float(v) for hv, v in zip(comb["h_values"], comb["D_h_per_h"])
}
diag = pd.read_csv(f"{RUN}/simulations/diagnostics/event_likelihoods.csv")
w_G_curve = diag.groupby("h")["w_G"].first()
assert np.allclose(w_G_curve.index.values, h)
w_G = w_G_curve.values
D_curve = np.asarray([D_h[round(float(x), 4)] for x in h])
beta_G = w_G * D_curve
S_lnbetaG = slope(beta_G)
S_lnD = slope(D_curve)
S_lnG = slope(G)

summary = {
    "h_grid_span": [float(h[0]), float(h[-1])],
    "global_curves_slopes_dlnperdh": {
        "beta_G": S_lnbetaG,
        "D": S_lnD,
        "w_G": slope(w_G),
        "Sigma_global_no_bh": S_lnG,
    },
    "events": {},
}

rows = []
for k, ev in res["events"].items():
    c = ev["curves"]
    S_L = slope(c["L_cat_kernel_vd"])
    S_N = slope(c["sumN_vd"])
    S_D = slope(c["sumD_vd"])
    S_L_bare = slope(c["L_cat_bare"])
    S_L_glob = slope(c["L_cat_global"])
    S_N_bare = slope(c["sumN_bare"])
    S_D_bare = slope(c["sumD_bare"])
    S_N_winfr = slope(c["sumN_win_frozen"])  # integrand-only contribution
    S_N_intfr = slope(c["sumN_integrand_frozen"])  # window-movement-only contribution
    S_D_pdfr = slope(c["sumD_pdet_map_frozen"])  # should be ~0 (prior h-invariant)
    # P1 surgical swap: numerator unchanged, denominator h-shape replaced by global
    S_L_swap = S_N - S_lnG
    row = {
        "event": int(k),
        "role": ev["role"],
        "z_inj": ev["z_inj"],
        "n_hosts": ev["n_hosts"],
        "S_lnLcat": S_L,
        "S_lnSumN": S_N,
        "S_lnSumD": S_D,
        "S_num_integrand": S_N_winfr,
        "S_num_window": S_N_intfr,
        "num_additivity_resid": S_N - (S_N_winfr + S_N_intfr - slope(
            [c["sumN_vd"][res["h_grid"].index(res["h_ref"])]] * len(h)
        ) if False else S_N_winfr + S_N_intfr),
        "S_D_pdet_frozen": S_D_pdfr,
        "S_lnLcat_bare": S_L_bare,
        "S_lnLcat_global": S_L_glob,
        "S_lnLcat_swapdenom": S_L_swap,
        "kernel_effect_vd_minus_bare": S_L - S_L_bare,
        "share_denominator": (-S_D) / S_L if S_L != 0 else float("nan"),
        "share_numerator": S_N / S_L if S_L != 0 else float("nan"),
        "validation_maxrel_vs_shipped": ev["validation_maxrel_vs_shipped"],
        "mirror_maxrel": max(c["kernel_batch_vs_mirror_maxrel"]),
        "Zg_scale_maxdev": max(c["Zg_scale_check"]),
        "prior_hinv_maxdev": max(c["prior_hinv_maxdev"]),
        "h_star_med": ev["h_star_summary"]["med"],
        "h_star_frac_below_060": ev["h_star_summary"]["frac_below_060"],
    }
    rows.append(row)
    summary["events"][k] = row

df = pd.DataFrame(rows).sort_values("z_inj")
pd.set_option("display.width", 250)
print(
    df[
        [
            "event",
            "z_inj",
            "n_hosts",
            "S_lnLcat",
            "S_lnSumN",
            "S_lnSumD",
            "S_num_integrand",
            "S_num_window",
            "S_lnLcat_bare",
            "S_lnLcat_global",
            "S_lnLcat_swapdenom",
            "h_star_med",
        ]
    ].to_string(index=False)
)

# ---- P1: correlation of denominator slope with total tilt (instrumented events) ----
mask = np.isfinite(df.S_lnLcat) & np.isfinite(df.S_lnSumD)
r_p1 = np.corrcoef(-df.S_lnSumD[mask], -df.S_lnLcat[mask])[0, 1]
deep = df[df.z_inj >= 0.25]
r_p1_deep = (
    np.corrcoef(-deep.S_lnSumD, -deep.S_lnLcat)[0, 1] if len(deep) > 2 else float("nan")
)
summary["P1"] = {
    "corr_dlnSumD_vs_dlnLcat_all": float(r_p1),
    "corr_deep_z_ge_0.25": float(r_p1_deep),
    "mean_share_denominator": float(np.nanmean(df.share_denominator)),
    "surgical_swap_slopes": {
        str(int(e)): float(s) for e, s in zip(df.event, df.S_lnLcat_swapdenom)
    },
}

# ---- P2: full-population per-event tilt vs -dln beta_G/dh ----
piv = diag.pivot_table(index="event_idx", columns="h", values="L_cat_no_bh")
piv = piv[sorted(piv.columns)]
hf = piv[(piv > 0).all(axis=1)]
lnL = np.log(hf.values)
S_ev = np.polyfit(h, lnL.T, 1)[0]  # per-event d ln L_cat / dh
summary["P2"] = {
    "minus_dlnbetaG_dh": -S_lnbetaG,
    "dlnSigmaGlobal_dh": S_lnG,
    "per_event_dlnLcat_dh": {
        "mean": float(np.mean(S_ev)),
        "median": float(np.median(S_ev)),
        "q16": float(np.quantile(S_ev, 0.16)),
        "q84": float(np.quantile(S_ev, 0.84)),
    },
    "note": "Gray-consistency predicts per-event tilt ~ -dln beta_G/dh if the "
    "local-ball denominator is the defect and beta_G*L_cat should be ~flat",
}
print("\nP2: -dln beta_G/dh =", -S_lnbetaG, " dln Sigma_global/dh =", S_lnG)
print("    per-event dlnLcat/dh mean/med:", np.mean(S_ev), np.median(S_ev))

# ---- P3: per-host tilt vs sigma_z at fixed z (within-event regression) ----
p3_rows = []
for k, ev in res["events"].items():
    ph = ev.get("per_host")
    if ph is None:
        continue
    N0, N1 = np.asarray(ph["N_0.60"]), np.asarray(ph["N_0.86"])
    D0, D1 = np.asarray(ph["D_0.60"]), np.asarray(ph["D_0.86"])
    z = np.asarray(ph["z"])
    sz = np.asarray(ph["sig_z_eff"])
    ok = (D0 > 0) & (D1 > 0)
    tilt_D = np.log(D0[ok] / D1[ok]) / (h[0] - h[-1])  # d ln D_g/dh (2-pt)
    zz, ss = z[ok], sz[ok]
    if len(zz) > 10 and np.std(ss) > 0:
        X = np.column_stack([np.ones_like(zz), zz, ss])
        coef, *_ = np.linalg.lstsq(X, tilt_D, rcond=None)
        p3_rows.append(
            {
                "event": int(k),
                "z_inj": ev["z_inj"],
                "n": int(ok.sum()),
                "dlnDg_dh_vs_sigma_z_coef": float(coef[2]),
                "dlnDg_dh_vs_z_coef": float(coef[1]),
                "mean_dlnDg_dh": float(np.mean(tilt_D)),
            }
        )
summary["P3"] = p3_rows
print("\nP3 (per-host d lnD_g/dh regression on [1, z_g, sigma_z]):")
for r in p3_rows:
    print(
        f"  ev {r['event']:5d} z_inj={r['z_inj']:.2f} n={r['n']:5d} "
        f"mean={r['mean_dlnDg_dh']:+.2f}  dz-coef={r['dlnDg_dh_vs_z_coef']:+.2f} "
        f"dsigma-coef={r['dlnDg_dh_vs_sigma_z_coef']:+.2f}"
    )

with open(f"{OUT}/analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print(f"\nwrote {OUT}/analysis_summary.json")
