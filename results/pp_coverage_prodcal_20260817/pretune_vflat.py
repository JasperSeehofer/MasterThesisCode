"""AMENDMENT-2 registered pretuning: d50 multiplier sweep (seed 20271222)."""
import json
import numpy as np
from darksiren_emri.validation.pp_coverage import (
    PPCoverageConfig, run_coverage, phi_marginal_survival_table, D50_GPC, W_PDET_GPC,
)
from pathlib import Path

outdir = Path("pretuning"); outdir.mkdir(exist_ok=True)
for mult in (2, 3, 4, 6):
    # min-window S-bar at h=0.72 (estimator-side diagnostic)
    zg, sb = phi_marginal_survival_table(
        np.array([0.72]), mass_slope=0.0, mass_horizon_index=0.25,
        d50=D50_GPC * mult, w_pdet=W_PDET_GPC, n_z=200, n_mass_quad=200)
    win = (zg > 0.40) & (zg <= 0.95)
    smin = float(sb[win, 0].min())
    cfg = PPCoverageConfig(
        n_realizations=8, n_events=250, injected_truths=[0.72], seed=20271222,
        kernel="volume", catalogue_mode=True, mixture_mode="absolute",
        z_support=0.40, sky_frac=1e-4, n_galaxies=200_000,
        mass_channel=True, mass_horizon_index=0.25, selection_cell="fused",
        gw_measurement_scatter=False, h_step=0.004, d50_gpc=D50_GPC * mult)
    res = run_coverage(cfg)
    cf = res["results"]["0.7200"]["completion_fraction"]
    (outdir / f"pretune_vflat_m{mult}.json").write_text(json.dumps(res))
    landed = smin >= 0.85 and cf >= 0.20
    print(f"mult={mult}: min-window S_bar={smin:.3f} completion={cf:.3f} -> {'LANDED' if landed else 'no'}", flush=True)
    if landed:
        (outdir / "CHOSEN_VFLAT.json").write_text(json.dumps(
            {"d50_mult": mult, "min_window_Sbar": smin, "completion_fraction": cf, "seed": 20271222}))
        print(f"FROZEN: d50 multiplier {mult}")
        break
else:
    print("SWEEP EXHAUSTED — STOP, return to author (AMENDMENT-3 discipline).")
