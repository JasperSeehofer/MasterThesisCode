"""AMENDMENT-4 registered pretuning (seed 20271333): z_support sweep at d50x8."""
import json
from pathlib import Path
from darksiren_emri.validation.pp_coverage import PPCoverageConfig, run_coverage, D50_GPC
outdir = Path("pretuning")
for zs in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
    cfg = PPCoverageConfig(n_realizations=8, n_events=250, injected_truths=[0.72], seed=20271333,
        kernel="volume", catalogue_mode=True, mixture_mode="absolute", z_support=zs, sky_frac=1e-4,
        n_galaxies=200_000, mass_channel=True, mass_horizon_index=0.25, selection_cell="fused",
        gw_measurement_scatter=False, h_step=0.004, d50_gpc=D50_GPC * 8)
    res = run_coverage(cfg)
    cf = res["results"]["0.7200"]["completion_fraction"]
    (outdir / f"pretune_vprod_zs{zs}.json").write_text(json.dumps(res))
    landed = 0.30 <= cf <= 0.42  # S-bar leg pinned at 0.950 (far-edge minimum, AMENDMENT-4)
    print(f"zs={zs}: completion={cf:.3f} -> {'LANDED' if landed else 'no'}", flush=True)
    if landed:
        (outdir / "CHOSEN_VPROD.json").write_text(json.dumps(
            {"z_support": zs, "d50_mult": 8, "completion_fraction": cf, "seed": 20271333}))
        break
else:
    print("SWEEP EXHAUSTED -> STOP, return to author (AMENDMENT-5).")
