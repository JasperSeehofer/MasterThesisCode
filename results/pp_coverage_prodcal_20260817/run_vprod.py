"""AMENDMENT-4 registered V-prod cells (seed 20271218 shared across pair)."""
import json, time
from pathlib import Path
from darksiren_emri.validation.pp_coverage import PPCoverageConfig, run_coverage, D50_GPC
zs = json.loads(Path("pretuning/CHOSEN_VPROD.json").read_text())["z_support"]
for sc in ("fused", "off"):
    cid = f"vprod_250_production_{sc}"
    out = Path("cells") / f"{cid}.json"
    if out.exists():
        print(f"{cid}: SKIP"); continue
    t0 = time.perf_counter()
    res = run_coverage(PPCoverageConfig(
        n_realizations=120, n_events=250, injected_truths=[0.62, 0.72, 0.84],
        seed=20271218, kernel="volume", catalogue_mode=True, mixture_mode="absolute",
        z_support=zs, sky_frac=1e-4, n_galaxies=200_000, mass_channel=True,
        mass_horizon_index=0.25, selection_cell=sc, gw_measurement_scatter=False,
        h_step=0.004, d50_gpc=D50_GPC * 8))
    out.write_text(json.dumps(res))
    print(f"{cid}: done in {time.perf_counter()-t0:.0f}s", flush=True)
print("VPROD COMPLETE")
