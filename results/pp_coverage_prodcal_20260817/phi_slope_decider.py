"""Decision-4 phi-slope decider (row #122 item 4, audit §C): does bias2d track
the phi slope at fixed sigma_m_gal? Tracks -> instrument defect (missing
phi-prior weight in the 2D catalogue-leg overlap); doesn't -> Malmquist-type
venue noise physics. Exploratory instrument experiment, not verdict-bearing."""
import json
from darksiren_emri.validation.pp_coverage import PPCoverageConfig, run_coverage

BASE = dict(
    n_realizations=20, n_events=250, injected_truths=[0.72], seed=20271111,
    kernel="volume", catalogue_mode=True, mixture_mode="absolute",
    z_support=0.40, sky_frac=1e-4, n_galaxies=200_000,
    mass_channel=True, mass_horizon_index=0.25, selection_cell="off",
    gw_measurement_scatter=False, h_step=0.004,
)
out = {}
for slope in (0.0, -1.0, -2.0, +1.0):
    r = run_coverage(PPCoverageConfig(**BASE, mass_slope=slope))
    b2 = r["results"]["0.7200"]["mass_channel_2d"]["map_bias"]
    b1 = r["results"]["0.7200"]["map_bias"]
    out[str(slope)] = {"bias2d": b2, "bias1d": b1}
    print(f"mass_slope {slope:+.1f}: bias2d {b2:+.5f}  bias1d {b1:+.5f}", flush=True)
open("phi_slope_decider_output.json", "w").write(json.dumps(out, indent=2))
