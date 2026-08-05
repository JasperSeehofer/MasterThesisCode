"""N-2 magnitude probe: the h-tilt the phi-marginal survival S_bar_phi would add
to the 1D completion numerator, under the point-GW (delta in z*) approximation.

Read-only. No source modified.
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
CRB = (
    "/home/jasper/Repositories/MasterThesisCode/results/run_20260804_postfix/"
    "iiib/diagnostics/prepared_cramer_rao_bounds.csv"
)

H_GRID = np.round(np.linspace(0.60, 0.86, 41), 5)

sdp = SimulationDetectionProbability(
    injection_data_dir=POOL,
    snr_threshold=20.0,
    expected_z_max=1.5,
    allow_shallow_pool=True,
)
print("pool loaded", flush=True)

tab = precompute_phi_marginal_survival([float(h) for h in H_GRID], sdp)
print("S_bar_phi tabulated", flush=True)

df = pd.read_csv(CRB)
dl = df["luminosity_distance"].to_numpy(dtype=float)
print("events", dl.size, "dL range", dl.min(), dl.max())

# ln S_bar_phi at the GW peak z*(dL, h) for each event and h
lnS = np.empty((dl.size, H_GRID.size))
zstar = np.empty_like(lnS)
for j, h in enumerate(H_GRID):
    zg, Sg = tab[float(h)]
    zs = np.array([float(dist_to_redshift(d, h=float(h))) for d in dl])
    zstar[:, j] = zs
    s = np.interp(zs, zg, Sg, left=Sg[0], right=Sg[-1])
    lnS[:, j] = np.log(np.clip(s, 1e-300, None))

i73 = int(np.argmin(np.abs(H_GRID - 0.73)))
# central difference in h at 0.73
slope = (lnS[:, i73 + 1] - lnS[:, i73 - 1]) / (H_GRID[i73 + 1] - H_GRID[i73 - 1])
chord = (lnS[:, -1] - lnS[:, 0]) / (H_GRID[-1] - H_GRID[0])

# local log-slope in (1+z) at fixed dL, for interpretation
dln1pz = (np.log(1 + zstar[:, i73 + 1]) - np.log(1 + zstar[:, i73 - 1])) / (
    H_GRID[i73 + 1] - H_GRID[i73 - 1]
)
with np.errstate(divide="ignore", invalid="ignore"):
    exponent = slope / dln1pz

out = {
    "n_events": int(dl.size),
    "S_at_h073": {
        "min": float(np.exp(lnS[:, i73]).min()),
        "q05": float(np.percentile(np.exp(lnS[:, i73]), 5)),
        "median": float(np.median(np.exp(lnS[:, i73]))),
        "q95": float(np.percentile(np.exp(lnS[:, i73]), 95)),
        "max": float(np.exp(lnS[:, i73]).max()),
    },
    "per_event_slope_dlnS_dh_at_073": {
        "min": float(slope.min()),
        "q05": float(np.percentile(slope, 5)),
        "median": float(np.median(slope)),
        "q95": float(np.percentile(slope, 95)),
        "max": float(slope.max()),
        "frac_positive": float((slope > 0).mean()),
    },
    "sum_slope_nats_per_h_at_073": float(slope.sum()),
    "sum_chord_nats_per_h": float(chord.sum()),
    "dln1pz_dh_median": float(np.median(dln1pz)),
    "effective_exponent_dlnS_dln1pz_median": float(np.nanmedian(exponent)),
    "zstar_median_at_073": float(np.median(zstar[:, i73])),
    "h_grid": [float(x) for x in H_GRID],
    "lnS_sum_over_events_per_h": [float(v) for v in lnS.sum(axis=0)],
}
print(json.dumps(out, indent=2))
with open(
    "/tmp/claude-1000/-home-jasper-Repositories-MasterThesisCode/"
    "3bd2c589-fcfd-4f78-82bc-f85781a2e321/scratchpad/n2_sphi_tilt.json",
    "w",
) as f:
    json.dump(out, f, indent=2)
