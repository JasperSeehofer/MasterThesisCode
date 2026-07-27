"""P1 parity audit, part D — exact A-cell translation of the production Σ arms.

Reads the cluster Σ tables (cellApp logs / zmz back-solve, as embedded in
p1a) and the LOCAL production-mode hybrids from p1a_results.json, and pushes
each Σ pair through the exact per-event A-cell formula
Sum_i ln(1 + s_i (r-1)) on the cellApp diagnostics. The on/off, grid-only/off
and on/grid-only rows are tautology checks that must reproduce the ZMZ
readout (-0.51/-1.18, +0.56/+1.01, -1.07/-2.19); the hybrid rows are the
clamp decomposition on the A-cell axis quoted in P1_PARITY_AUDIT.md §2.

Writes p1d_results.json.
"""

import json
import sys

import numpy as np
import pandas as pd

REPO = "/home/jasper/Repositories/MasterThesisCode"
BASE = f"{REPO}/results/lcat_h_dependence_20260725"
HERE = f"{BASE}/mass_ab_20260727/p1_parity"
sys.path.insert(0, REPO)

H_VENUE = [0.60, 0.65, 0.70, 0.73, 0.76, 0.80, 0.86]
SIG_OFF = {0.60: 2.700808e8, 0.65: 2.759635e8, 0.70: 2.815376e8, 0.73: 2.847445e8,
           0.76: 2.878574e8, 0.80: 2.918739e8, 0.86: 2.976373e8}
SIG_ON = {0.60: 2.767679e8, 0.65: 2.826148e8, 0.70: 2.885021e8, 0.73: 2.920316e8,
          0.76: 2.955726e8, 0.80: 3.003153e8, 0.86: 3.074504e8}
SIG_GO = {0.60: 2.838463e8, 0.65: 2.899200e8, 0.70: 2.956500e8, 0.73: 2.989355e8,
          0.76: 3.021176e8, 0.80: 3.062085e8, 0.86: 3.120567e8}

diag = pd.read_csv(f"{BASE}/mass_ab_20260727/cellApp/simulations/diagnostics/event_likelihoods.csv")
p1a = json.load(open(f"{HERE}/p1a_results.json"))
ph = p1a["per_h"]


def acell(ratio_by_h: dict[float, float]) -> dict[str, float]:
    shifts = {}
    for h in H_VENUE:
        dh = diag[np.isclose(diag.h, h)]
        lc = dh.w_G * dh.L_cat_with_bh
        new = dh.combined_with_bh + lc * (1.0 / ratio_by_h[h] - 1.0)
        shifts[h] = float(np.sum(np.log(new) - np.log(dh.combined_with_bh)))
    return {
        "delta_at_080": shifts[0.80] - shifts[0.73],
        "delta_at_086": shifts[0.86] - shifts[0.73],
        "raw_shifts": {str(h): shifts[h] for h in H_VENUE},
    }


pairs = {
    "production_on_vs_off": {h: SIG_ON[h] / SIG_OFF[h] for h in H_VENUE},
    "production_gridonly_vs_off": {h: SIG_GO[h] / SIG_OFF[h] for h in H_VENUE},
    "production_conditioning_only": {h: SIG_ON[h] / SIG_GO[h] for h in H_VENUE},
    "production_hyb_unclamped_vs_off": {
        h: ph[str(h)]["Sigma_hyb_unclamped_switched"] / ph[str(h)]["Sigma_off"] for h in H_VENUE
    },
    "production_hyb_clamped_vs_off": {
        h: ph[str(h)]["Sigma_hyb_clamped_switched"] / ph[str(h)]["Sigma_off"] for h in H_VENUE
    },
}
res = {k: acell(v) for k, v in pairs.items()}
res["sum_s_i_2D"] = {
    str(h): float(
        np.sum(
            (lambda d: d.w_G * d.L_cat_with_bh / d.combined_with_bh)(
                diag[np.isclose(diag.h, h)]
            )
        )
    )
    for h in (0.73, 0.80, 0.86)
}
with open(f"{HERE}/p1d_results.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "raw_shifts"} if isinstance(v, dict) else v for k, v in res.items()}, indent=1))
