"""S2 — detectability vs M_z near the current cap (issue #51, deliverable 5).

Uses the canonical deep pool (50,000 rows, z <= 1.5, M column = DETECTOR-frame
M_z with the symmetric truncation at 1e6: main.py injection_campaign skips
M_z > parameter_space.M.upper_limit). Question: at the top of the current
support, is the detection horizon d_hor = SNR * d_L / 20 still rising with
M_z? If yes, detections continue above 10^6 and the upper bound CANNOT be
narrowed without a pilot measurement.

Outputs cap_analysis.json.
"""

import glob
import json

import numpy as np
import pandas as pd

BASE = "/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725"
SNR_THR = 20.0

pool = pd.concat(
    [pd.read_csv(f) for f in sorted(glob.glob(f"{BASE}/data/injections/injection_h_0p73_task_*.csv"))],
    ignore_index=True,
)
lm = np.log10(pool["M"].to_numpy(np.float64))  # detector-frame log10 M_z
snr = pool["SNR"].to_numpy(np.float64)
dl = pool["luminosity_distance"].to_numpy(np.float64)
z = pool["z"].to_numpy(np.float64)
d_hor = snr * dl / SNR_THR

edges = np.arange(4.5, 6.01, 0.1)
rows = []
for a, b in zip(edges[:-1], edges[1:]):
    m = (lm >= a) & (lm < b)
    n = int(m.sum())
    if n == 0:
        continue
    dh = d_hor[m]
    rows.append({
        "lm_bin": [round(float(a), 2), round(float(b), 2)],
        "n": n,
        "det_frac_snr20": float((snr[m] >= SNR_THR).mean()),
        "d_hor_median_Gpc": float(np.median(dh)),
        "d_hor_p90_Gpc": float(np.quantile(dh, 0.90)),
        "d_hor_p99_Gpc": float(np.quantile(dh, 0.99)),
        "d_hor_max_Gpc": float(dh.max()),
        "z_median": float(np.median(z[m])),
    })

# top-half-dex slope check: is p90(d_hor) rising with lm at the boundary?
top = [r for r in rows if r["lm_bin"][0] >= 5.5]
lm_c = np.array([0.5 * (r["lm_bin"][0] + r["lm_bin"][1]) for r in top])
p90 = np.array([r["d_hor_p90_Gpc"] for r in top])
slope_p90 = float(np.polyfit(lm_c, np.log(p90), 1)[0])

res = {
    "pool": {"n": int(len(pool)), "lm_range": [float(lm.min()), float(lm.max())],
             "note": "M column is detector-frame M_z, truncated at 1e6 "
                     "(main.py symmetric M_z truncation)"},
    "bins": rows,
    "dlog_dhor_p90_dlm_top_half_dex": slope_p90,
    "d_hor_max_top_bin_Gpc": rows[-1]["d_hor_max_Gpc"],
    "d_hor_p90_top_bin_Gpc": rows[-1]["d_hor_p90_Gpc"],
}
with open(f"{BASE}/campaign_sizing_20260728/cap_analysis.json", "w") as f:
    json.dump(res, f, indent=1)
for r in rows:
    print(f"lm {r['lm_bin']}: n={r['n']:5d} det={r['det_frac_snr20']:.3f} "
          f"dhor med={r['d_hor_median_Gpc']:.3f} p90={r['d_hor_p90_Gpc']:.3f} "
          f"p99={r['d_hor_p99_Gpc']:.3f} max={r['d_hor_max_Gpc']:.3f}")
print("slope dln(p90 d_hor)/dlm over [5.5,6.0]:", slope_p90)
