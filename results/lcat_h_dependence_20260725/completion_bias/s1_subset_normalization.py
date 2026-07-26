"""S1 — Fallback-ensemble subset-normalization test (mechanism 0/4).

The fallback per-event likelihood is p_i(h) = B_num_i(h)/D(h). Decompose

    log p_i = log(B_num_i / beta_Gbar) + log(beta_Gbar(h)/D(h))
            = [conditioned dark-channel likelihood] + [mixture weight P(dark|det,h)]

For a SUBSET consisting exactly of the dark/zero-host events, only the first
term is an h-estimator of the dark channel data; the second is the (correct,
full-ensemble) mixture weight whose h-slope tilts the subset peak. This script
measures both peaks from the shipped diagnostics, with NO model rebuilding:
  - Sigma_fb log(B_num/D)       (replicates the 0.612 peak)
  - Sigma_fb log(B_num/beta_Gbar) = Sigma_fb log L_comp  (conditioned)
  - N_fb * log(beta_Gbar/D) shape (the tilt term)
Also extracts D(h), beta_Gbar(h), beta_G(h) tables from the 41 per-h run logs.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path("/home/jasper/Repositories/MasterThesisCode/results/campaign_phase2_runs/run_20260719_seed1000_exp40")
OUT = Path("/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725/completion_bias")

# --- tables from logs -------------------------------------------------------
D_re = re.compile(r"D\(h=([0-9.]+)\) = ([0-9.eE+-]+)")
Bg_re = re.compile(r"beta_Gbar\(h=([0-9.]+)\) = ([0-9.eE+-]+)")
D_h: dict[float, float] = {}
bgbar_h: dict[float, float] = {}
for log in RUN.glob("master_thesis_code_*.log"):
    t = log.read_text()
    for m in D_re.finditer(t):
        D_h[round(float(m.group(1)), 4)] = float(m.group(2))
    for m in Bg_re.finditer(t):
        bgbar_h[round(float(m.group(1)), 4)] = float(m.group(2))
h_grid = np.array(sorted(D_h))
assert len(h_grid) == 41, len(h_grid)
D = np.array([D_h[h] for h in h_grid])
bgbar = np.array([bgbar_h[h] for h in h_grid])
bg = D - bgbar

# --- fallback set from gzipped eval logs ------------------------------------
import gzip

det_re = re.compile(r"Detection (\d+): no catalogue hosts")
fb: set[int] = set()
for err in sorted((RUN / "logs").glob("evaluate_*.err.gz")):
    fb |= {int(m.group(1)) for m in det_re.finditer(gzip.open(err, "rt").read())}
print("n_fallback:", len(fb))

# --- diagnostics -------------------------------------------------------------
diag = pd.read_csv(RUN / "simulations" / "diagnostics" / "event_likelihoods.csv")
fb_sorted = sorted(fb)
diag_fb = diag[diag["event_idx"].isin(fb)]

def peak(hvals: np.ndarray, y: np.ndarray) -> dict:
    i = int(np.argmax(y))
    out = {"argmax_h": float(hvals[i]), "railed": i in (0, len(hvals) - 1)}
    if 0 < i < len(hvals) - 1:
        hm, h0, hp = hvals[i - 1 : i + 2]
        ym, y0, yp = y[i - 1 : i + 2]
        denom = ym - 2 * y0 + yp
        out["parabolic_h"] = float(h0 - 0.5 * (hp - hm) * (yp - ym) / (2 * denom))
        d2 = 2 * (ym / ((h0 - hm) * (hp - hm)) - y0 / ((hp - h0) * (h0 - hm)) + yp / ((hp - h0) * (hp - hm)))
        out["sigma"] = float(np.sqrt(-1.0 / d2)) if d2 < 0 else None
    return out

# per-h sums over fallback events
sums = diag_fb.groupby("h").agg(
    sum_log_p=("combined_no_bh", lambda x: np.sum(np.log(x))),
    sum_log_Lcomp=("L_comp", lambda x: np.sum(np.log(x))),
    n=("event_idx", "count"),
)
sums = sums.reindex(np.round(h_grid, 4))
assert (sums["n"] == len(fb)).all()

res = {
    "n_fallback": len(fb),
    "h_grid": h_grid.tolist(),
    "D_h": D.tolist(),
    "beta_Gbar_h": bgbar.tolist(),
    "beta_G_h": bg.tolist(),
    "wG_h(beta_G/D)": (bg / D).tolist(),
    "P_dark_det(beta_Gbar/D)": (bgbar / D).tolist(),
    "peak_sum_log_BnumOverD": peak(h_grid, sums["sum_log_p"].to_numpy()),
    "peak_sum_log_BnumOverBetaGbar": peak(h_grid, sums["sum_log_Lcomp"].to_numpy()),
    "tilt_term_Nfb_log_bgbar_over_D": (len(fb) * np.log(bgbar / D)).tolist(),
}
# cross-check: log L_comp + log(bgbar/D) == log p
chk = sums["sum_log_Lcomp"].to_numpy() + len(fb) * np.log(bgbar / D) - sums["sum_log_p"].to_numpy()
res["identity_check_max_abs"] = float(np.max(np.abs(chk)))

# tilt slope around truth
i73 = int(np.argmin(np.abs(h_grid - 0.73)))
slope = np.gradient(len(fb) * np.log(bgbar / D), h_grid)
res["tilt_slope_at_073_logL_per_h"] = float(slope[i73])
res["conditioned_curvature_proxy"] = peak(h_grid, sums["sum_log_Lcomp"].to_numpy())

with open(OUT / "s1_results.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2))
print("P_dark(0.60)=%.4f  P_dark(0.73)=%.4f  P_dark(0.86)=%.4f" % (bgbar[0]/D[0], bgbar[i73]/D[i73], bgbar[-1]/D[-1]))
