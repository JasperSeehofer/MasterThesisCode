"""Supporting computations for DERIVATION_ESTIMATOR_REDESIGN.md (2026-07-25).

Reads the shipped seed1000 EXP-40 per-event diagnostics CSV and the D1
instrumented curves (decomposition_results.json, global_sums.json) and produces
the three evidence tables quoted in the derivation document:

  T1 — catalogue-vs-dark weight ratios at h=0.73 under the current
       (self-normalized) estimator vs the absolute-mass (Variant 1) estimator.
  T2 — per-event d ln p_i/dh log-slopes: current, Variant 1, Variant 2
       (both dark-branch pairings), and the pure dark term.
  T3 — ball-vs-global selection-sum scale factors S_i = Sigma_glob/Sigma_ball
       and the commensurability-constant slope d ln n_bar_w/dh check.

No pipeline code is touched; everything is read from shipped artifacts.
Run from the repo root:  python3 results/lcat_h_dependence_20260725/variant_slope_tables.py
"""

import json

import numpy as np
import pandas as pd

CSV = "results/campaign_phase2_runs/run_20260719_seed1000_exp40/simulations/diagnostics/event_likelihoods.csv"
DECOMP = "results/lcat_h_dependence_20260725/decomposition_results.json"
GLOBALS = "results/lcat_h_dependence_20260725/global_sums.json"

# From D1 P2 (shipped w_G * D tables): -d ln beta_G/dh = +3.37 at the EXP-40 venue.
DLN_BETA_G_DH = -3.37


def slope(y: np.ndarray, h: np.ndarray) -> float:
    m = np.isfinite(y) & (y > 0)
    if m.sum() < 5:
        return float("nan")
    return float(np.polyfit(h[m], np.log(y[m]), 1)[0])


def main() -> None:
    df = pd.read_csv(CSV)
    d = json.load(open(DECOMP))
    g = json.load(open(GLOBALS))["no_bh"]
    hg = np.array(d["h_grid"])
    i73 = int(np.argmin(np.abs(hg - 0.73)))
    piv = {
        c: df.pivot_table(index="event_idx", columns="h", values=c)
        for c in ["w_G", "L_cat_no_bh", "B_num", "L_comp", "combined_no_bh"]
    }
    hcols = np.array(sorted(piv["w_G"].columns))
    sg = np.array([g["%.4f" % h] for h in hg])
    s_glob = float(np.polyfit(hg, np.log(sg), 1)[0])

    print("T3: d ln Sigma_glob/dh = %+0.3f ;  d ln n_bar_w/dh = %+0.3f (3/h = %0.3f)"
          % (s_glob, s_glob - DLN_BETA_G_DH, 3.0 / 0.73))
    hdr = ("ev", "z", "n", "S_cur", "S_V1", "S_V2(bGbar)", "S_V2(B/D)", "S_dark",
           "A/B@.73", "lam@.73", "S_i")
    print(("%5s %5s %5s" + " %11s" * 8) % hdr)
    for k, ev in d["events"].items():
        i = int(k)
        wG = piv["w_G"].loc[i][hcols].to_numpy()
        Lcomp = piv["L_comp"].loc[i][hcols].to_numpy()
        Lcat = piv["L_cat_no_bh"].loc[i][hcols].to_numpy()
        cur = piv["combined_no_bh"].loc[i][hcols].to_numpy()
        LcatG = np.interp(hcols, hg, np.array(ev["curves"]["L_cat_global"]))
        A = wG * LcatG            # = A_i/D  (absolute catalogue mass over D)
        B = (1.0 - wG) * Lcomp    # = B_num/D
        v1 = A + B
        lam = np.where(A + B > 0, A / (A + B), 0.0)
        v2_gbar = lam * Lcat + (1.0 - lam) * Lcomp   # dark branch B/beta_Gbar
        v2_bd = lam * Lcat + (1.0 - lam) * B         # dark branch B/D
        j73 = int(np.argmin(np.abs(hcols - 0.73)))
        s_i = g["0.7300"] / ev["curves"]["sumD_vd"][i73]
        print(("%5d %5.2f %5d" + " %+11.2f" * 5 + " %11.2e %11.2e %11.2e")
              % (i, ev["z_inj"], ev["n_hosts"], slope(cur, hcols), slope(v1, hcols),
                 slope(v2_gbar, hcols), slope(v2_bd, hcols), slope(B, hcols),
                 (A[j73] / B[j73]) if B[j73] > 0 else float("inf"), lam[j73], s_i))


if __name__ == "__main__":
    main()
