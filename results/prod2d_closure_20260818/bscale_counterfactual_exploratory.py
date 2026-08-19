"""Read-only counterfactual: what do the production 2D/1D posteriors do when the
B_scale = beta_Gbar_phi(h)/beta_Gbar(h) factor on the completion leg is (a) frozen
at its h=0.73 value, (b) removed (set to 1)?

Assembly (path-A branch, bayesian_statistics.py:4907-4914):
  combined_no_bh  = (beta_G_phi*L_cat_no_bh  + B_num    *B_scale)/D_tilde_phi
  combined_with_bh= (alpha_G_phi*L_cat_with_bh + B_num_wbh*B_scale)/D_tilde_phi
CSV stores combined_*, alpha_G_phi, D_tilde_phi, L_cat_*, B_num, B_num_wbh (raw).
B_scale(h) is recovered per (venue,h) from rows with L_cat_with_bh == 0:
  B_scale = combined_with_bh * D_tilde_phi / B_num_wbh.
beta_G_phi is recovered from any row with L_cat_no_bh > 0:
  beta_G_phi = (combined_no_bh*D_tilde - B_num*B_scale)/L_cat_no_bh.
Posterior: uniform prior, gradient-trapezoid weights over the h grid (P7-2a),
sum log likelihood over events (canonical raw Sigma log L).
"""

import numpy as np
import pandas as pd

REPO = "/home/jasper/Repositories/darksiren-emri"


def posterior_mean(h, logL):
    # gradient-trapezoid weights
    w = np.gradient(h)
    logL = logL - logL.max()
    p = np.exp(logL) * w
    p /= p.sum()
    return float((p * h).sum()), h[np.argmax(logL)]


for venue in ("iiib", "joint_r1"):
    df = pd.read_csv(f"{REPO}/results/run_20260804_postfix/{venue}/diagnostics/event_likelihoods.csv")
    hs = np.array(sorted(df.h.unique()))
    # recover B_scale(h)
    bs = {}
    for h in hs:
        sub = df[(df.h == h) & (df.L_cat_with_bh == 0.0) & (df.B_num_wbh > 0)]
        vals = sub.combined_with_bh * sub.D_tilde_phi / sub.B_num_wbh
        bs[h] = float(np.median(vals))
    bs_arr = np.array([bs[h] for h in hs])
    b73 = bs[0.73] if 0.73 in bs else np.interp(0.73, hs, bs_arr)
    print(f"\n=== {venue}:  B_scale range [{bs_arr.min():.6f}, {bs_arr.max():.6f}]  "
          f"@0.73={b73:.6f}  dln/dh={np.polyfit(hs, np.log(bs_arr), 1)[0]:.4f}")

    # recover beta_G_phi(h) from 1D rows with catalogue support
    bgphi = {}
    for h in hs:
        sub = df[(df.h == h) & (df.L_cat_no_bh > 0)]
        vals = (sub.combined_no_bh * sub.D_tilde_phi - sub.B_num * bs[h]) / sub.L_cat_no_bh
        bgphi[h] = float(np.median(vals))

    piv = {c: df.pivot(index="event_idx", columns="h", values=c).reindex(columns=hs)
           for c in ("alpha_G_phi", "D_tilde_phi", "L_cat_no_bh", "L_cat_with_bh",
                     "B_num", "B_num_wbh", "combined_no_bh", "combined_with_bh")}

    scen = {
        "production (as banked)": lambda h: bs[h],
        "B_scale frozen @0.73": lambda h: b73,
        "B_scale removed (=1)": lambda h: 1.0,
    }
    for name, f in scen.items():
        out = {}
        for ch, lcat_col, bnum_col, w_col in (
            ("2D", "L_cat_with_bh", "B_num_wbh", "alpha_G_phi"),
            ("1D", "L_cat_no_bh", "B_num", None),
        ):
            logL = np.zeros(len(hs))
            for j, h in enumerate(hs):
                wcat = piv["alpha_G_phi"][h] if ch == "2D" else bgphi[h]
                comb = (wcat * piv[lcat_col][h] + piv[bnum_col][h] * f(h)) / piv["D_tilde_phi"][h]
                comb = np.maximum(comb.values, 1e-300)
                logL[j] = np.log(comb).sum()
            m, mp = posterior_mean(hs, logL)
            out[ch] = (m, mp)
        print(f"  {name:28s}  2D mean={out['2D'][0]:.4f} map={out['2D'][1]:.3f}   "
              f"1D mean={out['1D'][0]:.4f} map={out['1D'][1]:.3f}")

    # sanity: reproduce banked combined columns
    for ch, col in (("2D", "combined_with_bh"), ("1D", "combined_no_bh")):
        logL = np.array([np.log(np.maximum(piv[col][h].values, 1e-300)).sum() for h in hs])
        m, mp = posterior_mean(hs, logL)
        print(f"  banked {ch}: mean={m:.4f} map={mp:.3f}")
