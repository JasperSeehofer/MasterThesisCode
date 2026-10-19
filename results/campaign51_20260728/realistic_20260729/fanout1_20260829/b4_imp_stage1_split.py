#!/usr/bin/env python3
"""B4.1 [IMP] part 2 -- stage-1 free read 2: split the impostor-leg score into the
GLOBAL mixture-weight h-slope and the PER-EVENT catalogue-leg h-slope (zero evaluate()).

launched under rows #222/#223 -- charter node B4.1 [IMP] part 2

Assembly identity (gated in b4_imp_stage1_forecast.py, GATE I):
    p_i(h) = (beta(h) * L_i(h) + B_i(h)) / D(h),   beta = alpha_G_phi / r_Malm
so with share c_i(h) = beta L_i / (beta L_i + B_i):
    d ln p_i/dh = c_i (s_beta + s_L,i) + (1 - c_i) s_B,i - s_D
    d ln pure_i/dh = s_B,i - s_D
    s_imp,i := d ln p_i/dh - d ln pure_i/dh = c_i (s_beta + s_L,i - s_B,i)
Central differences over 0.725/0.735 are used for every log-derivative and c_i is
taken at the midpoint 0.73 (a first-order split; the exact s_imp from the
forecast script is carried alongside for the residual). s_beta and s_D are
event-independent (verified: max spread printed).

Output: b4_imp_stage1_split.json. Forecast input only; no band, no verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/fanout1_20260829"
P3_WORK = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/p3_work"
SEEDS = [900101 + i for i in range(12)]
DH = 0.735 - 0.725


def piv(df: pd.DataFrame, col: str, h: float) -> pd.Series:
    d = df[np.isclose(df["h"], h)].set_index("event_idx")[col]
    return d


def main() -> int:
    out: dict = {}
    for arm in ("fc", "ft"):
        rows = []
        for seed in SEEDS:
            p = P3_WORK / f"{arm}_{seed}_work" / f"seed{seed}" / "simulations/diagnostics/event_likelihoods.csv"
            df = pd.read_csv(p)
            idx = sorted(df["event_idx"].unique())
            lo, hi, mid = 0.725, 0.735, 0.73
            beta = {h: (piv(df, "alpha_G_phi", h) / piv(df, "r_Malm", h)).reindex(idx) for h in (lo, hi, mid)}
            L = {h: piv(df, "L_cat_no_bh", h).reindex(idx) for h in (lo, hi)}
            B = {h: piv(df, "B_num", h).reindex(idx) for h in (lo, hi, mid)}
            D = {h: piv(df, "D_tilde_phi", h).reindex(idx) for h in (lo, hi)}
            Lm = piv(df, "L_cat_no_bh", mid).reindex(idx)
            s_beta = np.log(beta[hi].to_numpy()) - np.log(beta[lo].to_numpy())
            s_D = np.log(D[hi].to_numpy()) - np.log(D[lo].to_numpy())
            spread_beta = float(np.ptp(s_beta)); spread_D = float(np.ptp(s_D))
            s_beta = float(s_beta.mean()) / DH; s_D_v = float(s_D.mean()) / DH
            Llo, Lhi = L[lo].to_numpy(), L[hi].to_numpy()
            active = (Llo > 0) & (Lhi > 0)
            s_L = np.full(len(idx), np.nan); s_L[active] = (np.log(Lhi[active]) - np.log(Llo[active])) / DH
            s_B = (np.log(B[hi].to_numpy()) - np.log(B[lo].to_numpy())) / DH
            c = (beta[mid].to_numpy() * Lm.to_numpy()) / (beta[mid].to_numpy() * Lm.to_numpy() + B[mid].to_numpy())
            # exact s_imp (as in the forecast script)
            full_lo = (beta[lo].to_numpy() * Llo + B[lo].to_numpy()) / D[lo].to_numpy()
            full_hi = (beta[hi].to_numpy() * Lhi + B[hi].to_numpy()) / D[hi].to_numpy()
            pure_lo = B[lo].to_numpy() / D[lo].to_numpy(); pure_hi = B[hi].to_numpy() / D[hi].to_numpy()
            s_imp_exact = (np.log(full_hi) - np.log(full_lo) - np.log(pure_hi) + np.log(pure_lo)) / DH
            term_global = np.where(active, c * s_beta, 0.0)
            term_event = np.where(active, c * (s_L - s_B), 0.0)
            rows.append({
                "seed": seed, "n": len(idx), "n_active": int(active.sum()),
                "s_beta_global": s_beta, "s_D_global": s_D_v, "spread_s_beta": spread_beta / DH, "spread_s_D": spread_D / DH,
                "mean_c_active": float(c[active].mean()), "mean_c_all": float(c.mean()),
                "mean_s_L_active": float(np.nanmean(s_L[active])), "mean_s_B_active": float(s_B[active].mean()),
                "mean_s_B_all": float(s_B.mean()),
                "mean_term_global": float(term_global.mean()), "mean_term_event": float(term_event.mean()),
                "mean_s_imp_exact": float(s_imp_exact.mean()),
                "mean_s_imp_firstorder": float((term_global + term_event).mean()),
                # weighted mean of (s_L - s_B) over active events, weight c
                "c_weighted_sL_minus_sB": float((c[active] * (s_L[active] - s_B[active])).sum() / c[active].sum()),
            })
        r = pd.DataFrame(rows)
        out[arm] = {
            "per_seed": rows,
            "fleet": {k: {"mean": float(r[k].mean()), "seed_sd": float(r[k].std(ddof=1)), "seed_sem": float(r[k].std(ddof=1) / np.sqrt(len(r)))}
                      for k in ("s_beta_global", "s_D_global", "mean_c_active", "mean_s_L_active", "mean_s_B_active", "mean_term_global",
                                "mean_term_event", "mean_s_imp_exact", "mean_s_imp_firstorder", "c_weighted_sL_minus_sB")},
            "max_spread_s_beta": float(r["spread_s_beta"].max()), "max_spread_s_D": float(r["spread_s_D"].max()),
        }
    (OUT_DIR / "b4_imp_stage1_split.json").write_text(json.dumps(out, indent=1))
    for arm in out:
        print(arm, json.dumps(out[arm]["fleet"], indent=1), "spreads", out[arm]["max_spread_s_beta"], out[arm]["max_spread_s_D"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
