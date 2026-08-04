"""Adjudication of the g_frac scalar-vs-distribution discrepancy (2026-08-04).

Two agents disagreed:
- interpretation leg claimed g_frac is a per-h near-scalar (<=6 distinct values,
  0.152362 -> 0.158596) and that freezing it moves the 2D MAP to 0.700 both venues;
- viz-data leg found a wide per-event spread and frozen-g MAPs of 0.66/0.64.

This script is the deciding instrument. Verdict [LOCAL, both venues]:
- g_frac has 1587 distinct per-event values at h=0.73 (min/median/max
  0.076187/0.135240/0.241726) -> the near-scalar claim is REFUTED.
- g_frac == B_num_wbh/B_num per event to <=5e-8 (column self-consistent).
- Full 2D proxy MAPs: 0.780 (iiib) / 0.800 (joint_r1); 1D rail 0.600 both.
- Frozen per-event g_frac (each event's own h=0.73 value): 2D MAP -> 0.660
  (iiib) / 0.640 (joint_r1). Frozen event-summed scalar gbar: 0.63 / 0.62.
  The "-> 0.700 both venues" claim is REFUTED in detail; the displacement
  collapse is LARGER than claimed and overshoots below the injected 0.73.
- Event-summed gbar(h) = sum(B_num_wbh)/sum(B_num): 0.134769 (h=0.60) ->
  0.138202 (0.73) -> 0.141337 (0.86); Delta ln = 0.047586 over the grid;
  bit-identical across venues (completion machinery is venue-independent).
- The qualitative finding SURVIVES strengthened: the h-slope of the
  completion-leg mass factor owns the 2D high-h displacement in both venues.
"""

import numpy as np
import pandas as pd

for venue in ["iiib", "joint_r1"]:
    df = pd.read_csv(f"results/run_20260804_postfix/{venue}/diagnostics/event_likelihoods.csv")
    h73 = df[np.isclose(df.h, 0.73)]
    g = h73.g_frac
    print(venue, "| distinct g_frac@0.73:", g.nunique(),
          "| min/median/max:", round(g.min(), 6), round(g.median(), 6), round(g.max(), 6))
    print("   sum(B_num_wbh)/sum(B_num) @0.73:", round(h73.B_num_wbh.sum() / h73.B_num.sum(), 6))
    print("   max |g_frac - B_num_wbh/B_num|:", float((g - h73.B_num_wbh / h73.B_num).abs().max()))

    piv_g = df.pivot(index="event_idx", columns="h", values="g_frac")
    piv_Lc2 = df.pivot(index="event_idx", columns="h", values="L_cat_with_bh")
    piv_Lcomp = df.pivot(index="event_idx", columns="h", values="L_comp")
    piv_w = df.pivot(index="event_idx", columns="h", values="w_tilde_G")

    full = piv_w * piv_Lc2 + (1 - piv_w) * piv_Lcomp * piv_g
    frozen_pe = piv_w * piv_Lc2 + (1 - piv_w) * piv_Lcomp.mul(piv_g[0.73], axis=0)
    print("   full argmax:", np.log(full).sum(axis=0).idxmax(),
          "| frozen(per-event) argmax:", np.log(frozen_pe).sum(axis=0).idxmax())

    gbar = df.groupby("h").B_num_wbh.sum() / df.groupby("h").B_num.sum()
    print("   gbar(h): 0.60=%.6f 0.73=%.6f max-h=%.6f  dln(0.60->max)=%.6f"
          % (gbar[0.600], gbar[0.730], gbar[gbar.index.max()],
             np.log(gbar[gbar.index.max()] / gbar[0.600])))
    frozen_sc = piv_w * piv_Lc2 + (1 - piv_w) * piv_Lcomp * piv_g * (gbar[0.730] / gbar)
    print("   frozen(scalar gbar) argmax:", np.log(frozen_sc).sum(axis=0).idxmax())
