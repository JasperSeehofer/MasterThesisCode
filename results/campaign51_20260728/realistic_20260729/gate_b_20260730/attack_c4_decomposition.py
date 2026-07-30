"""Gate B attack on C4-as-mechanism: exact decomposition of the dark class's
+15.83-nat 2D-1D channel difference (RUNBOOK_NEXT_SESSION_6.md §4 item 1).

The claim's STORY is "impostor rejection -> completion fallback ... which pulls
up".  This script tests that story with an EXACT algebraic decomposition that is
fully computable from
    seed61000/real_r1/diagnostics/event_likelihoods.csv
(65108 rows = 1588 events x 41 h; columns event_idx,h,w_G,L_cat_no_bh,
L_cat_with_bh,B_num,L_comp,combined_no_bh,combined_with_bh).

Mixture structure (verified to 2.8e-16 by the previous session, re-verified here):

    p_i^ch(h) = w_G(h) L_cat,i^ch(h) + (1 - w_G(h)) L_comp,i(h)

with w_G and L_comp CHANNEL-COMMON.  Factor out the completion leg:

    C_i(h)  := (1 - w_G(h)) L_comp,i(h)          [channel-common]
    R_i^ch(h) := w_G(h) L_cat,i^ch(h) / C_i(h)   [catalogue-to-completion odds]
    ln p_i^ch(h) = ln C_i(h) + ln(1 + R_i^ch(h))

Therefore, for the channel difference of the h = 0.73 -> 0.81 tilt,

    Delta_i := [Dln p_i^2D] - [Dln p_i^1D]
             = Dln(1 + R_i^2D) - Dln(1 + R_i^1D)

**ln C_i CANCELS EXACTLY.**  The completion term contributes NOTHING to the
channel difference; it is a common factor.  The +15.83 is therefore entirely a
difference of *catalogue-leg odds tilts*, and the "gain of completion up-pull"
in the story is not a separate additive channel.

Partition of the dark class:
  Z  = dark events with L_cat_with_bh == 0 at EVERY h (2D leg identically absent)
       -> R^2D == 0 -> contribution is exactly  -Dln(1 + R^1D_i)
          ("pure loss of the 1D catalogue leg's h-tilt")
  Z1 = dark events already 1D-dead (L_cat_no_bh == 0 everywhere) -> contribute 0
  S  = dark survivors (2D leg nonzero somewhere) -> carry both terms

Also reported: the ABSOLUTE dark 2D total (+4.06) split the same way, where the
completion term does NOT cancel, so the "up-pull" language can be checked where
it actually applies.

Read-only w.r.t. master_thesis_code/.  Run from the repo root with .venv/bin/python.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REAL = HERE.parent
R1 = REAL / "seed61000" / "real_r1"
H_LO, H_HI = 0.73, 0.81


def main() -> None:
    crb = pd.read_csv(REAL / "seed61000" / "prepared_cramer_rao_bounds.csv")
    incat = set(crb.index[crb.host_galaxy_index >= 0])

    ev = pd.read_csv(R1 / "diagnostics" / "event_likelihoods.csv")
    ev["is_incat"] = ev.event_idx.isin(incat)

    out: dict[str, object] = {}

    # ---------- 0. structural verification of the mixture identity ----------
    rec1 = ev.w_G * ev.L_cat_no_bh + (1 - ev.w_G) * ev.L_comp
    rec2 = ev.w_G * ev.L_cat_with_bh + (1 - ev.w_G) * ev.L_comp
    r1e = float(np.max(np.abs(rec1 / ev.combined_no_bh - 1)))
    r2e = float(np.max(np.abs(rec2 / ev.combined_with_bh - 1)))
    print("=== 0. mixture identity p = w_G*L_cat + (1-w_G)*L_comp ===")
    print(f"  max rel err 1D {r1e:.2e} | 2D {r2e:.2e}   (rows={len(ev)})")
    out["mixture_identity_max_relerr"] = {"1D": r1e, "2D": r2e}

    # ---------- 1. slice at the two h values ----------
    lo = ev[np.isclose(ev.h, H_LO)].set_index("event_idx").sort_index()
    hi = ev[np.isclose(ev.h, H_HI)].set_index("event_idx").sort_index()
    assert (lo.index == hi.index).all()
    idx = lo.index.to_numpy()
    is_incat = lo.is_incat.to_numpy()

    C_lo = (1 - lo.w_G) * lo.L_comp
    C_hi = (1 - hi.w_G) * hi.L_comp
    R1_lo = (lo.w_G * lo.L_cat_no_bh / C_lo).to_numpy()
    R1_hi = (hi.w_G * hi.L_cat_no_bh / C_hi).to_numpy()
    R2_lo = (lo.w_G * lo.L_cat_with_bh / C_lo).to_numpy()
    R2_hi = (hi.w_G * hi.L_cat_with_bh / C_hi).to_numpy()

    dlnC = np.log(C_hi.to_numpy() / C_lo.to_numpy())
    t1 = np.log1p(R1_hi) - np.log1p(R1_lo)  # 1D catalogue-odds tilt
    t2 = np.log1p(R2_hi) - np.log1p(R2_lo)  # 2D catalogue-odds tilt

    d1 = dlnC + t1  # = Dln p^1D
    d2 = dlnC + t2  # = Dln p^2D
    # cross-check against the delivered posteriors' per-event values
    d1_ref = np.log(hi.combined_no_bh.to_numpy() / lo.combined_no_bh.to_numpy())
    d2_ref = np.log(hi.combined_with_bh.to_numpy() / lo.combined_with_bh.to_numpy())
    print("\n=== 1. decomposition closure ===")
    print(f"  max |d1 - d1_ref| = {np.max(np.abs(d1 - d1_ref)):.2e}")
    print(f"  max |d2 - d2_ref| = {np.max(np.abs(d2 - d2_ref)):.2e}")
    out["closure_max_abs"] = {
        "1D": float(np.max(np.abs(d1 - d1_ref))),
        "2D": float(np.max(np.abs(d2 - d2_ref))),
    }

    dark = ~is_incat
    print("\n=== 2. class totals, h = 0.73 -> 0.81 (nats) ===")
    for lab, m in (("IN-CAT", is_incat), ("DARK", dark)):
        print(
            f"  {lab:7s} n={m.sum():4d} | 1D {d1[m].sum():+8.2f} | 2D {d2[m].sum():+8.2f} "
            f"| diff {d2[m].sum() - d1[m].sum():+8.2f} "
            f"|| lnC {dlnC[m].sum():+9.2f}  t1 {t1[m].sum():+9.2f}  t2 {t2[m].sum():+9.2f}"
        )
    out["class_totals"] = {
        lab: {
            "n": int(m.sum()),
            "d1": float(d1[m].sum()),
            "d2": float(d2[m].sum()),
            "diff": float(d2[m].sum() - d1[m].sum()),
            "dlnC": float(dlnC[m].sum()),
            "t1": float(t1[m].sum()),
            "t2": float(t2[m].sum()),
        }
        for lab, m in (("IN-CAT", is_incat), ("DARK", dark))
    }

    # ---------- 3. dark-class partition ----------
    g = ev.groupby("event_idx").agg(
        any1d=("L_cat_no_bh", lambda s: bool((s > 0).any())),
        any2d=("L_cat_with_bh", lambda s: bool((s > 0).any())),
    )
    g = g.reindex(idx)
    any1d = g.any1d.to_numpy()
    any2d = g.any2d.to_numpy()

    grpZ = dark & any1d & ~any2d  # 2D leg identically zero, 1D alive
    grpZ1 = dark & ~any1d  # both legs dead -> pure completion in BOTH channels
    grpS = dark & any2d  # 2D survivors

    print("\n=== 3. DARK partition of the +15.83 channel difference ===")
    print(f"  {'group':22s} {'n':>5s} {'-t1':>9s} {'t2':>9s} {'diff':>9s} {'%of diff':>9s}")
    tot_diff = float((d2[dark] - d1[dark]).sum())
    rows = []
    for lab, m in (
        ("Z  2D-dead, 1D alive", grpZ),
        ("Z1 both dead", grpZ1),
        ("S  2D survivors", grpS),
    ):
        dd = float((t2[m] - t1[m]).sum())
        print(
            f"  {lab:22s} {int(m.sum()):5d} {-t1[m].sum():+9.2f} {t2[m].sum():+9.2f} "
            f"{dd:+9.2f} {100 * dd / tot_diff:8.1f}%"
        )
        rows.append(
            {
                "group": lab,
                "n": int(m.sum()),
                "minus_t1": float(-t1[m].sum()),
                "t2": float(t2[m].sum()),
                "diff": dd,
                "pct_of_diff": 100 * dd / tot_diff,
                "dlnC": float(dlnC[m].sum()),
                "d2": float(d2[m].sum()),
                "d1": float(d1[m].sum()),
            }
        )
    print(
        f"  {'TOTAL DARK':22s} {int(dark.sum()):5d} {-t1[dark].sum():+9.2f} "
        f"{t2[dark].sum():+9.2f} {tot_diff:+9.2f}   100.0%"
    )
    out["dark_partition"] = rows
    out["dark_total_diff"] = tot_diff

    # ---------- 4. the STORY test: is any of the +15.83 a "completion up-pull"? ----------
    print("\n=== 4. STORY TEST: where does the +15.83 come from? ===")
    print(f"  completion-leg contribution to the DIFFERENCE (dlnC cancels): {0.0:+.2f} nats")
    print(f"  loss of 1D catalogue down-pull  (-Sum t1, dark)            : {-t1[dark].sum():+.2f}")
    print(f"  residual 2D catalogue tilt      (+Sum t2, dark)            : {t2[dark].sum():+.2f}")
    print(f"  -> exact sum                                              : {tot_diff:+.2f}")
    print(
        f"  of the loss term, carried by 2D-dead events (Z)            : {-t1[grpZ].sum():+.2f} "
        f"({100 * float(-t1[grpZ].sum()) / tot_diff:.1f}% of diff)"
    )
    print(
        f"  of the loss term, carried by survivors (S)                 : {-t1[grpS].sum():+.2f} "
        f"({100 * float(-t1[grpS].sum()) / tot_diff:.1f}%)"
    )
    print(
        f"  survivor residual 2D tilt (S only)                         : {t2[grpS].sum():+.2f} "
        f"({100 * float(t2[grpS].sum()) / tot_diff:.1f}%)"
    )

    # ---------- 5. absolute dark 2D total: where the "up-pull" language DOES apply ----------
    print("\n=== 5. absolute DARK 2D total (+4.06) — completion does NOT cancel here ===")
    for lab, m in (
        ("Z  2D-dead, 1D alive", grpZ),
        ("Z1 both dead", grpZ1),
        ("S  2D survivors", grpS),
        ("TOTAL DARK", dark),
    ):
        print(
            f"  {lab:22s} n={int(m.sum()):4d} | 2D total {d2[m].sum():+8.2f} "
            f"= lnC {dlnC[m].sum():+8.2f} + t2 {t2[m].sum():+8.2f}"
        )
    pure_comp = grpZ | grpZ1
    print(
        f"  pure-completion dark events (Z+Z1, n={int(pure_comp.sum())}): "
        f"2D total {d2[pure_comp].sum():+.2f} of the dark {d2[dark].sum():+.2f} "
        f"({100 * float(d2[pure_comp].sum()) / float(d2[dark].sum()):.1f}%)"
    )
    out["pure_completion"] = {
        "n": int(pure_comp.sum()),
        "d2": float(d2[pure_comp].sum()),
        "dark_d2": float(d2[dark].sum()),
    }

    # ---------- 6. is the completion leg genuinely up-tilted in THIS venue? ----------
    print("\n=== 6. is the completion leg (1-w_G)L_comp genuinely UP-tilted here? ===")
    print(f"  Sum dln[(1-w_G)L_comp]  ALL  (n={len(idx)}): {dlnC.sum():+.2f}")
    print(f"  Sum dln[(1-w_G)L_comp]  DARK (n={int(dark.sum())}): {dlnC[dark].sum():+.2f}")
    print(
        f"  Sum dln[(1-w_G)L_comp]  IN-CAT (n={int(is_incat.sum())}): {dlnC[is_incat].sum():+.2f}"
    )
    n = len(idx)
    dwg = n * np.log((1 - hi.w_G.iloc[0]) / (1 - lo.w_G.iloc[0]))
    print(f"    of which N*dln(1-w_G) = {dwg:+.2f}  and Sum dln L_comp = {dlnC.sum() - dwg:+.2f}")
    print(f"  fraction of dark events with dlnC > 0: {100 * float((dlnC[dark] > 0).mean()):.1f}%")
    out["completion_tilt"] = {
        "all": float(dlnC.sum()),
        "dark": float(dlnC[dark].sum()),
        "incat": float(dlnC[is_incat].sum()),
        "N_dln_1mwG": float(dwg),
        "sum_dlnLcomp": float(dlnC.sum() - dwg),
        "frac_dark_positive": float((dlnC[dark] > 0).mean()),
    }

    # ---------- 7. full-grid version: is 0.73->0.81 representative? ----------
    print("\n=== 7. full-grid profiles: class-summed ln p and their argmax ===")
    hs = np.sort(ev.h.unique())
    prof = {}
    for chan, col in (("1D", "combined_no_bh"), ("2D", "combined_with_bh")):
        for lab, sel in (("IN-CAT", True), ("DARK", False)):
            sub = ev[ev.is_incat == sel]
            s = sub.groupby("h")[col].apply(lambda c: float(np.log(c).sum())).reindex(hs)
            prof[f"{chan}_{lab}"] = s.to_numpy()
            print(f"  {chan} {lab:7s} argmax = {hs[int(np.argmax(s.to_numpy()))]:.3f}")
    # completion-only profile
    ev["_C"] = (1 - ev.w_G) * ev.L_comp
    for lab, sel in (("IN-CAT", True), ("DARK", False)):
        sub = ev[ev.is_incat == sel]
        s = sub.groupby("h")._C.apply(lambda c: float(np.log(c).sum())).reindex(hs)
        prof[f"C_{lab}"] = s.to_numpy()
        print(
            f"  C  {lab:7s} argmax = {hs[int(np.argmax(s.to_numpy()))]:.3f}  (completion leg alone)"
        )
    out["profiles"] = {"h": hs.tolist(), **{k: v.tolist() for k, v in prof.items()}}

    with open(HERE / "c4_decomposition_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {HERE / 'c4_decomposition_results.json'}")


if __name__ == "__main__":
    main()
