"""Gate A2 of RUNBOOK_NEXT_SESSION_6.md — provenance repair for claims C3/C4.

The 2D per-event numbers behind C3/C4 were [AGENT]-provenance (measured from a
/tmp extract that evaporated). This script recomputes them from the artifacts
re-pulled from the cluster on 2026-07-30:

  seed61000/real_r1/posteriors_with_bh_mass/h_0_{725,73,735,81}.json
  seed61000/real_r1/diagnostics/event_likelihoods.csv
    (columns: event_idx,h,w_G,L_cat_no_bh,L_cat_with_bh,B_num,L_comp,
     combined_no_bh,combined_with_bh)

Targets, with the claim file's values in brackets:

  C3 — class split of the +18.80-nat channel difference, h=0.73 -> 0.81:
       IN-CAT [+2.97], DARK [+15.83]; 2D class totals [+5.45 / +4.06].
       Constraint: C2 fixes the sum at +18.80 [LOCAL], so only the partition
       is at risk.
  C4 — mechanism measurements at h=0.73:
       dark events with L_cat_with_bh == 0 exactly [64.7%] (1D analog [32.5%]);
       events with nonzero 1D catalogue term but identically-zero 2D term at
       EVERY h [488 of 1095, 487 dark];
       median L_cat_2D/L_cat_1D among nonzero survivors [7.8e-3];
       Sum ln(L_cat_2D/L_cat_1D) 0.73 -> 0.81: dark [-504.8], in-cat [+0.27].

Conventions as attack_c1_c5.py / score_realistic.py: in-cat = row index with
`host_galaxy_index >= 0` in seed61000/prepared_cramer_rao_bounds.csv.
Read-only w.r.t. master_thesis_code/. Run from the repo root with .venv/bin/python.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
R1 = HERE / "seed61000" / "real_r1"


def load_per_event(path: Path) -> dict[int, float]:
    with open(path) as f:
        j = json.load(f)
    return {int(k): (j[k][0] if isinstance(j[k], list) else j[k]) for k in j if k.isdigit()}


def main() -> None:
    crb = pd.read_csv(HERE / "seed61000" / "prepared_cramer_rao_bounds.csv")
    incat = set(crb.index[crb.host_galaxy_index >= 0])

    # --- C3: 2D class totals and channel-difference split ---
    p2 = {t: load_per_event(R1 / "posteriors_with_bh_mass" / f"h_0_{t}.json") for t in ("73", "81")}
    p1 = {t: load_per_event(R1 / "posteriors" / f"h_0_{t}.json") for t in ("73", "81")}

    d2_in = d2_dark = d1_in = d1_dark = 0.0
    for k in p2["73"]:
        d2 = float(np.log(p2["81"][k] / p2["73"][k]))
        d1 = float(np.log(p1["81"][k] / p1["73"][k]))
        if k in incat:
            d2_in += d2
            d1_in += d1
        else:
            d2_dark += d2
            d1_dark += d1

    print("=== C3: class split, h = 0.73 -> 0.81 (claim in brackets) ===")
    print(f"  2D class totals: IN-CAT {d2_in:+.2f} [+5.45], DARK {d2_dark:+.2f} [+4.06]")
    print(f"  1D class totals: IN-CAT {d1_in:+.2f} [+2.48], DARK {d1_dark:+.2f} [-11.77]")
    print(
        f"  channel diff:    IN-CAT {d2_in - d1_in:+.2f} [+2.97], "
        f"DARK {d2_dark - d1_dark:+.2f} [+15.83], "
        f"TOTAL {d2_in + d2_dark - d1_in - d1_dark:+.2f} [+18.80]"
    )
    tot = d2_in + d2_dark - d1_in - d1_dark
    print(f"  dark share of channel diff: {100 * (d2_dark - d1_dark) / tot:.1f}% [84%]")

    # --- C4: mechanism measurements from the diagnostics CSV ---
    ev = pd.read_csv(R1 / "diagnostics" / "event_likelihoods.csv")
    n_events = ev.event_idx.nunique()
    is_incat = ev.event_idx.isin(incat)
    print(f"\n=== C4: from event_likelihoods.csv ({len(ev)} rows, {n_events} events) ===")

    at73 = ev[np.isclose(ev.h, 0.73)]
    dark73 = at73[~at73.event_idx.isin(incat)]
    incat73 = at73[at73.event_idx.isin(incat)]
    print(
        f"  dark events with L_cat_with_bh == 0 at h=0.73: "
        f"{100 * (dark73.L_cat_with_bh == 0).mean():.1f}% [64.7%]  "
        f"(1D analog L_cat_no_bh == 0: {100 * (dark73.L_cat_no_bh == 0).mean():.1f}% [32.5%])"
    )

    # events with nonzero 1D catalogue term anywhere, but 2D == 0 at EVERY h
    g = ev.groupby("event_idx").agg(
        any1d=("L_cat_no_bh", lambda s: bool((s > 0).any())),
        all2d0=("L_cat_with_bh", lambda s: bool((s == 0).all())),
    )
    n_1d_nonzero = int(g.any1d.sum())
    zero2d = g[g.any1d & g.all2d0]
    n_zero2d_dark = int((~zero2d.index.isin(incat)).sum())
    print(
        f"  events with nonzero 1D cat term: {n_1d_nonzero} [1095]; "
        f"of those, 2D identically zero at every h: {len(zero2d)} [488], "
        f"dark among them: {n_zero2d_dark} [487]"
    )

    surv73 = at73[(at73.L_cat_with_bh > 0) & (at73.L_cat_no_bh > 0)]
    print(
        f"  median L_cat_2D/L_cat_1D among h=0.73 survivors: "
        f"{(surv73.L_cat_with_bh / surv73.L_cat_no_bh).median():.2e} [7.8e-3]"
    )

    # Sum ln(L_cat_2D/L_cat_1D) at 0.73 and 0.81, events nonzero in both at both h
    at81 = ev[np.isclose(ev.h, 0.81)]
    m73 = at73.set_index("event_idx")
    m81 = at81.set_index("event_idx")
    ok = (
        (m73.L_cat_with_bh > 0)
        & (m73.L_cat_no_bh > 0)
        & (m81.L_cat_with_bh > 0)
        & (m81.L_cat_no_bh > 0)
    )
    ratio73 = np.log(m73.L_cat_with_bh[ok] / m73.L_cat_no_bh[ok])
    ratio81 = np.log(m81.L_cat_with_bh[ok] / m81.L_cat_no_bh[ok])
    isin = ok.index[ok].isin(incat)
    for label, mask, target in (
        ("DARK", ~isin, "[-3165.7 -> -3670.5, D=-504.8]"),
        ("IN-CAT", isin, "[-99.4 -> -99.1, D=+0.27]"),
    ):
        s73, s81 = float(ratio73[mask].sum()), float(ratio81[mask].sum())
        print(
            f"  Sum ln(Lcat2D/Lcat1D) {label} ({int(mask.sum())} ev): "
            f"{s73:.1f} -> {s81:.1f}, D = {s81 - s73:+.2f} {target}"
        )

    # --- bonus: w_G table at full precision + B_num/L_comp tilt by class ---
    wg = ev.groupby("h").w_G.first().sort_index()
    print("\n=== w_G(h) from the diagnostics CSV (claim: 0.0697 at h=0.73) ===")
    print(f"  w_G(0.73) = {wg.loc[np.isclose(wg.index, 0.73)].iloc[0]:.7f}")
    print(f"  w_G(0.81) = {wg.loc[np.isclose(wg.index, 0.81)].iloc[0]:.7f}")
    print(f"  empirical in-cat rate: {len(incat)}/{n_events} = {len(incat) / n_events:.4f}")

    comp = ev[ev.h.isin([0.73, 0.81])].copy()
    comp["cls"] = np.where(comp.event_idx.isin(incat), "incat", "dark")
    for cls in ("incat", "dark"):
        sub = comp[comp.cls == cls]
        s = sub.groupby("h").apply(lambda d: float(np.log(d.L_comp).sum()), include_groups=False)
        print(
            f"  completion-leg tilt Sum ln L_comp {cls}: "
            f"{s.loc[0.73]:.1f} -> {s.loc[0.81]:.1f}, D = {s.loc[0.81] - s.loc[0.73]:+.2f}"
        )


if __name__ == "__main__":
    main()
