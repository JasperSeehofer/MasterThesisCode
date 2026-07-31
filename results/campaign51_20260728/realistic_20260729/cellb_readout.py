"""Cell B of the 2x2 — pre-registered readout (PREREGISTRATION_2x2_cellB.md).

Cell B = the UNSCATTERED #51 catalogue + seed61000 CRB evaluated through the
#53 estimator (absolute_marginal + volume_deconv + auto mass kernel), jobs
6103219/6103220 (resubmission of 6101146/6101147 after the missing raw-CRB
symlink was fixed; failure was setup-only, pre-registration unchanged).

Reads (pulled from the cluster 2026-07-31, `seed61000/estimatorB_2x2/`):
  - posteriors/combined_posterior.json          (1D combined)
  - posteriors_with_bh_mass/combined_posterior.json  (2D combined)
  - diagnostics/event_likelihoods.csv           (per-event, both channels, 41 h)
  - mixture_leg_log_extract.txt                 (D, beta_Gbar, Partition-norm per h)

Pre-registered comparisons (runbook §6 / ADJUDICATION §3):
  A = #51 idealized:  1D 0.7299, 2D 0.7300   (generator_marginal + point)
  C = #53 realistic:  1D 0.732 (r1 MAP 0.740), 2D 0.8133 (r1)
  B - A = estimator effect;  C - B = scatter effect.
  Joint prediction (adjudication §3 + history A'): ESTIMATOR OWNS IT — B ≈ C
  (2D ≈ 0.78-0.82, in-cat class argmax ≈ 0.86, 1D ≈ 0.70-0.74 as a crossing).
  Secondary reads: per-class C1 analog both channels; in-cat per-event 1D
  argmax 0.86-edge fraction; w_G(h) expected bit-identical to the #53 curve
  (0.1215039 at h=0.73).

Conventions reused from attack_c1_c5.py / attack_c3_c4_allruns.py. Run from
the repo root with .venv/bin/python.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
B = HERE / "seed61000" / "estimatorB_2x2"


def moments(path: Path) -> dict[str, float]:
    d = json.load(open(path))
    h = np.asarray(d["h_values"], float)
    p = np.asarray(d["posterior"], float)
    o = np.argsort(h)
    h, p = h[o], p[o]
    p = np.where(np.isfinite(p), p, 0.0)
    p /= np.trapezoid(p, h)
    mean = float(np.trapezoid(p * h, h))
    sig = float(np.sqrt(np.trapezoid(p * (h - mean) ** 2, h)))
    return {
        "map": float(d["map_h"]),
        "mean": mean,
        "sigma": sig,
        "edge": float(max(p[0], p[-1]) / p.max()),
    }


def main() -> None:
    crb = pd.read_csv(HERE / "seed61000" / "prepared_cramer_rao_bounds.csv")
    incat = set(crb.index[crb.host_galaxy_index >= 0])

    m1 = moments(B / "posteriors" / "combined_posterior.json")
    m2 = moments(B / "posteriors_with_bh_mass" / "combined_posterior.json")
    print("=== Cell B combined posteriors ===")
    print(
        f"  1D: MAP {m1['map']:.4f}  mean {m1['mean']:.4f}  sigma {m1['sigma']:.4f}  edge/peak {m1['edge']:.1e}"
    )
    print(
        f"  2D: MAP {m2['map']:.4f}  mean {m2['mean']:.4f}  sigma {m2['sigma']:.4f}  edge/peak {m2['edge']:.1e}"
    )

    ev = pd.read_csv(B / "diagnostics" / "event_likelihoods.csv")
    n_events = ev.event_idx.nunique()
    hs = np.sort(ev.h.unique())
    print(f"  diagnostics: {len(ev)} rows, {n_events} events, {len(hs)} h-values")

    # w_G bit-identity vs the #53 realistic curve
    wg = ev.groupby("h").w_G.first().sort_index()
    print("\n=== w_G check (pre-registered: bit-identical to #53) ===")
    print(f"  w_G(0.73) = {wg.loc[np.isclose(wg.index, 0.73)].iloc[0]:.7f}   [#53: 0.1215039]")
    print(f"  w_G(0.81) = {wg.loc[np.isclose(wg.index, 0.81)].iloc[0]:.7f}   [#53: 0.1038732]")
    print(f"  w_G(0.60) = {wg.loc[np.isclose(wg.index, 0.60)].iloc[0]:.7f}   [#53: 0.1625175]")

    # per-class summed ln p profiles, both channels
    print("\n=== per-class summed ln-likelihood profiles (argmax over the full grid) ===")
    piv1 = ev.pivot(index="event_idx", columns="h", values="combined_no_bh")
    piv2 = ev.pivot(index="event_idx", columns="h", values="combined_with_bh")
    for label, piv in (("1D", piv1), ("2D", piv2)):
        ok = (piv > 0).all(axis=1)
        lnp = np.log(piv[ok])
        isin = lnp.index.isin(incat)
        s_in = lnp[isin].sum(axis=0)
        s_dk = lnp[~isin].sum(axis=0)
        print(
            f"  {label}: IN-CAT argmax {s_in.idxmax():.3f}  DARK argmax {s_dk.idxmax():.3f}  "
            f"(N={int(isin.sum())}/{int((~isin).sum())}, dropped {int((~ok).sum())})"
        )

    # C1 analog: per-class Delta ln p, 0.73 -> 0.81, both channels + C3 split
    at73 = ev[np.isclose(ev.h, 0.73)].set_index("event_idx")
    at81 = ev[np.isclose(ev.h, 0.81)].set_index("event_idx")
    print("\n=== per-class nats budget 0.73 -> 0.81 (C1/C3 analog) ===")
    res = {}
    for label, col in (("1D", "combined_no_bh"), ("2D", "combined_with_bh")):
        d = np.log(at81[col] / at73[col])
        isin = d.index.isin(incat)
        res[label] = (float(d[isin].sum()), float(d[~isin].sum()))
        print(
            f"  {label}: IN-CAT {res[label][0]:+.2f}  DARK {res[label][1]:+.2f}  total {sum(res[label]):+.2f}"
        )
    di = res["2D"][0] - res["1D"][0]
    dd = res["2D"][1] - res["1D"][1]
    print(
        f"  channel diff: IN-CAT {di:+.2f}  DARK {dd:+.2f}  TOTAL {di + dd:+.2f}"
        f"   [#53 r1: +2.97 / +15.83 / +18.80]"
    )

    # in-cat per-event 1D argmax (C5 analog)
    ok = (piv1 > 0).all(axis=1)
    lnp1 = np.log(piv1[ok])
    sub = lnp1[lnp1.index.isin(incat)]
    peaks = sub.idxmax(axis=1).astype(float)
    print("\n=== in-cat per-event 1D argmax (C5 analog) ===")
    print(
        f"  N {len(peaks)}  median peak {peaks.median():.3f}  "
        f"at 0.86 edge: {(peaks >= 0.859).sum()}/{len(peaks)} = {100 * (peaks >= 0.859).mean():.1f}%"
        f"   [#53 r1: 0.860, 57.9% | idealized #51 estimator: 0.730, 5.3%]"
    )

    # dark per-event argmax
    dsub = lnp1[~lnp1.index.isin(incat)]
    dpeaks = dsub.idxmax(axis=1).astype(float)
    print(
        f"  dark median peak {dpeaks.median():.3f}, frac <= 0.66: {100 * (dpeaks <= 0.66).mean():.1f}%"
    )

    # the 2x2
    print("\n=== THE 2x2 (MAPs) ===")
    print("  A (#51, unscattered+point/genmarg):  1D 0.7299   2D 0.7300")
    print(f"  B (this run, unscattered+#53 est.):  1D {m1['map']:.4f}   2D {m2['map']:.4f}")
    print("  C (#53 r1, scattered+#53 est.):      1D 0.7400   2D 0.8133")
    print(
        f"  B - A (estimator effect):            1D {m1['map'] - 0.7299:+.4f}   2D {m2['map'] - 0.7300:+.4f}"
    )
    print(
        f"  C - B (scatter effect):              1D {0.7400 - m1['map']:+.4f}   2D {0.8133 - m2['map']:+.4f}"
    )


if __name__ == "__main__":
    main()
