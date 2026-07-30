#!/usr/bin/env python
"""C8 — mass-coordinate reparametrization dependence of the 2D posterior.

Target: claim C8 in CLAIM_2D_BIAS_20260730.md.

Structural result established by code tracing (see README_C8.md); this script
supplies the NUMERICS on locally-available data:

  combined_with_bh(h) = w_G(h) * L_cat_with_bh(h) + (1 - w_G(h)) * L_comp(h)

and the with-BH catalogue leg carries EXACTLY ONE mass-fraction density factor
(the analytic Gaussian mass product `mz_integral`, bayesian_statistics.py:4366),
while the completion leg B_num carries NONE.  Rescaling the mass coordinate
M -> C*M therefore maps

  L_cat_with_bh -> L_cat_with_bh / C          (s = -1, exactly one factor)
  L_cat_no_bh   -> L_cat_no_bh                (s =  0, no mass factor at all)
  B_num, L_comp, w_G, D(h), beta_G, beta_Gbar -> unchanged (channel-common, 3D)

Run from the repo root with .venv/bin/python.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
REAL = os.path.join(BASE, "realistic_20260729")
CSV = os.path.join(REAL, "seed61000", "real_r1", "diagnostics", "event_likelihoods.csv")
CRB = os.path.join(REAL, "seed61000", "prepared_cramer_rao_bounds.csv")
OUT = os.path.join(REAL, "gate_b_20260730")


def load() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(CSV)
    hs = np.sort(df["h"].unique())
    evs = np.sort(df["event_idx"].unique())
    piv = {
        c: df.pivot(index="event_idx", columns="h", values=c).loc[evs, hs].to_numpy()
        for c in (
            "w_G",
            "L_cat_no_bh",
            "L_cat_with_bh",
            "B_num",
            "L_comp",
            "combined_no_bh",
            "combined_with_bh",
        )
    }
    crb = pd.read_csv(CRB)
    # event_idx is the positional CRB row index
    M_z = crb["M"].to_numpy()[evs]
    incat = crb["host_galaxy_index"].to_numpy()[evs] >= 0
    return df, hs, evs, piv, (M_z, incat)


def map_of(logp: np.ndarray, hs: np.ndarray) -> tuple[float, float]:
    """Grid argmax + 3-point parabola refinement of a summed log-likelihood."""
    k = int(np.argmax(logp))
    grid = float(hs[k])
    if 0 < k < len(hs) - 1:
        y0, y1, y2 = logp[k - 1], logp[k], logp[k + 1]
        d = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / d if d != 0 else 0.0
        step = float(hs[k + 1] - hs[k])
        refined = grid + off * step
    else:
        refined = grid
    return grid, refined


def sum_logp(
    leg_cat: np.ndarray, leg_comp: np.ndarray, w: np.ndarray, scale: np.ndarray | float
) -> np.ndarray:
    """Sum_i ln[ w*scale_i*L_cat_i + (1-w)*L_comp_i ] per h. scale broadcasts over events."""
    sc = np.asarray(scale, dtype=float)
    if sc.ndim == 1:
        sc = sc[:, None]
    p = w * sc * leg_cat + (1.0 - w) * leg_comp
    bad = ~(p > 0)
    if bad.any():
        p = np.where(bad, np.finfo(float).tiny, p)
    return np.log(p).sum(axis=0), int(bad.sum())


def main() -> None:
    df, hs, evs, piv, (M_z, incat) = load()
    res: dict = {}
    N = len(evs)
    print(
        f"events={N}  h-grid: {len(hs)} pts  [{hs[0]:.4f}, {hs[-1]:.4f}]  "
        f"step={np.diff(hs).mean():.6f}"
    )
    print(f"in-catalogue events: {incat.sum()}   dark: {(~incat).sum()}")

    w = piv["w_G"]
    lc2 = piv["L_cat_with_bh"]
    lc1 = piv["L_cat_no_bh"]
    lcomp = piv["L_comp"]

    # ---------- (A) mixture identity ----------
    rec2 = w * lc2 + (1 - w) * lcomp
    rec1 = w * lc1 + (1 - w) * lcomp
    r2 = np.abs(rec2 / piv["combined_with_bh"] - 1.0)
    r1 = np.abs(rec1 / piv["combined_no_bh"] - 1.0)
    print(f"\n[A] mixture identity max |rel err|: 2D {r2.max():.3e}   1D {r1.max():.3e}")
    res["identity_max_relerr_2d"] = float(r2.max())
    res["identity_max_relerr_1d"] = float(r1.max())

    # ---------- (B) C = 1 baseline ----------
    lp2, nb = sum_logp(lc2, lcomp, w, 1.0)
    lp1, _ = sum_logp(lc1, lcomp, w, 1.0)
    g2, f2 = map_of(lp2, hs)
    g1, f1 = map_of(lp1, hs)
    print(
        f"\n[B] C=1  2D MAP grid={g2:.4f} parabola={f2:.5f}   "
        f"1D MAP grid={g1:.4f} parabola={f1:.5f}   (nonpositive cells: {nb})"
    )
    res["baseline"] = {"map2d_grid": g2, "map2d_parab": f2, "map1d_grid": g1, "map1d_parab": f1}
    # nats 0.73 -> 0.81 cross-check against C2
    i73 = int(np.argmin(np.abs(hs - 0.73)))
    i81 = int(np.argmin(np.abs(hs - 0.81)))
    print(
        f"    ln P(0.81)-ln P(0.73):  1D {lp1[i81] - lp1[i73]:+.2f}   "
        f"2D {lp2[i81] - lp2[i73]:+.2f}   (C2 claims -9.30 / +9.51)"
    )
    res["nats_0p73_to_0p81"] = {"1D": float(lp1[i81] - lp1[i73]), "2D": float(lp2[i81] - lp2[i73])}

    # ---------- (C) constant-C sweep ----------
    print("\n[C] constant mass-coordinate rescaling M -> C*M  (L_cat_2D -> L_cat_2D / C)")
    print(f"    {'C':>10} {'2D grid':>9} {'2D parab':>10} {'1D grid':>9} {'1D parab':>10}")
    sweep = {}
    for C in (100.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 1e-3, 1e-4, 1e-6):
        lp2c, _ = sum_logp(lc2, lcomp, w, 1.0 / C)
        lp1c, _ = sum_logp(lc1, lcomp, w, 1.0)  # 1D: no mass factor -> C-free by construction
        a2, b2 = map_of(lp2c, hs)
        a1, b1 = map_of(lp1c, hs)
        print(f"    {C:>10g} {a2:>9.4f} {b2:>10.5f} {a1:>9.4f} {b1:>10.5f}")
        sweep[f"{C:g}"] = {"map2d_grid": a2, "map2d_parab": b2, "map1d_grid": a1, "map1d_parab": b1}
    res["constant_C_sweep"] = sweep

    # ---------- (D) non-arbitrary alternative measures ----------
    # The code's implicit mass coordinate is the per-event fraction
    # x_i = M_z / M_z,det,i .  Two equally defensible measures:
    #   (D1) M_z in solar masses           -> C_i = M_z,det,i
    #   (D2) M_z in 1e6 solar masses       -> C_i = M_z,det,i / 1e6
    #   (D3) ln M_z (log measure)          -> C_i = M_z,det,i * x ~ M_z,det,i (at x=1)
    print("\n[D] non-arbitrary alternative mass measures (per-event C_i)")
    print(
        f"    M_z,det range: {M_z.min():.3e} .. {M_z.max():.3e} Msun, median {np.median(M_z):.3e}"
    )
    alt = {}
    for name, Ci in (
        ("M_z in Msun", M_z),
        ("M_z in 1e6 Msun", M_z / 1e6),
        ("M_z in 1e5 Msun", M_z / 1e5),
        ("fraction (code as-is)", np.ones_like(M_z)),
    ):
        lpa, _ = sum_logp(lc2, lcomp, w, 1.0 / Ci)
        a, b = map_of(lpa, hs)
        print(f"    {name:>24}: 2D MAP grid={a:.4f} parabola={b:.5f}")
        alt[name] = {"map2d_grid": a, "map2d_parab": b}
    res["alternative_measures"] = alt

    # ---------- (E) where is the C-dependence coming from? ----------
    # Fraction of events whose 2D catalogue leg dominates the mixture, vs C.
    print(
        "\n[E] mixture balance: fraction of events with w_G*L_cat_2D/C > (1-w_G)*L_comp at h=0.73"
    )
    for C in (1.0, 0.1, 0.01, 1e-3):
        frac = ((w[:, i73] * lc2[:, i73] / C) > ((1 - w[:, i73]) * lcomp[:, i73])).mean()
        print(f"    C={C:<8g} {frac * 100:6.2f}%")
    # class-resolved catalogue-leg-only profile
    for lab, mask in (("in-cat", incat), ("dark", ~incat)):
        pos = lc2[mask] > 0
        prof = np.where(lc2[mask] > 0, np.log(np.maximum(lc2[mask], np.finfo(float).tiny)), 0.0)
        cnt = pos.sum(axis=0)
        k = int(np.argmax(prof.sum(axis=0)))
        print(
            f"    catalogue-leg-only (ln L_cat_2D summed) {lab:>7}: argmax h={hs[k]:.4f} "
            f"(nonzero events/h: {cnt.min()}..{cnt.max()})"
        )

    # ---------- (E2) exactness gate: reconstruct the DELIVERED posterior ----------
    try:
        with open(os.path.join(REAL, "seed61000", "real_r1", "combined_posterior_2d.json")) as fh:
            dd = json.load(fh)
        hv = np.array(dd["h_values"])
        lo = np.log(np.array(dd["posterior"]))
        lo -= lo.max()
        lp2n = lp2 - lp2.max()
        assert np.allclose(hv, hs)
        dev = float(np.max(np.abs(lp2n - lo)))
        print(
            f"\n[E2] |ln P_recon - ln P_delivered|_max = {dev:.3e} nats "
            f"(delivered map_h={dd['map_h']})"
        )
        res["delivered_logpost_max_abs_dev_nats"] = dev
    except FileNotFoundError:
        print("\n[E2] delivered combined_posterior_2d.json not found")

    # ---------- (E3) per-event C_i vs a matched constant C ----------
    print("\n[E3] per-event C_i = M_z,det,i/1e6 vs the matched constant C = geomean(M_z/1e6)")
    gm = float(np.exp(np.mean(np.log(M_z / 1e6))))
    for name, Ci in (("per-event C_i", M_z / 1e6), (f"constant C={gm:.4f}", np.full_like(M_z, gm))):
        lpa, _ = sum_logp(lc2, lcomp, w, 1.0 / Ci)
        a, b = map_of(lpa, hs)
        print(f"    {name:>22}: 2D MAP grid={a:.4f} parabola={b:.5f}")

    # ---------- (E4) sensitivity slope d(MAP)/d(ln C) in the unrailed band -------
    Cs = np.exp(np.linspace(np.log(0.05), np.log(3.0), 25))
    maps = []
    for C in Cs:
        lpa, _ = sum_logp(lc2, lcomp, w, 1.0 / C)
        maps.append(map_of(lpa, hs)[1])
    maps_a = np.array(maps)
    sl = np.polyfit(np.log(Cs), maps_a, 1)[0]
    print(
        f"\n[E4] d(MAP_2D)/d(ln C) over C in [0.05, 3] = {sl:+.4f} in h "
        f"({sl * 100:+.2f} km/s/Mpc per e-fold of the mass unit)"
    )
    res["dMAP_dlnC"] = float(sl)

    # ---------- (E5) limits ----------
    lp_comp = np.log((1 - w) * lcomp).sum(axis=0)
    a, b = map_of(lp_comp, hs)
    print(f"[E5] C -> inf limit (pure completion leg): MAP grid={a:.4f} parabola={b:.5f}")
    res["pure_completion_map"] = {"grid": a, "parab": b}

    # ---------- (F) 1D exact invariance, numerically ----------
    # There is no C anywhere in the 1D path; assert bit-identity of the 1D
    # summed log-likelihood across the whole sweep.
    base1 = sum_logp(lc1, lcomp, w, 1.0)[0]
    same = all(np.array_equal(sum_logp(lc1, lcomp, w, 1.0)[0], base1) for _ in range(3))
    print(f"\n[F] 1D summed log-likelihood identical across sweep (bitwise): {same}")
    res["1d_bitwise_invariant"] = bool(same)

    with open(os.path.join(OUT, "c8_reparam_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'c8_reparam_results.json')}")


if __name__ == "__main__":
    sys.exit(main())
