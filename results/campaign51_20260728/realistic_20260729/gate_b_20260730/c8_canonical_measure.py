#!/usr/bin/env python
r"""C8 part (3) — is there a CANONICAL mass measure, or is the C-dependence arbitrary?

INDICATIVE ESTIMATE, NOT a ratified physics change.  Read-only: nothing in
master_thesis_code/ is modified; the population mass function is only *called*.

Structural point (see README_C8.md):
    p_i(h) = [ beta_G*L_cat_2D(h)  +  B_num(h) ] / D(h)
             \_______4D density_______/   \___3D density___/

The invariance-restoring degree of freedom is the MISSING mass-data likelihood
in the completion leg, not the dimensionality of D(h) (D, beta_G, beta_Gbar are
dimensionless-in-mass selection integrals in BOTH channels).  For a dark host at
redshift z the missing factor is

    g_i(z) = p(M_z,obs,i | dark host at z)
           = INTEGRAL phi(M) N(M_z,obs,i; M(1+z), sigma_Mz,i) dM
           ~= phi( M_z,obs,i / (1+z) ) / (1+z)         [sigma_Mz/M_z ~ 1e-4]

with phi(M) the population's source-frame MBH mass prior,
phi(M) ∝ mbh_mass_function(M) * R_eff_per_mbh(M) / (M ln10), normalised over
[M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX].  Expressed in the code's own
mass-fraction coordinate x = M_z/M_z,det,i the factor is g_i * M_z,det,i.

Approximation: B_num's 4-sigma d_L window is narrow, and g varies slowly with z
over it, so B_num_corrected(h) ~= B_num(h) * g_i(z_i(h)) with
z_i(h) = dist_to_redshift(d_L,i; h).  This is a SCALE estimate, good to tens of
percent, deliberately not a re-quadrature.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.integrate import quad

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")

from master_thesis_code.constants import M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN  # noqa: E402
from master_thesis_code.emri_rate import R_eff_per_mbh, mbh_mass_function  # noqa: E402
from master_thesis_code.physical_relations import dist_to_redshift  # noqa: E402

REAL = "/home/jasper/Repositories/MasterThesisCode/results/campaign51_20260728/realistic_20260729"
OUT = os.path.join(REAL, "gate_b_20260730")


def phi_M(M: np.ndarray) -> np.ndarray:
    """Normalised source-frame MBH mass prior, density per solar mass."""
    return np.asarray(mbh_mass_function(M) * R_eff_per_mbh(M) / (M * np.log(10.0)))


def main() -> None:
    norm = quad(
        lambda m: float(phi_M(np.array([m]))[0]), M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX, limit=400
    )[0]
    print(f"phi normalisation over [{M_SOURCE_FRAME_MIN:g}, {M_SOURCE_FRAME_MAX:g}] = {norm:.6e}")

    df = pd.read_csv(
        os.path.join(REAL, "seed61000", "real_r1", "diagnostics", "event_likelihoods.csv")
    )
    hs = np.sort(df["h"].unique())
    evs = np.sort(df["event_idx"].unique())
    piv = {
        c: df.pivot(index="event_idx", columns="h", values=c).loc[evs, hs].to_numpy()
        for c in ("w_G", "L_cat_with_bh", "L_comp", "L_cat_no_bh")
    }
    crb = pd.read_csv(os.path.join(REAL, "seed61000", "prepared_cramer_rao_bounds.csv"))
    M_z = crb["M"].to_numpy()[evs]
    d_L = crb["luminosity_distance"].to_numpy()[evs]

    # z_i(h) at each event's measured d_L
    z_ih = np.empty((len(evs), len(hs)))
    for j, h in enumerate(hs):
        z_ih[:, j] = np.array([dist_to_redshift(float(d), h=float(h)) for d in d_L])

    Msrc = M_z[:, None] / (1.0 + z_ih)
    inside = (Msrc >= M_SOURCE_FRAME_MIN) & (Msrc <= M_SOURCE_FRAME_MAX)
    g = np.where(inside, phi_M(np.clip(Msrc, M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX)) / norm, 0.0)
    g = g / (1.0 + z_ih)  # density in M_z
    g_frac = g * M_z[:, None]  # density in x = M_z / M_z,det,i
    print(
        f"events outside the mass prior support at some h: "
        f"{int((~inside).any(axis=1).sum())} / {len(evs)}"
    )
    print(
        f"g_frac (completion-leg mass factor, fraction units): "
        f"median {np.median(g_frac):.4f}, "
        f"10-90% [{np.percentile(g_frac, 10):.4f}, {np.percentile(g_frac, 90):.4f}]"
    )
    print("  -> the code implicitly uses g_frac == 1.0 for every event and every h.")

    w, lc2, lcomp = piv["w_G"], piv["L_cat_with_bh"], piv["L_comp"]

    def mapof(logp):
        k = int(np.argmax(logp))
        if 0 < k < len(hs) - 1:
            y0, y1, y2 = logp[k - 1], logp[k], logp[k + 1]
            d = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / d if d else 0.0
            return float(hs[k]), float(hs[k] + off * (hs[k + 1] - hs[k]))
        return float(hs[k]), float(hs[k])

    base = np.log(w * lc2 + (1 - w) * lcomp).sum(axis=0)
    corr_p = w * lc2 + (1 - w) * lcomp * g_frac
    n_zero = int((corr_p <= 0).sum())
    corr = np.log(np.where(corr_p > 0, corr_p, np.finfo(float).tiny)).sum(axis=0)
    base1 = np.log(w * piv["L_cat_no_bh"] + (1 - w) * lcomp).sum(axis=0)

    print(
        f"\n2D MAP as delivered (g_frac=1)          : {mapof(base)[0]:.4f} "
        f"(parabola {mapof(base)[1]:.5f})"
    )
    print(
        f"2D MAP with the estimated g_frac(h)     : {mapof(corr)[0]:.4f} "
        f"(parabola {mapof(corr)[1]:.5f})   [{n_zero} floored cells]"
    )
    print(
        f"1D MAP (unchanged, mass-free)           : {mapof(base1)[0]:.4f} "
        f"(parabola {mapof(base1)[1]:.5f})"
    )
    # Decomposition: freeze g at h=0.73 (a PURE measure change, h-independent)
    # vs the full g(h) (measure change + the dark population's mass-redshift
    # information, which is genuine new h-dependence, not a measure artifact).
    i73 = int(np.argmin(np.abs(hs - 0.73)))
    i81 = int(np.argmin(np.abs(hs - 0.81)))
    g_frozen = g_frac[:, i73][:, None] * np.ones_like(g_frac)
    froz = np.log(w * lc2 + (1 - w) * lcomp * g_frozen).sum(axis=0)
    print(
        f"\n  decomposition: h-frozen g(0.73) (pure measure) : {mapof(froz)[0]:.4f} "
        f"(parabola {mapof(froz)[1]:.5f})"
    )
    tilt = float((np.log(g_frac[:, i81]) - np.log(g_frac[:, i73])).sum())
    print(f"                 g(h) h-tilt Sum_i ln g_i, 0.73->0.81 = {tilt:+.2f} nats")

    print(
        "\nequivalent constant-C reading: g_frac ~ median "
        f"{np.median(g_frac):.4f}  <=>  C ~ {np.median(g_frac):.4f} "
        "in c8_reparam.py's sweep"
    )

    with open(os.path.join(OUT, "c8_canonical_measure_results.json"), "w") as fh:
        json.dump(
            {
                "g_frac_median": float(np.median(g_frac)),
                "g_frac_p10": float(np.percentile(g_frac, 10)),
                "g_frac_p90": float(np.percentile(g_frac, 90)),
                "map2d_delivered": mapof(base),
                "map2d_with_g": mapof(corr),
                "map1d": mapof(base1),
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
