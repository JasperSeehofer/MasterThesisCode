"""Adversarial verification: Angle B GLOBAL comoving-volume de-counting g(z)=p_bg/(S p_bg).

Replicates run_closure_photoz with the four candidates side by side:
  STANDARD          : bare Gaussian numerator kernel, GLOBAL Option-A denom (baseline).
  CONSISTENT-DENOM  : disqualified local convolved denom control.
  REG-KERNEL (A/C)  : per-galaxy posterior k_g = N*p_bg/Z_g, global denom kept (disqualified).
  GLOBAL-VOLDECOUNT : Angle B, numerator integrand x g(z)=p_bg/(S p_bg), global denom kept.

All four keep the GLOBAL denom except CONSISTENT-DENOM. Only the numerator changes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

logging.disable(logging.WARNING)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _bridge_lib as B  # noqa: E402
from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.physical_relations import (  # noqa: E402
    dist,
    dist_to_redshift,
    dist_vectorized,
)


def run_closure_photoz(h_true, sigma_z, *, n_gal=12000, n_events=250,
                       sigma_dL_frac=0.05, seed=0, consistent_denom=False,
                       regularised_kernel=False, global_voldecount=False):
    rng = np.random.default_rng(seed)
    hs = [float(h) for h in np.round(np.arange(0.60, 0.8701, 0.01), 4)]
    z_true_g, M = B.sample_population(rng, n_gal, h_true)
    z_cat = np.clip(z_true_g + rng.normal(0.0, sigma_z, n_gal), 1e-3, None)
    w_true = np.asarray(R_eff_per_mbh(M), float) / (1.0 + z_true_g)

    p = w_true / w_true.sum()
    events = []
    tries = 0
    while len(events) < n_events and tries < 400 * n_events:
        tries += 1
        g = int(rng.choice(n_gal, p=p))
        d_true = float(dist(z_true_g[g], h=h_true))
        sdL = sigma_dL_frac * d_true
        d_meas = d_true + sdL * rng.standard_normal()
        if d_meas <= 0:
            continue
        if rng.uniform() < float(B._p_det_of_dl(np.asarray([d_meas]))[0]):
            events.append((d_meas, sdL))

    order = np.argsort(z_cat)
    zc = z_cat[order]
    wc = np.asarray(R_eff_per_mbh(M[order]), float) / (1.0 + zc)
    pdet = B.MockPdet()
    catalog = B._ClosureCatalog(zc, M[order])
    D_tab = B.precompute_completion_denominator(hs, pdet, Omega_m=B._OMEGA_M, Omega_DE=B._OMEGA_DE)  # noqa: F841
    gdenom = B.precompute_global_catalog_selection(hs, catalog, pdet, with_bh_mass=False)

    # Angle B: precompute the GLOBAL, h-independent de-counting factor g(z)=p_bg/(S p_bg).
    g_zfine = g_of_z = None
    if global_voldecount:
        g_zfine = np.linspace(1e-3, 0.5, 4000)
        pbg_f = np.asarray(B.comoving_volume_element(g_zfine, h=h_true), float) / (1.0 + g_zfine)
        sig = max(sigma_z, 1e-4)
        Kk = np.exp(-0.5 * ((g_zfine[None, :] - g_zfine[:, None]) / sig) ** 2) / (
            np.sqrt(2 * np.pi) * sig
        )
        dzf = g_zfine[1] - g_zfine[0]
        pbg_smooth = (Kk @ pbg_f) * dzf  # (S p_bg)(z)
        g_of_z = pbg_f / np.where(pbg_smooth > 0, pbg_smooth, 1e-300)

    logpost = np.zeros(len(hs))
    for i, h in enumerate(hs):
        gd = gdenom[h]
        total = 0.0
        for d_meas, sdL in events:
            zlo = max(dist_to_redshift(max(d_meas - 5 * sdL, 1e-4), h=0.60) - 4 * sigma_z, 1e-5)
            zhi = dist_to_redshift(d_meas + 5 * sdL, h=0.87) + 4 * sigma_z
            i0 = int(np.searchsorted(zc, zlo)); i1 = int(np.searchsorted(zc, zhi))
            zg = zc[i0:i1]; wg = wc[i0:i1]
            if zg.size == 0:
                total += -1e30
                continue
            ngrid = int(np.clip((zhi - zlo) / (0.4 * max(sigma_z, 2e-3)), 120, 500))
            zgrid = np.linspace(zlo, zhi, ngrid)
            dzg = zgrid[1] - zgrid[0]
            gw = norm.pdf(np.asarray(dist_vectorized(zgrid, h=h), float), loc=d_meas, scale=sdL)
            nm = np.exp(-0.5 * ((zgrid[None, :] - zg[:, None]) / max(sigma_z, 1e-4)) ** 2) / (
                np.sqrt(2 * np.pi) * max(sigma_z, 1e-4)
            )
            if regularised_kernel:
                pbg = np.asarray(B.comoving_volume_element(zgrid, h=0.73), float) / (1.0 + zgrid)
                Z_g = nm @ (pbg * dzg)
                Z_g = np.where(Z_g > 0, Z_g, 1.0)
                nm = nm * pbg[None, :] / Z_g[:, None]
            if global_voldecount:
                gw = gw * np.interp(zgrid, g_zfine, g_of_z)
            N_g = nm @ (gw * dzg)
            if consistent_denom:
                pdet_grid = B._p_det_of_dl(np.asarray(dist_vectorized(zgrid, h=h), float))
                D_g = nm @ (pdet_grid * dzg)
                denom = float(np.sum(wg * D_g))
            else:
                denom = gd
            L_cat = float(np.sum(wg * N_g)) / denom if denom > 0 else 0.0
            total += np.log(L_cat) if L_cat > 0 else -1e30
        logpost[i] = total
    res = B.extract_map(hs, logpost, h_true)
    res["sigma_z"] = sigma_z
    res["logpost"] = list(logpost)
    res["hs"] = hs
    return res


def main():
    h_true = 0.73
    print("=== STANDARD (bare kernel, global denom) ===", flush=True)
    for sz in [0.002, 0.035]:
        r = run_closure_photoz(h_true, sz, seed=1)
        print(f"  sigma_z={sz:.3f}: MAP={r['h_refined']:.4f} bias={r['bias']:+.4f} railed={r['railed']}", flush=True)
    print("=== REG-KERNEL A/C (per-galaxy posterior, global denom) [disqualified control] ===", flush=True)
    for sz in [0.002, 0.035]:
        r = run_closure_photoz(h_true, sz, seed=1, regularised_kernel=True)
        print(f"  sigma_z={sz:.3f}: MAP={r['h_refined']:.4f} bias={r['bias']:+.4f} railed={r['railed']}", flush=True)
    print("=== ANGLE B: GLOBAL de-counting g(z)=p_bg/(S p_bg), global denom ===", flush=True)
    for sz in [0.002, 0.035]:
        r = run_closure_photoz(h_true, sz, seed=1, global_voldecount=True)
        # show posterior shape: is it interior-peaked or rail-pinned?
        post = np.exp(np.array(r["logpost"]) - np.max(r["logpost"]))
        argmax = int(np.argmax(post))
        edge = "EDGE" if argmax in (0, len(post) - 1) else "interior"
        print(f"  sigma_z={sz:.3f}: MAP={r['h_refined']:.4f} bias={r['bias']:+.4f} "
              f"railed={r['railed']} peak@{edge}", flush=True)


if __name__ == "__main__":
    main()
