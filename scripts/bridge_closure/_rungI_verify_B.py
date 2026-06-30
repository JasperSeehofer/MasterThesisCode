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
                       regularised_kernel=False, global_voldecount=False,
                       hierarchical_shared_latent=False):
    # The candidate flags are mutually exclusive: each defines a distinct
    # numerator/denominator pairing and they must not be combined.
    if sum([consistent_denom, regularised_kernel, global_voldecount,
            hierarchical_shared_latent]) > 1:
        raise ValueError("candidate flags are mutually exclusive")
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

    # ------------------------------------------------------------------
    # CANDIDATE (hierarchical_shared_latent): GLOBAL photo-z-SMEARED same-kernel
    # selection D_sm(h) = sum_g w_g INTEGRAL p_det^GW(d_L(z,h)) p_red(z|z_cat_g) dz
    # over the WHOLE catalogue (zc, wc), using the SAME N(z;z_cat_g,sigma_z)
    # kernel and the SAME p_bg(z) ~ dV_c(z;h=0.73)/(1+z) as the regularised
    # numerator. Mirrors precompute_global_catalog_selection (the POINT D(h)) but
    # with the photo-z smear. Per-galaxy contributions are kept for diagnostics.
    D_sm = D_sm_g = D_point_g = None
    if hierarchical_shared_latent:
        sig = max(sigma_z, 1e-4)
        z_top = float(dist_to_redshift(1.3, h=0.87))  # just beyond horizon (most-compressing h)
        # Grid must contain the FULL kernel of every CONTRIBUTING galaxy. A galaxy
        # contributes only if its kernel overlaps the p_det>0 region (z < z_top).
        # Galaxies with z_cat > z_top+5*sig have p_det~0 across their whole kernel,
        # so their contribution is exactly 0 -- excluded to avoid grid-truncation
        # bias (partial kernel -> tiny Z_g -> spurious 1/Z_g blow-up).
        zmax_grid = z_top + 10.0 * sig
        n_zgrid = int(np.clip((zmax_grid - 1e-3) / (0.25 * sig), 400, 4000))
        z_grid = np.linspace(1e-3, zmax_grid, n_zgrid)
        dz_sm = z_grid[1] - z_grid[0]
        # p_bg: identical fixed-h=0.73 shape used by the regularised numerator
        pbg_sm = np.asarray(B.comoving_volume_element(z_grid, h=0.73), float) / (1.0 + z_grid)
        # kernel matrix K[g,j] = N(z_grid[j]; z_cat_g, sig) over the FULL catalogue
        Ksm = np.exp(-0.5 * ((z_grid[None, :] - zc[:, None]) / sig) ** 2) / (
            np.sqrt(2 * np.pi) * sig
        )
        KP = Ksm * pbg_sm[None, :]            # K * p_bg                (n_gal, n_zgrid)
        Z_sm = (KP @ np.ones(n_zgrid)) * dz_sm  # Z_g = INTEGRAL K p_bg dz   (n_gal,)
        in_grid = (zc < (z_top + 5.0 * sig)) & (Z_sm > 1e-30)  # full kernel captured
        coef = np.zeros_like(wc)               # w_g / Z_g (0 for off-grid galaxies)
        coef[in_grid] = wc[in_grid] / Z_sm[in_grid]
        D_sm = {}
        D_sm_g = {}
        D_point_g = {}
        for h in hs:
            pdet_h = B._p_det_of_dl(np.asarray(dist_vectorized(z_grid, h=h), float))
            inner = (KP @ pdet_h) * dz_sm      # INTEGRAL K p_bg p_det dz  (n_gal,)
            contrib = coef * inner             # w_g INTEGRAL p_red p_det dz
            D_sm[h] = float(contrib.sum())
            D_sm_g[h] = contrib
            # per-galaxy POINT selection (same kernel collapsed to z_cat_g)
            D_point_g[h] = wc * B._p_det_of_dl(np.asarray(dist_vectorized(zc, h=h), float))

    # the candidate reuses the regularised-posterior (p_red, dV_c once) numerator
    use_reg = regularised_kernel or hierarchical_shared_latent
    if hierarchical_shared_latent:
        num_i_h = np.zeros((len(events), len(hs)))
    logpost = np.zeros(len(hs))
    for i, h in enumerate(hs):
        gd = gdenom[h]
        total = 0.0
        for j_ev, (d_meas, sdL) in enumerate(events):
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
            if use_reg:
                pbg = np.asarray(B.comoving_volume_element(zgrid, h=0.73), float) / (1.0 + zgrid)
                Z_g = nm @ (pbg * dzg)
                Z_g = np.where(Z_g > 0, Z_g, 1.0)
                nm = nm * pbg[None, :] / Z_g[:, None]
            if global_voldecount:
                gw = gw * np.interp(zgrid, g_zfine, g_of_z)
            N_g = nm @ (gw * dzg)
            num_i = float(np.sum(wg * N_g))  # Ñ_i(h) for this event
            if hierarchical_shared_latent:
                denom = D_sm[h]
                num_i_h[j_ev, i] = num_i
            elif consistent_denom:
                pdet_grid = B._p_det_of_dl(np.asarray(dist_vectorized(zgrid, h=h), float))
                D_g = nm @ (pdet_grid * dzg)
                denom = float(np.sum(wg * D_g))
            else:
                denom = gd
            L_cat = num_i / denom if denom > 0 else 0.0
            total += np.log(L_cat) if L_cat > 0 else -1e30
        logpost[i] = total
    res = B.extract_map(hs, logpost, h_true)
    res["sigma_z"] = sigma_z
    res["logpost"] = list(logpost)
    res["hs"] = hs
    if hierarchical_shared_latent:
        res.update(_hierarchical_diagnostics(hs, gdenom, D_sm, D_sm_g, D_point_g, zc, num_i_h))
    return res


def _hierarchical_diagnostics(hs, gdenom, D_sm, D_sm_g, D_point_g, zc, num_i_h):
    """CORRECTED diagnostics (verification addendum): the h-GRADIENT of D_sm/D and
    the edge-galaxy (z_cat>0.12) fractional contribution. The absolute (D_sm-D)/D
    cancels in the normalised posterior, so it is NOT used as a gate."""
    hs_arr = np.asarray(hs, float)
    D_arr = np.asarray([gdenom[h] for h in hs], float)            # POINT D(h) = gdenom
    Dsm_arr = np.asarray([D_sm[h] for h in hs], float)
    Dpt_mine = np.asarray([float(D_point_g[h].sum()) for h in hs], float)  # cross-check of D
    ratio = Dsm_arr / np.where(D_arr > 0, D_arr, np.nan)
    # (1) h-gradient of log(D_sm/D) vs a typical numerator gradient d/dh log Ñ_i
    grad_logratio = np.gradient(np.log(ratio), hs_arr)
    log_num = np.log(np.where(num_i_h > 0, num_i_h, np.nan))      # (n_events, n_h)
    grad_lognum_per_event = np.gradient(log_num, hs_arr, axis=1)  # (n_events, n_h)
    grad_lognum_typical = np.nanmedian(grad_lognum_per_event, axis=0)  # median event
    # (2) edge-galaxy (z_cat>0.12) fractional contribution to D_sm and to (D_sm-D)
    edge = zc > 0.12
    edge_frac_Dsm = {}
    edge_frac_delta = {}
    for h in hs:
        c = D_sm_g[h]
        edge_frac_Dsm[h] = float(c[edge].sum() / c.sum()) if c.sum() != 0 else float("nan")
        delta = D_sm_g[h] - D_point_g[h]
        tot = delta.sum()
        edge_frac_delta[h] = float(delta[edge].sum() / tot) if tot != 0 else float("nan")
    return {
        "D_sm": [float(x) for x in Dsm_arr],
        "D_point": [float(x) for x in D_arr],
        "D_point_mine": [float(x) for x in Dpt_mine],
        "ratio_Dsm_D": [float(x) for x in ratio],
        "grad_log_ratio": [float(x) for x in grad_logratio],
        "grad_log_num_typical": [float(x) for x in grad_lognum_typical],
        "edge_frac_Dsm": {float(h): edge_frac_Dsm[h] for h in hs},
        "edge_frac_delta": {float(h): edge_frac_delta[h] for h in hs},
        "edge_galaxy_count": int(edge.sum()),
        "n_catalogue": int(zc.size),
    }


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

    print("=== CANDIDATE (★★): regularised p_red numerator, GLOBAL photo-z-SMEARED "
          "denom D_sm(h) ===", flush=True)
    results = {}
    for sz in [0.002, 0.035]:
        r = run_closure_photoz(h_true, sz, seed=1, hierarchical_shared_latent=True)
        results[sz] = r
        post = np.exp(np.array(r["logpost"]) - np.max(r["logpost"]))
        argmax = int(np.argmax(post))
        peak = "EDGE" if argmax in (0, len(post) - 1) else "interior"
        tag = "GATE" if sz == 0.002 else "DE-RAIL"
        print(f"  [{tag}] sigma_z={sz:.3f}: MAP={r['h_refined']:.4f} bias={r['bias']:+.4f} "
              f"railed={r['railed']} peak@{peak}", flush=True)

    # Gate verdict (sigma_z=0.002 must match standard ~0.7438 within ~0.01)
    g = results[0.002]
    gate_pass = (not g["railed"]) and abs(g["h_refined"] - 0.7438) <= 0.012
    print(f"\n  GATE (sigma_z=0.002): MAP={g['h_refined']:.4f} vs standard 0.7438 -> "
          f"{'PASS' if gate_pass else 'FAIL'}", flush=True)

    # CORRECTED diagnostics for the de-rail run (sigma_z=0.035)
    d = results[0.035]
    hs = d["hs"]
    print("\n  --- CORRECTED DIAGNOSTICS (sigma_z=0.035) ---", flush=True)
    print("  (1) D_sm(h), D(h)=gdenom, ratio, and h-gradients across the grid:", flush=True)
    print(f"      {'h':>6} {'D_sm':>11} {'D_point':>11} {'Dsm/D':>8} "
          f"{'dlog(Dsm/D)/dh':>15} {'dlogN~/dh':>11}", flush=True)
    for k, h in enumerate(hs):
        if k % 3 == 0 or h in (0.60, 0.73, 0.87):
            print(f"      {h:6.2f} {d['D_sm'][k]:11.4e} {d['D_point'][k]:11.4e} "
                  f"{d['ratio_Dsm_D'][k]:8.4f} {d['grad_log_ratio'][k]:15.3f} "
                  f"{d['grad_log_num_typical'][k]:11.3f}", flush=True)
    # summary of the gradient battle near the truth
    i73 = hs.index(0.73)
    print(f"\n      @h=0.73: d/dh log(D_sm/D) = {d['grad_log_ratio'][i73]:+.3f} ; "
          f"typical d/dh log N~_i = {d['grad_log_num_typical'][i73]:+.3f}", flush=True)
    print(f"      grid-mean |d/dh log(D_sm/D)| = {np.nanmean(np.abs(d['grad_log_ratio'])):.3f} ; "
          f"grid-mean |d/dh log N~| = {np.nanmean(np.abs(d['grad_log_num_typical'])):.3f}", flush=True)
    print(f"      D_point cross-check (mine vs gdenom) @h=0.73: "
          f"{d['D_point_mine'][i73]:.4e} vs {d['D_point'][i73]:.4e}", flush=True)

    print(f"\n  (2) edge-galaxy (z_cat>0.12) fractional contribution "
          f"[{d['edge_galaxy_count']}/{d['n_catalogue']} galaxies]:", flush=True)
    print(f"      {'h':>6} {'edge-frac D_sm':>15} {'edge-frac (D_sm-D)':>20}", flush=True)
    for h in (0.60, 0.73, 0.87):
        print(f"      {h:6.2f} {d['edge_frac_Dsm'][h]:15.4f} {d['edge_frac_delta'][h]:20.4f}",
              flush=True)

    print("\n  (3) MAP / peak location:", flush=True)
    post = np.exp(np.array(d["logpost"]) - np.max(d["logpost"]))
    argmax = int(np.argmax(post))
    peak = "EDGE-RAIL" if argmax in (0, len(post) - 1) else "INTERIOR"
    print(f"      MAP={d['h_refined']:.4f} (grid {hs[argmax]:.2f}) bias={d['bias']:+.4f} "
          f"peak@{peak}", flush=True)


if __name__ == "__main__":
    main()
