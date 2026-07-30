"""C7 — MEASURE the host-z kernel's induced h-shift (RUNBOOK_NEXT_SESSION_6 §4.3).

CLAIM C7 (CLAIM_2D_BIAS_20260730.md): the volume_deconv host-z numerator kernel
weights by w_pop = dV_c/dz/(1+z) with no p_det and no catalogue selection phi_cat,
and deconvolving a wide photo-z against a monotonically rising volume prior shifts
the effective host z UP by ~2(sigma_z/z)^2, predicting +11%..+36% h inflation ->
rails at 0.86.  Status in the claim file: "a prediction that matches, not a
measurement of the code's kernel".  This script makes the measurement.

WHAT IS COMPUTED
----------------
For each of the 76 in-catalogue hosts of seed61000 (CRB rows with
host_galaxy_index >= 0) we rebuild the code's OWN in-catalogue numerator N_g(h)
(bayesian_statistics.py:4099-4245, the batched evaluate path), for two kernels:

  volume_deconv:  N_g(h) = INT_{z_lo(h)}^{z_hi(h)} p_GW(x | z, Omega_g; h)
                            * N(z; z_obs, sigma_z) * w_pop(z; h) / Z_g(h) dz
                  w_pop(z;h) = comoving_volume_element(z,h)/(1+z)
                  Z_g(h)     = INT_{z_obs-4s}^{z_obs+4s} N * w_pop dz   (z >= 1e-6)
                  [z_lo,z_hi]= dist_to_redshift(d_L_hat -+ 4 sigma_dL, h)
                  50-pt Gauss-Legendre, exactly as the code.

  point (delta): N_g(h) = p_GW(x | z_obs, Omega_g; h)          (:4160-4170)

In `absolute_marginal` (the #53 mode) the per-event catalogue leg is
L_cat = (SUM_ball w_g N_g)/Sigma_glob(h), and Sigma_glob is EVENT-INDEPENDENT and
IDENTICAL for both kernels (generator_marginal joins the volume_deconv set for
the DENOMINATOR/Z_g machinery only, :4125-4135).  So for the single true host the
argmax_h of N_g(h) IS the argmax of that host's contribution to the catalogue leg,
and the difference between the two kernels is exactly the kernel-induced shift.

INPUTS AND PROVENANCE
---------------------
* d_L_hat, sigma_dL, phi, theta, full 3x3 CRB covariance: from
  seed61000/prepared_cramer_rao_bounds.csv exactly as Detection.__init__ and
  BayesianStatistics (:2377-2460) build them.  The pipeline never draws a noisy
  measurement (Detection.convert_to_best_guess_parameters is dead code), so
  d_L_hat == the injected true d_L and the point kernel peaks at h_true by
  construction -- which is the whole point of the comparison.
* Host sky == event sky for the true host, so the 3x3 MVN collapses EXACTLY to a
  1D Gaussian in the d_L fraction with the CONDITIONAL sigma 1/sqrt(Lambda_33)
  and an h-independent (phi,theta) prefactor -- shapes in h are unaffected.
* z_true = dist_to_redshift(d_L_hat, h=0.73)  (generator cosmology, constants.H).
* sigma_z: SWEPT over sigma_z/z in {0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.49,
  0.80} so no conclusion hinges on the catalogue column, PLUS an "indicative" leg
  from the LOCAL parent catalogue's photometric z_error(z) median relation.
  ** The local reduced_galaxy_catalogue.csv is NOT the realization parent: it
  differs in exactly the z_error column (#40b PV width).  Everything tagged
  "indicative" inherits that caveat. **
* SIGMA_V_PEC_KM_S = 0.0 in constants.py, so host_z_error_eff == sigma_z exactly
  (:3638) -- no PV term is added on top here either.

LEGS
----
  A   z_obs = z_true (no realization scatter): the PURE kernel shift.
  A'  same, at the indicative catalogue sigma_z(z).
  B   z_obs = z_true + sigma_z*N(0,1): the realized in-cat population.
  S   sigma_z -> 0 scaling gate (RUNBOOK §7): the shift must vanish ~ (sigma_z/z)^2.
      Run at HIGH quadrature order because at sigma_z below the GW window width the
      code's own 50-node GL over the WIDE event window cannot resolve the prior
      spike; the limit is a statement about the integral, not about GL-50.

Read-only w.r.t. master_thesis_code/.  Run from the repo root:
    .venv/bin/python results/campaign51_20260728/realistic_20260729/gate_b_20260730/c7_kernel_measure.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import roots_legendre

from master_thesis_code.bayesian_inference.bayesian_statistics import (
    _GL_NODES_50,
    _GL_WEIGHTS_50,
    _batched_gl_nodes,
    _batched_gl_reduce,
    _gaussian_pdf,
)
from master_thesis_code.constants import H as H_TRUE
from master_thesis_code.physical_relations import (
    comoving_volume_element,
    dist_to_redshift,
    dist_vectorized,
)

HERE = Path(__file__).parent
CRB_PATH = HERE.parent / "seed61000" / "prepared_cramer_rao_bounds.csv"
CAT_PATH = Path("master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv")
SIGMA_MULT = 4.0  # integration_limit_sigma_multiplier, :4098


# --------------------------------------------------------------------------- #
def load_incat_events() -> pd.DataFrame:
    crb = pd.read_csv(CRB_PATH)
    sel = crb[crb.host_galaxy_index >= 0].copy()
    d_L = sel["luminosity_distance"].to_numpy()
    sigma_dL = np.sqrt(sel["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
    n = len(sel)
    cov = np.zeros((n, 3, 3))
    cov[:, 0, 0] = sel["delta_phiS_delta_phiS"].to_numpy()
    cov[:, 1, 1] = sel["delta_qS_delta_qS"].to_numpy()
    cov[:, 0, 1] = cov[:, 1, 0] = sel["delta_phiS_delta_qS"].to_numpy()
    cov[:, 0, 2] = cov[:, 2, 0] = sel["delta_phiS_delta_luminosity_distance"].to_numpy() / d_L
    cov[:, 1, 2] = cov[:, 2, 1] = sel["delta_qS_delta_luminosity_distance"].to_numpy() / d_L
    cov[:, 2, 2] = sigma_dL**2 / d_L**2
    cov_inv = np.linalg.pinv(cov)
    out = pd.DataFrame(
        dict(
            row=sel.index.to_numpy(),
            d_L=d_L,
            sigma_dL=sigma_dL,
            rel_dL=sigma_dL / d_L,
            sigma_frac_cond=1.0 / np.sqrt(cov_inv[:, 2, 2]),
            snr=sel["SNR"].to_numpy(),
        )
    )
    out["z_true"] = np.array([dist_to_redshift(x, h=H_TRUE) for x in d_L])
    return out


def indicative_sigma_z_over_z(z: np.ndarray) -> np.ndarray:
    """Median photometric z_error/z at comparable z from the LOCAL parent catalogue.

    INDICATIVE ONLY -- the local z_error column is the pre-#40b PV width and is NOT
    the realization parent (sidecar parent_csv_sha256 7af3f4f4a2...).
    """
    names = ["ra", "dec", "bmag", "z", "z_err", "Mstar", "Mstar_err", "zflag"]
    cat = pd.read_csv(CAT_PATH, names=names, usecols=["z", "z_err", "zflag"])
    cat = cat[(cat.z > 0) & (cat.z_err > 0) & (cat.zflag == 1)]
    lz = np.log10(cat.z.to_numpy())
    bins = np.arange(-3.0, 0.31, 0.05)
    idx = np.digitize(lz, bins)
    ratio = cat.z_err.to_numpy() / cat.z.to_numpy()
    med = np.full(len(bins) + 1, np.nan)
    for b in range(len(bins) + 1):
        m = idx == b
        if m.sum() > 50:
            med[b] = np.median(ratio[m])
    r = med[np.digitize(np.log10(z), bins)]
    return np.where(np.isfinite(r), r, np.nanmedian(med))


# --------------------------------------------------------------------------- #
def deconv_profile(
    h_grid: np.ndarray,
    d_L_hat: np.ndarray,
    sigma_dL: np.ndarray,
    sigma_frac: np.ndarray,
    z_obs: np.ndarray,
    sigma_z: np.ndarray,
    n_nodes: int = 50,
) -> np.ndarray:
    """N_g(h) for all hosts, shape (n_host, n_h).  Code-faithful at n_nodes=50."""
    if n_nodes == 50:
        nodes, weights = _GL_NODES_50, _GL_WEIGHTS_50
    else:
        nodes, weights = roots_legendre(n_nodes)
    n = len(z_obs)
    out = np.empty((n, len(h_grid)))
    den_lo = np.maximum(z_obs - SIGMA_MULT * sigma_z, 1e-6)  # :4120-4123
    den_hi = z_obs + SIGMA_MULT * sigma_z
    y_den = _batched_gl_nodes(den_lo, den_hi, nodes)
    gauss_den = _gaussian_pdf(y_den, z_obs[:, None], sigma_z[:, None])
    dl_hi = d_L_hat + SIGMA_MULT * sigma_dL
    dl_lo = np.maximum(d_L_hat - SIGMA_MULT * sigma_dL, 1e-9)

    for j, h in enumerate(h_grid):
        w_pop_den = (
            np.asarray(comoving_volume_element(y_den.reshape(-1), h=h), float)
            / (1.0 + y_den.reshape(-1))
        ).reshape(y_den.shape)
        Zg = _batched_gl_reduce(den_lo, den_hi, weights, gauss_den * w_pop_den)
        Zg = np.where(Zg <= 0.0, 1.0, Zg)

        z_hi = np.array([dist_to_redshift(v, h=h) for v in dl_hi])
        z_lo = np.array([dist_to_redshift(v, h=h) for v in dl_lo])
        y_num = _batched_gl_nodes(z_lo, z_hi, nodes)  # (n, k)
        d_L_num = dist_vectorized(y_num.reshape(-1), h=h).reshape(y_num.shape)
        gw = np.exp(-0.5 * ((d_L_num / d_L_hat[:, None] - 1.0) / sigma_frac[:, None]) ** 2)
        w_pop_num = (
            np.asarray(comoving_volume_element(y_num.reshape(-1), h=h), float)
            / (1.0 + y_num.reshape(-1))
        ).reshape(y_num.shape)
        base = _gaussian_pdf(y_num, z_obs[:, None], sigma_z[:, None])
        out[:, j] = _batched_gl_reduce(z_lo, z_hi, weights, gw * base * w_pop_num / Zg[:, None])
    return out


def point_peak_h(z_obs: np.ndarray, d_L_hat: np.ndarray) -> np.ndarray:
    """EXACT argmax of the delta-kernel numerator: d_L(z_obs; h) == d_L_hat.

    d_L(z;h) = f(z)/h with f h-independent, so h_peak = f(z_obs)/d_L_hat
    = dist(z_obs, h=1)/d_L_hat.
    """
    return np.asarray(dist_vectorized(z_obs, h=1.0), float) / d_L_hat


def refined_peak(
    h_grid: np.ndarray, prof: np.ndarray, *, refine_args: dict | None = None
) -> np.ndarray:
    """Grid argmax + one local refinement pass + parabolic interpolation."""
    j = np.argmax(prof, axis=1)
    peak = h_grid[j].astype(float)
    if refine_args is None:
        return peak
    step = h_grid[1] - h_grid[0]
    out = peak.copy()
    for i in range(prof.shape[0]):
        lo, hi = peak[i] - 1.5 * step, peak[i] + 1.5 * step
        hh = np.linspace(max(lo, 1e-3), hi, 41)
        pr = deconv_profile(
            hh,
            refine_args["d_L_hat"][i : i + 1],
            refine_args["sigma_dL"][i : i + 1],
            refine_args["sigma_frac"][i : i + 1],
            refine_args["z_obs"][i : i + 1],
            refine_args["sigma_z"][i : i + 1],
            n_nodes=refine_args.get("n_nodes", 50),
        )[0]
        k = int(np.argmax(pr))
        if 0 < k < len(hh) - 1:
            y0, y1, y2 = pr[k - 1], pr[k], pr[k + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            out[i] = hh[k] + np.clip(d, -1, 1) * (hh[1] - hh[0])
        else:
            out[i] = hh[k]
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ev = load_incat_events()
    n = len(ev)
    d_L = ev.d_L.to_numpy()
    s_dL = ev.sigma_dL.to_numpy()
    s_fr = ev.sigma_frac_cond.to_numpy()
    z_t = ev.z_true.to_numpy()

    print(f"in-catalogue hosts: {n}")
    print(
        "z_true       "
        + "  ".join(f"p{p}={np.percentile(z_t, p):.4f}" for p in (5, 25, 50, 75, 95))
    )
    print(
        "sigma_dL/d_L " + "  ".join(f"p{p}={np.percentile(ev.rel_dL, p):.2e}" for p in (5, 50, 95))
    )
    print("cond sigma_f " + "  ".join(f"p{p}={np.percentile(s_fr, p):.2e}" for p in (5, 50, 95)))
    # GW numerator window half-width in z, relative to z  (why GL-50 is fine at
    # large sigma_z and fails at small sigma_z)
    zw = np.array([dist_to_redshift(d_L[i] + 4 * s_dL[i], h=H_TRUE) for i in range(n)]) - z_t
    print(
        "GW window half-width / z: "
        + "  ".join(f"p{p}={np.percentile(zw / z_t, p):.4f}" for p in (5, 50, 95))
    )

    eps_ind = indicative_sigma_z_over_z(z_t)
    print(
        "indicative sigma_z/z (LOCAL STALE parent, photometric): "
        + "  ".join(f"p{p}={np.percentile(eps_ind, p):.3f}" for p in (5, 25, 50, 75, 95))
    )

    h_fine = np.arange(0.30, 2.401, 0.004)
    pk_pt = point_peak_h(z_t, d_L)
    print(
        f"\npoint-kernel peak (exact): median {np.median(pk_pt):.6f}  "
        f"[min {pk_pt.min():.6f}, max {pk_pt.max():.6f}]  (h_true = {H_TRUE})"
    )

    eps_list = [0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.49, 0.80]
    rows, per_host = [], {}
    for eps in eps_list:
        sz = eps * z_t
        prof = deconv_profile(h_fine, d_L, s_dL, s_fr, z_t, sz)
        pk = refined_peak(
            h_fine,
            prof,
            refine_args=dict(d_L_hat=d_L, sigma_dL=s_dL, sigma_frac=s_fr, z_obs=z_t, sigma_z=sz),
        )
        shift = pk / pk_pt - 1.0
        rows.append(
            dict(
                eps=eps,
                median_peak_deconv=float(np.median(pk)),
                median_frac_shift=float(np.median(shift)),
                p16=float(np.percentile(shift, 16)),
                p84=float(np.percentile(shift, 84)),
                mode_formula=float((1 + np.sqrt(1 + 8 * eps**2)) / 2 - 1),
                quadratic_2eps2=float(2 * eps**2),
                frac_peak_above_086=float(np.mean(pk > 0.86)),
            )
        )
        per_host[f"eps_{eps}"] = pk.tolist()
        print(f"  eps={eps}: median peak {np.median(pk):.4f}, shift {np.median(shift):+.4f}")

    df = pd.DataFrame(rows)
    print("\n=== LEG A: pure kernel shift (z_obs = z_true), 76 in-cat hosts ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    # ---- LEG A': indicative catalogue sigma_z --------------------------------
    sz = eps_ind * z_t
    prof = deconv_profile(h_fine, d_L, s_dL, s_fr, z_t, sz)
    pk_ind = refined_peak(
        h_fine,
        prof,
        refine_args=dict(d_L_hat=d_L, sigma_dL=s_dL, sigma_frac=s_fr, z_obs=z_t, sigma_z=sz),
    )
    print("\n=== LEG A': indicative catalogue sigma_z (STALE-COLUMN CAVEAT) ===")
    print(
        f"median peak = {np.median(pk_ind):.4f}   median frac shift = "
        f"{np.median(pk_ind / pk_pt - 1):+.4f}   frac(peak > 0.86) = {np.mean(pk_ind > 0.86):.3f}"
    )
    per_host["indicative"] = pk_ind.tolist()

    # ---- LEG B: with realization scatter -------------------------------------
    rng = np.random.default_rng(20260730)
    legB = []
    NDRAW = 20
    for eps in (0.15, 0.25, 0.35, 0.49, 0.80):
        pk_all = []
        for _ in range(NDRAW):
            zo = np.clip(z_t + eps * z_t * rng.standard_normal(n), 1e-4, None)
            sz = eps * z_t  # z_error column is COPIED unchanged by the realization
            prof = deconv_profile(h_fine, d_L, s_dL, s_fr, zo, sz)
            pk_all.append(h_fine[np.argmax(prof, axis=1)])
        pk_all = np.concatenate(pk_all)
        legB.append(
            dict(
                eps=eps,
                median_peak=float(np.median(pk_all)),
                p16=float(np.percentile(pk_all, 16)),
                p84=float(np.percentile(pk_all, 84)),
                frac_above_086=float(np.mean(pk_all > 0.86)),
                frac_below_060=float(np.mean(pk_all < 0.60)),
            )
        )
        print(f"  legB eps={eps} done")
    print("\n=== LEG B: z_obs = z_true + sigma_z*N(0,1) (20 draws x 76 hosts) ===")
    print(pd.DataFrame(legB).to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    # ---- LEG S: sigma_z -> 0 scaling gate ------------------------------------
    # High quadrature order so the limit is a statement about the INTEGRAL.
    sub = np.arange(0, n, 6)  # 13 representative hosts (cost control)
    eps_s = [0.30, 0.20, 0.14, 0.10, 0.07, 0.05, 0.035, 0.025, 0.017, 0.012]
    legS = []
    for eps in eps_s:
        sz = eps * z_t[sub]
        nn = int(np.clip(60 * np.max(zw[sub] / sz) * 4, 200, 6000))
        hw = 0.35 * eps**2 + 0.02
        hg = np.linspace(max(0.60, H_TRUE * (1 - hw)), H_TRUE * (1 + 3 * hw), 260)
        prof = deconv_profile(hg, d_L[sub], s_dL[sub], s_fr[sub], z_t[sub], sz, n_nodes=nn)
        pk = refined_peak(
            hg,
            prof,
            refine_args=dict(
                d_L_hat=d_L[sub],
                sigma_dL=s_dL[sub],
                sigma_frac=s_fr[sub],
                z_obs=z_t[sub],
                sigma_z=sz,
                n_nodes=nn,
            ),
        )
        sh = np.median(pk / pk_pt[sub] - 1.0)
        legS.append(
            dict(
                eps=eps,
                n_nodes=nn,
                median_frac_shift=float(sh),
                ratio_to_2eps2=float(sh / (2 * eps**2)),
            )
        )
        print(f"  legS eps={eps} nodes={nn} shift={sh:+.5f} ratio/2eps^2={sh / (2 * eps**2):.3f}")
    dS = pd.DataFrame(legS)
    print("\n=== LEG S: sigma_z -> 0 scaling (13 hosts, high-order quadrature) ===")
    print(dS.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    lo = np.log(dS.eps.to_numpy())
    ls = np.log(np.abs(dS.median_frac_shift.to_numpy()))
    slope = np.polyfit(lo, ls, 1)[0]
    print(f"log-log slope d ln(shift)/d ln(sigma_z/z) = {slope:.3f}   (expect 2)")

    with open(HERE / "c7_kernel_measure_results.json", "w") as f:
        json.dump(
            dict(
                legA=rows,
                legA_prime=dict(median_peak=float(np.median(pk_ind))),
                legB=legB,
                legS=legS,
                loglog_slope=float(slope),
                per_host_peak=per_host,
                point_peak=pk_pt.tolist(),
                z_true=z_t.tolist(),
                eps_indicative=eps_ind.tolist(),
                rel_dL=ev.rel_dL.tolist(),
                sigma_frac_cond=s_fr.tolist(),
                gw_window_halfwidth_over_z=(zw / z_t).tolist(),
            ),
            f,
            indent=1,
        )
    print(f"\nwrote {HERE / 'c7_kernel_measure_results.json'}")


if __name__ == "__main__":
    main()
