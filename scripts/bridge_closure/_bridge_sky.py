"""Sky + 3-D MVN extensions for the bridge (Rung C).

Rungs A/B (no sky, 1-D d_L likelihood) do NOT rail. The remaining real-pipeline
ingredient is the SKY dimension: the in-catalogue numerator is a 3-D MVN in
(phi, theta, d_L/d_meas) with the FULL Fisher covariance, summed over candidates
selected in a sky-Fisher cone (real ``get_possible_hosts_from_ball_tree``). This
module reproduces that channel in the controlled harness and supports ablations:

  C-real : real GLADE sky + real events + 3-D MVN (full cov)   -> reproduce rail?
  C-iso  : real catalogue with sky positions SHUFFLED          -> clustering?
  C-1d   : sky-cone candidate selection but 1-D d_L likelihood -> MVN coupling?
  C-diag : 3-D MVN with DIAGONAL covariance                    -> d_L-sky corr?

The selection normalisation (global_denom, D(h), beta_Gbar) is kept on the SAME
mock p_det as rungs A/B, so any railing here is attributable purely to the
in-catalogue NUMERATOR (sky + MVN), not the selection h-scaling.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _bridge_lib as B  # noqa: E402
from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.physical_relations import comoving_volume_element, dist_vectorized  # noqa: E402

# CRB covariance column names (lower-triangle Fisher inverse)
_C = {
    "phi2": "delta_phiS_delta_phiS",
    "the2": "delta_qS_delta_qS",
    "dL2": "delta_luminosity_distance_delta_luminosity_distance",
    "phi_the": "delta_phiS_delta_qS",
    "phi_dL": "delta_phiS_delta_luminosity_distance",
    "the_dL": "delta_qS_delta_luminosity_distance",
}


def load_real_events_with_sky(apply_cuts: bool = True) -> list[dict]:
    """Real seed-600 events: measured (phi, theta, d_L) + the 3x3 Fisher covariance."""
    prep = pd.read_csv(B._PREPARED_CRB)
    raw = pd.read_csv(B._RAW_CRB)
    d_meas = prep["luminosity_distance"].to_numpy(float)
    sig_dL = np.sqrt(prep[_C["dL2"]].to_numpy(float))
    snr = prep["SNR"].to_numpy(float)
    keep = np.ones(len(prep), bool)
    if apply_cuts:
        keep = (snr >= 20.0) & (sig_dL / d_meas < 0.10)
    idx = np.where(keep)[0]
    events = []
    for i in idx:
        events.append(
            {
                "phi": float(prep["phiS"].iloc[i]),
                "theta": float(prep["qS"].iloc[i]),
                "d_meas": float(prep["luminosity_distance"].iloc[i]),
                "sigma_dL": float(np.sqrt(prep[_C["dL2"]].iloc[i])),
                "phi2": float(prep[_C["phi2"]].iloc[i]),
                "the2": float(prep[_C["the2"]].iloc[i]),
                "phi_the": float(prep[_C["phi_the"]].iloc[i]),
                "phi_dL": float(prep[_C["phi_dL"]].iloc[i]),
                "the_dL": float(prep[_C["the_dL"]].iloc[i]),
                "in_catalog": bool(raw["in_catalog"].iloc[i]),
            }
        )
    return events


def _cov3d(ev: dict) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, npt.NDArray[np.float64]]:
    """Build the normalised (phi, theta, d_L/d_meas) covariance exactly as
    bayesian_statistics.py:799-817; return (mean, cov_inv, log_norm, cov)."""
    d = ev["d_meas"]
    cov = np.array(
        [
            [ev["phi2"], ev["phi_the"], ev["phi_dL"] / d],
            [ev["phi_the"], ev["the2"], ev["the_dL"] / d],
            [ev["phi_dL"] / d, ev["the_dL"] / d, ev["sigma_dL"] ** 2 / d**2],
        ]
    )
    return _finish_cov(ev, cov)


def _cov3d_diag(ev: dict) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, npt.NDArray[np.float64]]:
    """Diagonal version (drop all d_L-sky / sky-sky correlations)."""
    d = ev["d_meas"]
    cov = np.diag([ev["phi2"], ev["the2"], ev["sigma_dL"] ** 2 / d**2])
    return _finish_cov(ev, cov)


def _finish_cov(ev: dict, cov: npt.NDArray[np.float64]):
    mean = np.array([ev["phi"], ev["theta"], 1.0])
    cov_inv = np.linalg.pinv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    log_norm = -0.5 * (3 * np.log(2 * np.pi) + logdet)
    return mean, cov_inv, float(log_norm), cov


def _polar_to_cart(phi: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


class SkyCatalog:
    """Real GLADE catalogue with sky, indexed by a unit-sphere KDTree."""

    def __init__(self, shuffle_sky: bool = False, seed: int = 0, max_zerr: float | None = None) -> None:
        from master_thesis_code.cosmological_model import Model1CrossCheck
        from master_thesis_code.galaxy_catalogue.handler import (
            GalaxyCatalogueHandler,
            InternalCatalogColumns,
        )

        cm = Model1CrossCheck(rng=np.random.default_rng(0))
        h = GalaxyCatalogueHandler(
            M_min=cm.parameter_space.M.lower_limit,
            M_max=cm.parameter_space.M.upper_limit,
            z_max=cm.max_redshift,
        )
        cat = h.reduced_galaxy_catalog
        z = cat[InternalCatalogColumns.REDSHIFT].to_numpy(float)
        M = cat[InternalCatalogColumns.BH_MASS].to_numpy(float)
        phi = cat[InternalCatalogColumns.PHI_S].to_numpy(float)
        theta = cat[InternalCatalogColumns.THETA_S].to_numpy(float)
        zerr = cat[InternalCatalogColumns.REDSHIFT_ERROR].to_numpy(float)
        good = (
            np.isfinite(z) & np.isfinite(M) & np.isfinite(phi) & np.isfinite(theta)
            & np.isfinite(zerr) & (z > 0)
        )
        z, M, phi, theta, zerr = z[good], M[good], phi[good], theta[good], zerr[good]
        if max_zerr is not None:
            # spectroscopic-only subset (GLADE flag-3 proxy): keep small-sigma_z hosts
            sel = zerr < max_zerr
            z, M, phi, theta, zerr = z[sel], M[sel], phi[sel], theta[sel], zerr[sel]
        if shuffle_sky:
            # break sky-z clustering while preserving the n(z) and sky MARGINALS
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(phi))
            phi, theta = phi[perm], theta[perm]
        self.z, self.M, self.phi, self.theta, self.zerr = z, M, phi, theta, zerr
        self.w = np.asarray(R_eff_per_mbh(M), float) / (1.0 + z)
        self.handler = h
        self.tree = cKDTree(_polar_to_cart(phi, theta))

    def candidates(self, phi: float, theta: float, radius_chord: float) -> npt.NDArray[np.int64]:
        q = _polar_to_cart(np.array([phi]), np.array([theta]))[0]
        return np.asarray(self.tree.query_ball_point(q, radius_chord), dtype=np.int64)


def _b_num(ev: dict, h: float, completeness, cov_inv, log_norm) -> float:
    """Completion numerator B_num = INTEGRAL (1-f_k(z)) p_GW(z) dVc/(1+z) dz.

    p_GW(z) is the 3-D MVN at the event sky (offset 0), i.e. a 1-D Gaussian in
    d_L/d_meas with the conditional precision cov_inv[2,2] -- exactly the real
    completion_numerator_integrand (bayesian_statistics.py:1446) collapsed to the
    event direction. f_k is evaluated at the event's HEALPix pixel.
    """
    from master_thesis_code.bayesian_inference.bayesian_statistics import dist_to_redshift
    from master_thesis_code.physical_relations import comoving_volume_element
    from scipy.integrate import fixed_quad

    d = ev["d_meas"]
    sdL = ev["sigma_dL"]
    pixel = completeness.ang2pix(ev["phi"], ev["theta"]) if hasattr(completeness, "ang2pix") else 0
    z_lo = max(dist_to_redshift(max(d - 4.0 * sdL, 1e-4), h=h), 1e-6)
    z_hi = dist_to_redshift(d + 4.0 * sdL, h=h)
    prec = float(cov_inv[2, 2])

    def integ(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        d_model = np.asarray(dist_vectorized(z, h=h), float)
        dl_frac = d_model / d
        p_gw = np.exp(log_norm - 0.5 * prec * (dl_frac - 1.0) ** 2)
        dVc = np.atleast_1d(np.asarray(comoving_volume_element(z, h=h), float))
        try:
            f_z = np.clip(np.asarray(completeness.f_k(z, pixel, h), float), 0.0, 1.0)
        except TypeError:
            f_z = np.clip(np.asarray(completeness.get_completeness_at_redshift(z, h), float), 0.0, 1.0)
        return (1.0 - f_z) * p_gw * dVc / (1.0 + z)

    return float(fixed_quad(integ, z_lo, z_hi, n=50)[0])


def event_loglik_sky(
    ev: dict,
    cat: SkyCatalog,
    h: float,
    D_h: float,
    beta_G: float,
    global_denom: float,
    *,
    mode: str = "mvn",  # "mvn" | "mvn_diag" | "1d" | "conv"
    sigma_mult: float = 4.0,
    completeness=None,
    include_bnum: bool = False,
    zerr_scale: float = 1.0,
    regularise_photoz: bool = False,
) -> float:
    """log p_i(h) = log[(beta_G * L_cat + B_num) / D(h)] using the sky channel.

    ``zerr_scale`` multiplies the catalogue photo-z error (mode="conv" only),
    for the sigma_z sweep that isolates the photo-z-domination bias.
    """
    d = ev["d_meas"]
    sdL = ev["sigma_dL"]
    mean, cov_inv, log_norm, _ = _cov3d_diag(ev) if mode == "mvn_diag" else _cov3d(ev)
    # sky search radius: sigma_mult * great-circle sky sigma (chord approx)
    ssky = np.sqrt(max(ev["phi2"] * np.sin(ev["theta"]) ** 2, ev["the2"]))
    radius = float(sigma_mult * max(ssky, 1e-3))
    cand = cat.candidates(ev["phi"], ev["theta"], radius)
    L_cat = 0.0
    if cand.size and mode == "conv":
        # host-z convolution: N_g = sky_weight * INTEGRAL gw(d_L(z,h)) norm(z;z_g,sigma_z) dz
        # (the real single_host_likelihood convolves each candidate with its photo-z PDF).
        from master_thesis_code.bayesian_inference.bayesian_statistics import dist_to_redshift

        zg = cat.z[cand]
        szg = np.maximum(cat.zerr[cand] * zerr_scale, 1e-5)
        # generous, H0-INDEPENDENT z-window widened by the photo-z scale
        zlo = max(dist_to_redshift(max(d - 5 * sdL, 1e-4), h=0.60) - 4 * float(np.median(szg)), 1e-5)
        zhi = dist_to_redshift(d + 5 * sdL, h=0.87) + 4 * float(np.median(szg))
        keep = (zg > zlo) & (zg < zhi)
        if np.any(keep):
            ci = cand[keep]
            zg = cat.z[ci]; szg = np.maximum(cat.zerr[ci] * zerr_scale, 1e-5); wc = cat.w[ci]
            # FULL 3-D MVN convolved over the host photo-z PDF (so conv -> mvn exactly
            # as sigma_z -> 0). Only the d_L axis depends on z; write the quadratic form
            #   dx(z) = (dphi, dthe, u(z)-1),  u(z) = d_L(z,h)/d_meas
            #   quad(z) = Q0 + 2 (u-1) Q1 + (u-1)^2 Q2,
            # with Q0 = dx_sky^T Cinv dx_sky, Q1 = dx_sky^T Cinv e_dL, Q2 = Cinv[2,2].
            dphi = cat.phi[ci] - mean[0]; dthe = cat.theta[ci] - mean[1]
            Ci = cov_inv
            Q0 = Ci[0, 0] * dphi**2 + 2 * Ci[0, 1] * dphi * dthe + Ci[1, 1] * dthe**2
            Q1 = Ci[0, 2] * dphi + Ci[1, 2] * dthe
            Q2 = float(Ci[2, 2])
            # z-grid (resolve the typical sigma_z robustly; one tiny-sigma_z outlier
            # must not blow up the grid).
            sz_res = max(float(np.percentile(szg, 10)), 1e-4)
            ngrid = int(np.clip((zhi - zlo) / (0.3 * sz_res), 200, 1200))
            zgrid = np.linspace(zlo, zhi, ngrid)
            um1 = np.asarray(dist_vectorized(zgrid, h=h), float) / d - 1.0  # (ngrid,)
            dzg = zgrid[1] - zgrid[0]
            quad = Q0[:, None] + 2.0 * um1[None, :] * Q1[:, None] + (um1[None, :] ** 2) * Q2
            mvn_z = np.exp(log_norm - 0.5 * quad)  # (ncand, ngrid)
            # host photo-z PDF norm(z; z_g, sigma_z)  -> (ncand, ngrid)
            nm = np.exp(-0.5 * ((zgrid[None, :] - zg[:, None]) / szg[:, None]) ** 2) / (
                np.sqrt(2 * np.pi) * szg[:, None]
            )
            if regularise_photoz:
                # EXP-1: comoving-volume-regularised host posterior p_red(z|z_g) =
                # norm(z;z_g,sigma_z) * p_bg(z) / Z_g, p_bg ∝ (1/(1+z)) dVc/dz
                # (Hitchhiker 2212.08694 Eq.16/32; Gray 2020 Eq.25).
                p_bg = np.asarray(comoving_volume_element(zgrid, h=h), float) / (1.0 + zgrid)
                nm = nm * p_bg[None, :]
                Z_g = (nm @ np.ones_like(zgrid)) * dzg
                nm = nm / np.maximum(Z_g, 1e-300)[:, None]
            N_g = np.sum(mvn_z * nm, axis=1) * dzg  # (ncand,)
            cat_num = float(np.sum(wc * N_g))
            L_cat = cat_num / global_denom if global_denom > 0 else 0.0
    elif cand.size:
        zc = cat.z[cand]
        d_model = np.asarray(dist_vectorized(zc, h=h), float)
        within = np.abs(d_model - d) < (sigma_mult + 1.0) * sdL
        if np.any(within):
            ci = cand[within]
            zc = cat.z[ci]
            wc = cat.w[ci]
            d_model = np.asarray(dist_vectorized(zc, h=h), float)
            if mode == "1d":
                from scipy.stats import norm

                vals = norm.pdf(d_model, loc=d, scale=sdL)
            else:
                x = np.column_stack([cat.phi[ci], cat.theta[ci], d_model / d])
                dx = x - mean
                quad = np.einsum("ij,jk,ik->i", dx, cov_inv, dx)
                vals = np.exp(log_norm - 0.5 * quad)
            cat_num = float(np.sum(wc * vals))
            L_cat = cat_num / global_denom if global_denom > 0 else 0.0
    B_num = 0.0
    if include_bnum and completeness is not None:
        B_num = _b_num(ev, h, completeness, cov_inv, log_norm)
    p_i = (beta_G * L_cat + B_num) / D_h if D_h > 0 else 0.0
    return float(np.log(p_i)) if p_i > 0 else -1e30


def make_real_pdet():
    """Build the REAL survival-function p_det from the injection campaign."""
    from master_thesis_code.bayesian_inference.simulation_detection_probability import (
        SimulationDetectionProbability,
    )
    from master_thesis_code.constants import INJECTION_DATA_DIR, SNR_THRESHOLD

    return SimulationDetectionProbability(
        injection_data_dir=INJECTION_DATA_DIR, snr_threshold=SNR_THRESHOLD
    )


def make_real_completeness():
    """Load the REAL frozen pixelated completeness (m_th map)."""
    from master_thesis_code.galaxy_catalogue.pixel_completeness import from_cache_or_build

    return from_cache_or_build()


def run_sky_rung(
    name: str,
    cat: SkyCatalog,
    events: list[dict],
    *,
    mode: str = "mvn",
    h_grid: list[float] | None = None,
    completeness: str = "declining",
    pdet_obj=None,
    completeness_obj=None,
    include_bnum: bool = False,
    sigma_mult: float = 4.0,
    zerr_scale: float = 1.0,
    regularise_photoz: bool = False,
) -> dict:
    """Run the sky in-catalogue channel over an h-grid.

    pdet_obj/completeness_obj default to the MOCK p_det and a toy f(z); pass the
    real objects (make_real_pdet / make_real_completeness) to swap in the real
    selection / pixelated completeness. include_bnum adds the completion term.
    """
    if h_grid is None:
        h_grid = [float(x) for x in np.round(np.arange(0.60, 0.8701, 0.01), 4)]
    comp = completeness_obj if completeness_obj is not None else B._make_completeness(
        B.BridgeConfig(name=name, completeness=completeness)
    )
    pdet = pdet_obj if pdet_obj is not None else B.MockPdet()
    D_tab = B.precompute_completion_denominator(h_grid, pdet, Omega_m=B._OMEGA_M, Omega_DE=B._OMEGA_DE)
    bGbar = B.precompute_missing_completion_denominator(h_grid, pdet, completeness=comp)
    # global selection denominator over the SkyCatalog's actual galaxies (so it stays
    # consistent when the catalogue is filtered, e.g. spectroscopic-only).
    gcat = B._ClosureCatalog(cat.z, cat.M)
    gdenom = B.precompute_global_catalog_selection(h_grid, gcat, pdet, with_bh_mass=False)
    logpost = np.zeros(len(h_grid))
    t0 = time.time()
    for i, h in enumerate(h_grid):
        D_h = D_tab[h]
        bG = D_h - bGbar[h]
        gd = gdenom[h]
        tot = 0.0
        for ev in events:
            tot += event_loglik_sky(
                ev, cat, h, D_h, bG, gd, mode=mode,
                completeness=comp, include_bnum=include_bnum, sigma_mult=sigma_mult,
                zerr_scale=zerr_scale, regularise_photoz=regularise_photoz,
            )
        logpost[i] = tot
    res = B.extract_map(h_grid, logpost, B.TRUE_H)
    res.update({"name": name, "mode": mode, "n_events": len(events),
                "include_bnum": include_bnum, "elapsed_s": round(time.time() - t0, 1)})
    print(f"[{name}] mode={mode} bnum={include_bnum} n={len(events)} MAP={res['h_refined']:.4f} "
          f"bias={res['bias']:+.4f} railed={res['railed']} ({res['elapsed_s']}s)", flush=True)
    return res
