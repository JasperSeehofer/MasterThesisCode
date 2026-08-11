"""End-to-end closure for the Change-5 per-HEALPix-pixel completeness inference.

A self-consistent mock pipeline with SKY structure: generate a galaxy population at
a known H0 with isotropic sky positions, tag each galaxy's HEALPix pixel, split it
into a per-pixel catalog (kept with probability ``f_k(z, pixel)``) and missing/dark
galaxies, inject rate-weighted EMRI events from the FULL population (each carrying
its host's pixel), give each a Gaussian luminosity-distance measurement and a smooth
detection probability, then run the partition-norm inference with the Change-5
per-pixel pieces and assert the MAP recovers H0:

* the completion numerator ``B_num`` evaluates ``f_k`` at the EVENT's pixel (Change 5.3);
* ``beta_Gbar`` uses the sky-average ``f_bar`` (Change 5.2);
* both sides call ONE shared :class:`PixelCompleteness` (C1).

This is the end-to-end correctness gate that the per-pixel completeness does NOT bias
H0 (positive test), plus a C1 negative control showing that breaking the
shared-map requirement (inference using a DIFFERENT m_th map than the injection) DOES
bias H0 -- the single H0-bias path the derivation identifies
(.planning/derivation-change5-healpix-estimator/DERIVATION.md Sec. 4).

Scope of THIS test (what it does and does NOT validate):
* It validates the INFERENCE side -- B_num at the event pixel (Change 5.3) + f_bar in
  beta_Gbar (Change 5.2) + ONE shared frozen f map (C1) -- CONDITIONAL on a correctly
  distributed injection. It injects the real population's true ``(z, pixel)`` joint
  directly (``_inject_events``); it deliberately does NOT call the production
  ``draw_dark_hosts`` / ``_draw_dark_hosts_pixelated`` sampler.
* The FIX-A ``W_k`` dark-host sampler (Change 5.5) is therefore validated SEPARATELY,
  end-to-end, in test_dark_event_injection.py
  (test_pixelated_dark_draw_*): that the sampler reproduces the JOINT
  ``p(z, k) ∝ (1 - f_k(z)) p_pop(z)`` -- pixel pick frequency tracks
  ``W_k = INTEGRAL (1 - f_k) p_pop dz`` (dark hosts cluster in low-completeness / ZoA
  directions) and ``z|k*`` follows the pixel's incompleteness-weighted population.
  This JOINT (z, Omega) correlation is the load-bearing FIX-A correction (DERIVATION
  Sec. 3, "the only surviving correction"); it is INVARIANT under the z-marginal
  (the old isotropic draw shares the same ``(1-f_bar)p_pop`` z-marginal), so the
  marginal cannot witness it -- which is exactly why FIX-A is validated at the joint
  distribution level, not via this closure's marginal recovery.

Findings encoded here (see the derivation Sec. 4-5):
* The per-pixel inference recovers H0 at the fiducial h=0.70 to |median bias| < 0.008.
* H0 unbiasedness comes from injection<->inference SELF-CONSISTENCY (the same frozen
  ``f_k``), NOT from ``f_model = f_real`` -- so a SHARED-map analysis is unbiased even
  though the estimator is conservative.
* A C1 violation (a DIFFERENT m_th map on the two sides) biases H0 by ~0.1 -- the
  single H0-bias path the derivation identifies (negative control below).

Off-center h recovery and the scalar-f regression are covered by
test_partition_norm_closure.py (Task A). Marked ``slow``.
"""

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from astropy_healpix import HEALPix
from scipy.integrate import fixed_quad
from scipy.stats import norm

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    precompute_completion_denominator,
    precompute_global_catalog_selection,
    precompute_missing_completion_denominator,
)
from darksiren_emri.emri_rate import R_eff_per_mbh, mbh_mass_function
from darksiren_emri.galaxy_catalogue.handler import InternalCatalogColumns
from darksiren_emri.galaxy_catalogue.pixel_completeness import PixelCompleteness
from darksiren_emri.physical_relations import (
    comoving_volume_element,
    dist,
    dist_to_redshift,
    dist_vectorized,
)

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]

_Z_MIN = 1e-3
_Z_MAX = 0.5
_M_MIN = 1.0e4
_M_MAX = 1.0e7
_D_HOR_GPC = 1.2
_H_MIN, _H_MAX = 0.60, 0.80  # same range as the Task-A closure
_NSIDE = 2
_HP = HEALPix(nside=_NSIDE, order="ring")


def _p_det_of_dl(d_L: FloatArr) -> FloatArr:
    """Smooth detection probability vs luminosity distance (Gpc), -> 0 past horizon."""
    d = np.asarray(d_L, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp((d - 0.7 * _D_HOR_GPC) / (0.08 * _D_HOR_GPC)))
    return np.asarray(np.where(d > _D_HOR_GPC, 0.0, p), dtype=np.float64)


class _MockPdet:
    """Sky-independent detection-probability stub (matches the Task-A closure)."""

    def get_dl_max(self, h: float) -> float:
        return _D_HOR_GPC

    def detection_probability_without_bh_mass_interpolated_zero_fill(
        self, d_L: FloatArr, phi: FloatArr, theta: FloatArr, *, h: float
    ) -> FloatArr:
        return _p_det_of_dl(d_L)

    def detection_probability_with_bh_mass_interpolated(
        self, d_L: FloatArr, M_z: FloatArr, phi: FloatArr, theta: FloatArr, *, h: float
    ) -> FloatArr:
        return _p_det_of_dl(d_L)


class _ClosureCatalog:
    """Minimal catalog handler exposing ``reduced_galaxy_catalog`` (z, M columns)."""

    def __init__(self, z: FloatArr, M: FloatArr) -> None:
        self.reduced_galaxy_catalog = pd.DataFrame(
            {InternalCatalogColumns.REDSHIFT: z, InternalCatalogColumns.BH_MASS: M}
        )


def _vec_ang2pix(phi: FloatArr, theta: FloatArr) -> IntArr:
    """Vectorized ecliptic (phi azimuth, theta colatitude) [rad] -> pixel index."""
    return np.asarray(
        _HP.lonlat_to_healpix(np.asarray(phi) * u.rad, (np.pi / 2.0 - np.asarray(theta)) * u.rad),
        dtype=np.int64,
    )


def _synthetic_pixel_completeness(m_th_shift: float = 0.0) -> PixelCompleteness:
    """A 48-pixel m_th map with a 3-level completeness gradient + two empty pixels.

    ``m_th_shift`` offsets every finite pixel's threshold (used to build a DIFFERENT
    inference map for the C1-violation negative control).
    """
    npix = 12 * _NSIDE * _NSIDE  # 48
    m_th = np.empty(npix, dtype=np.float64)
    third = npix // 3
    m_th[:third] = 21.0  # most complete
    m_th[third : 2 * third] = 20.0
    m_th[2 * third :] = 18.5  # least complete
    m_th[5] = -np.inf  # empty / ZoA
    m_th[20] = -np.inf
    finite = np.isfinite(m_th)
    m_th[finite] += m_th_shift
    return PixelCompleteness(m_th, nside=_NSIDE)


def _f_k_vectorized(pc: PixelCompleteness, z: FloatArr, pix: IntArr, h: float) -> FloatArr:
    """Per-galaxy ``f_k(z_g, pixel_g)`` (grouped by pixel for speed)."""
    out = np.zeros_like(z)
    for p in np.unique(pix):
        mask = pix == p
        out[mask] = np.asarray(pc.f_k(z[mask], int(p), h), dtype=np.float64)
    return np.clip(out, 0.0, 1.0)


def _sample_population(
    rng: np.random.Generator, n_gal: int, h_true: float
) -> tuple[FloatArr, FloatArr, IntArr]:
    """Full population: z ~ (1/(1+z)) dVc/dz, log10 M ~ phi_MBH(M) R_eff(M), isotropic sky."""
    zg = np.linspace(_Z_MIN, _Z_MAX, 4000)
    wz = np.asarray(comoving_volume_element(zg, h=h_true), dtype=np.float64) / (1.0 + zg)
    cdf = np.cumsum(wz)
    cdf /= cdf[-1]
    z = np.asarray(np.interp(rng.uniform(size=n_gal), cdf, zg), dtype=np.float64)
    lmg = np.linspace(np.log10(_M_MIN), np.log10(_M_MAX), 2000)
    wm = np.asarray(mbh_mass_function(10.0**lmg), dtype=np.float64) * np.asarray(
        R_eff_per_mbh(10.0**lmg), dtype=np.float64
    )
    cdfm = np.cumsum(wm)
    cdfm /= cdfm[-1]
    M = np.asarray(10.0 ** np.interp(rng.uniform(size=n_gal), cdfm, lmg), dtype=np.float64)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_gal)
    theta = np.arccos(rng.uniform(-1.0, 1.0, n_gal))
    return z, M, _vec_ang2pix(phi, theta)


def _inject_events(
    rng: np.random.Generator,
    z_all: FloatArr,
    M_all: FloatArr,
    pix_all: IntArr,
    h_true: float,
    n_target: int,
    sigma_frac: float,
) -> list[dict[str, float]]:
    """Rate-weighted hosts from the FULL population -> Gaussian d_L -> detection filter.

    Each event carries its host's pixel; dark events are simply population hosts whose
    galaxy did not enter the catalog (the same population the W_k sampler reconstructs).
    """
    w = np.asarray(R_eff_per_mbh(M_all), dtype=np.float64) / (1.0 + z_all)
    p = w / w.sum()
    events: list[dict[str, float]] = []
    tries = 0
    while len(events) < n_target and tries < 400 * n_target:
        tries += 1
        g = int(rng.choice(len(z_all), p=p))
        z_host = float(z_all[g])
        d_true = float(dist(z_host, h=h_true))
        sigma_dL = sigma_frac * d_true
        d_meas = d_true + sigma_dL * rng.standard_normal()
        if d_meas <= 0:
            continue
        if rng.uniform() < float(_p_det_of_dl(np.asarray([d_meas]))[0]):
            events.append({"d_meas": d_meas, "sigma_dL": sigma_dL, "pixel": float(pix_all[g])})
    return events


def _event_log_likelihood(
    event: dict[str, float],
    catalog_z: FloatArr,
    catalog_M: FloatArr,
    catalog_pix: IntArr,
    completeness: PixelCompleteness,
    h: float,
    D_h: float,
    beta_G: float,
    global_denom: float,
) -> float:
    """log p_i(h) = log[(beta_G L_cat + B_num) / D(h)] with per-pixel completeness."""
    d_meas = event["d_meas"]
    sigma_dL = event["sigma_dL"]
    event_pixel = int(event["pixel"])

    def p_gw_of_z(z: FloatArr) -> FloatArr:
        d_model = np.asarray(dist_vectorized(z, h=h), dtype=np.float64)
        return np.asarray(norm.pdf(d_model, loc=d_meas, scale=sigma_dL), dtype=np.float64)

    # L_cat over catalog galaxies IN THE EVENT PIXEL and z-window (the per-direction
    # in-catalog term; analog of the production sky-localization ball).
    z_lo = dist_to_redshift(max(d_meas - 5.0 * sigma_dL, 1e-4), h=_H_MIN)
    z_hi = dist_to_redshift(d_meas + 5.0 * sigma_dL, h=_H_MAX)
    cand = (catalog_pix == event_pixel) & (catalog_z >= z_lo) & (catalog_z <= z_hi)
    cat_num_sum = 0.0
    if np.any(cand):
        zc = catalog_z[cand]
        wc = np.asarray(R_eff_per_mbh(catalog_M[cand]), dtype=np.float64) / (1.0 + zc)
        cat_num_sum = float(np.sum(wc * p_gw_of_z(zc)))
    L_cat = cat_num_sum / global_denom if global_denom > 0 else 0.0

    # B_num with the EVENT-PIXEL incompleteness (1 - f_{k(event)}(z)) (Change 5.3).
    bz_lo = max(dist_to_redshift(max(d_meas - 4.0 * sigma_dL, 1e-4), h=h), 1e-6)
    bz_hi = dist_to_redshift(d_meas + 4.0 * sigma_dL, h=h)

    def b_integrand(z: FloatArr) -> FloatArr:
        f_z = np.clip(np.asarray(completeness.f_k(z, event_pixel, h), dtype=np.float64), 0.0, 1.0)
        dVc = np.asarray(comoving_volume_element(z, h=h), dtype=np.float64)
        return (1.0 - f_z) * p_gw_of_z(z) * dVc / (1.0 + z)

    B_num = float(fixed_quad(b_integrand, bz_lo, bz_hi, n=50)[0])
    p_i = (beta_G * L_cat + B_num) / D_h if D_h > 0 else 0.0
    return float(np.log(p_i)) if p_i > 0 else -1e30


def _map_bias(
    *,
    h_true: float,
    seed: int,
    inference_m_th_shift: float = 0.0,
    n_gal: int = 14000,
    n_events: int = 300,
    sigma_frac: float = 0.015,
) -> float:
    """One pixel-f closure realization; MAP(H0) - h_true (parabola-refined).

    ``inference_m_th_shift`` != 0 builds a DIFFERENT inference completeness map than the
    one used to split the catalog / inject events -> a C1 violation. Realizations whose
    MAP lands at a grid edge are under-constrained (finite-N variance, not bias); they
    return the edge offset so the median over seeds absorbs them (as in the Task-A test).
    """
    rng = np.random.default_rng(seed)
    pc_truth = _synthetic_pixel_completeness()
    pc_infer = (
        _synthetic_pixel_completeness(inference_m_th_shift) if inference_m_th_shift else pc_truth
    )

    z_all, M_all, pix_all = _sample_population(rng, n_gal, h_true)
    f_at = _f_k_vectorized(pc_truth, z_all, pix_all, h_true)
    in_cat = rng.uniform(size=n_gal) < f_at
    cz, cM, cpix = z_all[in_cat], M_all[in_cat], pix_all[in_cat]
    catalog = _ClosureCatalog(cz, cM)
    events = _inject_events(rng, z_all, M_all, pix_all, h_true, n_events, sigma_frac)

    pdet = _MockPdet()
    hs = [float(h) for h in np.round(np.arange(_H_MIN, _H_MAX + 1e-9, 0.02), 4)]
    D_tab = precompute_completion_denominator(hs, pdet, Omega_m=0.25, Omega_DE=0.75)  # type: ignore[arg-type]
    bGbar_tab = precompute_missing_completion_denominator(hs, pdet, completeness=pc_infer)  # type: ignore[arg-type]
    gd_tab = precompute_global_catalog_selection(hs, catalog, pdet, with_bh_mass=False)  # type: ignore[arg-type]

    logpost = np.zeros(len(hs))
    for i, h in enumerate(hs):
        beta_G = D_tab[h] - bGbar_tab[h]
        logpost[i] = sum(
            _event_log_likelihood(ev, cz, cM, cpix, pc_infer, h, D_tab[h], beta_G, gd_tab[h])
            for ev in events
        )
    logpost -= logpost.max()
    i_map = int(np.argmax(logpost))
    if i_map == 0 or i_map == len(hs) - 1:
        return float(hs[i_map] - h_true)  # under-constrained; median absorbs it
    y0, y1, y2 = logpost[i_map - 1], logpost[i_map], logpost[i_map + 1]
    denom = y0 - 2 * y1 + y2
    dh = hs[1] - hs[0]
    h_map = hs[i_map] + (0.5 * (y0 - y2) / denom * dh if denom != 0 else 0.0)
    return float(h_map - h_true)


def _median_bias(*, h_true: float, inference_m_th_shift: float = 0.0, n_seeds: int = 7) -> float:
    """Median MAP bias over ``n_seeds`` realizations (median is the unbiasedness statistic)."""
    return float(
        np.median(
            [
                _map_bias(h_true=h_true, seed=s, inference_m_th_shift=inference_m_th_shift)
                for s in range(n_seeds)
            ]
        )
    )


@pytest.mark.slow
def test_closure_pixel_f_recovers_h0() -> None:
    """Per-pixel completeness (B_num at event pixel + f_bar) recovers the injected H0."""
    bias = _median_bias(h_true=0.70)
    assert abs(bias) < 0.008, f"pixel-f median MAP bias {bias:+.4f} exceeds 0.008"


@pytest.mark.slow
def test_closure_breaks_when_inference_map_differs() -> None:
    """C1 negative control: inference using a DIFFERENT m_th map than injection biases H0.

    This is the single H0-bias path the derivation identifies (DERIVATION Sec. 4):
    injection and inference MUST share one frozen f map. Shifting the inference map's
    thresholds (less-complete) breaks the byte-identity and biases the MAP far beyond
    the unbiased tolerance, proving the shared-map (C1) requirement is load-bearing.
    """
    bias = _median_bias(h_true=0.70, inference_m_th_shift=-1.5)
    assert abs(bias) > 0.02, (
        f"C1-violating inference (wrong m_th map) should bias H0; got {bias:+.4f}"
    )
