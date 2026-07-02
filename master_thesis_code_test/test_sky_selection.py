"""Tests for the sky-aware (ecliptic-latitude) selection function.

Covers PHYSICS-CHANGE-PROTOCOL "Sky-Aware Selection Function" Changes 1-4:

* Change 1 -- per-ecliptic-latitude-band detection-horizon survival p_det(d_L|Omega)
  re-binned from the existing isotropic injection pool (Route A, empirical).
* Change 2 -- full-volume selection D(h) as the per-pixel sky sum.
* Change 3 -- missing-completion beta_Gbar(h) as the per-pixel (1-f_k) p_det sum.
* Change 4 -- global in-catalog denominator using each galaxy's real ecliptic sky.

Tests (protocol Sec. 6):

* T1 -- ISOTROPIC-LIMIT REGRESSION (machine precision): with n_sky_bands=1 the
        sky path reproduces the isotropic D, beta_Gbar, Sigma_global bit-for-bit.
* T2 -- partition identity D(h) == beta_G(h) + beta_Gbar(h) (per-pixel f+(1-f)=1).
* T3 -- sky-marginal invariance: the injection-count-weighted band-survival average
        equals the pooled isotropic survival (identical band edges).
* T4 -- north-south (|beta|) symmetry of the injection survival.
* T5 -- band-count convergence (Nband 4->6->8 stable).
* T6 -- anisotropic closure (MANDATORY acceptance, slow): sky-aware selection
        recovers the true anisotropic-population H0; isotropic selection is biased
        (negative control).
* T7 -- frame assertion (BarycentricTrueEcliptic J2000 ecliptic pole).

References:
    Cutler (1998), arXiv:gr-qc/9703068 -- LISA orbit-averaged response R(beta).
    Gray, Gerosa et al. (2023), arXiv:2308.02281, Eq. 2.3 -- per-pixel selection sum.
    Gray-Messenger-Veitch (2022), arXiv:2111.04629, Eq. 5 -- pixelated in/out split.
    Mandel-Farr-Gair (2019), arXiv:1809.02063, Eq. 6 -- selection self-consistency.
    Hogg (1999), arXiv:astro-ph/9905116, Eq. 16 -- 1/d_L amplitude scaling.
"""

import os
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import fixed_quad

from master_thesis_code.bayesian_inference.bayesian_statistics import (
    precompute_completion_denominator,
    precompute_global_catalog_selection,
    precompute_missing_completion_denominator,
)
from master_thesis_code.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
)
from master_thesis_code.galaxy_catalogue.pixel_completeness import (
    CompletenessModel,
    PixelCompleteness,
    from_cache_or_build,
)
from master_thesis_code.physical_relations import (
    comoving_volume_element,
    dist_to_redshift,
    dist_vectorized,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_INJECTION_DIR = str(_REPO_ROOT / "simulations" / "injections")
_H = 0.73
_OMEGA_M = 0.25
_OMEGA_DE = 0.75

_HAVE_INJECTIONS = os.path.isdir(_INJECTION_DIR) and bool(
    list(Path(_INJECTION_DIR).glob("injection_h_*.csv"))
)
_needs_injections = pytest.mark.skipif(
    not _HAVE_INJECTIONS, reason="canonical injection CSVs not present"
)


# ======================================================================
# Fixtures: real injection pools + real per-pixel completeness
# ======================================================================


@pytest.fixture(scope="module")
def completeness() -> PixelCompleteness:
    return from_cache_or_build()


@pytest.fixture(scope="module")
def pdet_iso() -> SimulationDetectionProbability:
    """Real pooled injections, single sky band (the isotropic regression fallback)."""
    return SimulationDetectionProbability(_INJECTION_DIR, snr_threshold=20.0, n_sky_bands=1)


@pytest.fixture(scope="module")
def pdet_sky() -> SimulationDetectionProbability:
    """Real pooled injections, 6 equal-|sin beta| bands (production default)."""
    return SimulationDetectionProbability(_INJECTION_DIR, snr_threshold=20.0, n_sky_bands=6)


class _FakeCatalog:
    """Minimal catalog handler exposing ``reduced_galaxy_catalog``.

    ``with_sky=False`` omits PHI_S/THETA_S so the global selection takes the
    isotropic fallback (used for the T1 regression reference).
    """

    def __init__(
        self,
        z: np.ndarray,
        M: np.ndarray,  # noqa: N803
        phi: np.ndarray,
        theta: np.ndarray,
        *,
        with_sky: bool,
    ) -> None:
        data = {
            InternalCatalogColumns.REDSHIFT: z,
            InternalCatalogColumns.BH_MASS: M,
        }
        if with_sky:
            data[InternalCatalogColumns.PHI_S] = phi
            data[InternalCatalogColumns.THETA_S] = theta
        self.reduced_galaxy_catalog = pd.DataFrame(data)


def _make_catalog(seed: int, n: int, *, with_sky: bool) -> _FakeCatalog:
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.01, 0.20, n)
    M = rng.uniform(1e5, 5e6, n)  # noqa: N806
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    theta = np.arccos(rng.uniform(-1.0, 1.0, n))  # ecliptic colatitude
    return _FakeCatalog(z, M, phi, theta, with_sky=with_sky)


# ======================================================================
# T1 -- ISOTROPIC-LIMIT REGRESSION (MANDATORY, machine precision)
# ======================================================================


@_needs_injections
def test_T1_D_regression_nband1_equals_isotropic(  # noqa: N802
    pdet_iso: SimulationDetectionProbability, completeness: PixelCompleteness
) -> None:
    """D(h) via the sky path with n_sky_bands=1 == the isotropic D(h) to ~1e-12."""
    d_sky = precompute_completion_denominator(
        [_H], pdet_iso, _OMEGA_M, _OMEGA_DE, completeness=completeness
    )[_H]
    d_iso = precompute_completion_denominator(
        [_H], pdet_iso, _OMEGA_M, _OMEGA_DE, completeness=None
    )[_H]
    assert d_sky == pytest.approx(d_iso, rel=1e-12, abs=0.0)


@_needs_injections
def test_T1_beta_gbar_regression_nband1_equals_isotropic(  # noqa: N802
    pdet_iso: SimulationDetectionProbability, completeness: PixelCompleteness
) -> None:
    """beta_Gbar via the sky path (n_sky_bands=1) == the isotropic f_bar path to ~1e-12."""
    bg_sky = precompute_missing_completion_denominator([_H], pdet_iso, completeness)[_H]

    # Isotropic reference: (1 - f_bar(z)) <p_det>_iso, computed with the SAME
    # quadrature the production integral uses.
    z_max = dist_to_redshift(pdet_iso.get_dl_max(_H), h=_H)

    def _iso_integrand(z: np.ndarray) -> np.ndarray:
        d_L = np.asarray(dist_vectorized(z, h=_H), dtype=np.float64)
        p_det = np.asarray(
            pdet_iso.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L, np.zeros_like(z), np.zeros_like(z), h=_H
            ),
            dtype=np.float64,
        )
        dVc = np.atleast_1d(np.asarray(comoving_volume_element(z, h=_H), dtype=np.float64))  # noqa: N806
        f_z = np.clip(np.asarray(completeness.f_bar(z, _H), dtype=np.float64), 0.0, 1.0)
        return (1.0 - f_z) * p_det * dVc / (1.0 + z)

    bg_iso = fixed_quad(_iso_integrand, 1e-6, z_max, n=100)[0]
    assert bg_sky == pytest.approx(bg_iso, rel=1e-12, abs=0.0)


@_needs_injections
def test_T1_sigma_global_regression_nband1_equals_isotropic(  # noqa: N802
    pdet_iso: SimulationDetectionProbability,
) -> None:
    """Sigma_global with real galaxy sky (n_sky_bands=1) == the isotropic sum to ~1e-12."""
    cat_sky = _make_catalog(seed=3, n=400, with_sky=True)
    cat_iso = _make_catalog(seed=3, n=400, with_sky=False)
    sig_sky = precompute_global_catalog_selection(
        [_H], cast(GalaxyCatalogueHandler, cat_sky), pdet_iso, with_bh_mass=False
    )[_H]
    sig_iso = precompute_global_catalog_selection(
        [_H], cast(GalaxyCatalogueHandler, cat_iso), pdet_iso, with_bh_mass=False
    )[_H]
    assert sig_sky == pytest.approx(sig_iso, rel=1e-12, abs=0.0)


# ======================================================================
# T2 -- PARTITION IDENTITY (MANDATORY)
# ======================================================================


@_needs_injections
def test_T2_partition_D_equals_betaG_plus_betaGbar(  # noqa: N802
    pdet_sky: SimulationDetectionProbability, completeness: PixelCompleteness
) -> None:
    """D(h) == beta_G(h) + beta_Gbar(h) with beta_G the INDEPENDENT f-weighted sum.

    Verifies the per-(band, z) partition ``c_b = Sf_b + S1mf_b`` (from
    ``f_k + (1 - f_k) = 1``), not merely the definitional ``beta_G := D - beta_Gbar``.
    """
    d_h = precompute_completion_denominator(
        [_H], pdet_sky, _OMEGA_M, _OMEGA_DE, completeness=completeness
    )[_H]
    beta_gbar = precompute_missing_completion_denominator([_H], pdet_sky, completeness)[_H]

    # Independent beta_G(h) = INTEGRAL (1/Npix) sum_k f_k p_det(Omega_k) dVc/(1+z).
    phi_k, theta_k = completeness.pixel_centers()
    u_k = np.abs(np.cos(np.asarray(theta_k, dtype=np.float64)))
    edges = pdet_sky.band_edges_sin_beta()
    n_bands = edges.size - 1
    band_of_pix = np.clip(np.searchsorted(edges, u_k, side="right") - 1, 0, n_bands - 1)
    membership = (band_of_pix[None, :] == np.arange(n_bands)[:, None]).astype(np.float64)
    npix = u_k.size
    z_max = dist_to_redshift(pdet_sky.get_dl_max(_H), h=_H)

    def _betaG_integrand(z: np.ndarray) -> np.ndarray:  # noqa: N802
        d_L = np.asarray(dist_vectorized(z, h=_H), dtype=np.float64)
        f_pix = np.clip(
            np.asarray(completeness.f_pixels(z, _H), dtype=np.float64),
            0.0,
            1.0,
        )  # (Z, npix)
        sf_b = (membership @ f_pix.T) / float(npix)  # (n_bands, Z)
        s_band = np.asarray(pdet_sky.survival_per_band(d_L), dtype=np.float64)
        integrand = np.einsum("bz,bz->z", sf_b, s_band)
        dVc = np.atleast_1d(np.asarray(comoving_volume_element(z, h=_H), dtype=np.float64))  # noqa: N806
        return np.asarray(integrand * dVc / (1.0 + z))

    beta_g = fixed_quad(_betaG_integrand, 1e-6, z_max, n=100)[0]
    assert d_h == pytest.approx(beta_g + beta_gbar, rel=1e-12, abs=0.0)


# ======================================================================
# T3 -- SKY-MARGINAL INVARIANCE (MANDATORY)
# ======================================================================


@_needs_injections
def test_T3_injection_weighted_band_average_equals_pooled(  # noqa: N802
    pdet_sky: SimulationDetectionProbability,
) -> None:
    """The injection-count-weighted band survival average == the pooled survival.

    Exact identity (to machine precision) because the bands PARTITION the
    injection set with the SAME edges: sum_b (N_b/N) S_b == (1/N) sum_all.
    """
    dl = np.linspace(0.0, pdet_sky.get_dl_max(_H) * 0.95, 25)
    s_band = pdet_sky.survival_per_band(dl)  # (n_bands, N)
    n_b = np.asarray(pdet_sky._n_inj_by_band, dtype=np.float64)
    inj_weight = n_b / n_b.sum()
    pooled = pdet_sky._survival_at(dl)
    band_avg = inj_weight @ s_band
    assert np.max(np.abs(band_avg - pooled)) < 1e-12


@_needs_injections
def test_T3_pixel_weighted_band_average_approximates_pooled(  # noqa: N802
    pdet_sky: SimulationDetectionProbability, completeness: PixelCompleteness
) -> None:
    """The equal-area (pixel-count) band average is close to the pooled survival.

    Not exact (pixel-count vs injection-count per band differ by pixelization +
    Poisson noise), but small for isotropic injections in equal-solid-angle bands.
    """
    phi_k, theta_k = completeness.pixel_centers()
    u_k = np.abs(np.cos(np.asarray(theta_k, dtype=np.float64)))
    edges = pdet_sky.band_edges_sin_beta()
    n_bands = edges.size - 1
    band_of_pix = np.clip(np.searchsorted(edges, u_k, side="right") - 1, 0, n_bands - 1)
    c_b = np.bincount(band_of_pix, minlength=n_bands).astype(np.float64) / u_k.size

    dl = np.linspace(0.0, pdet_sky.get_dl_max(_H) * 0.95, 25)
    band_avg = c_b @ pdet_sky.survival_per_band(dl)
    pooled = pdet_sky._survival_at(dl)
    assert np.max(np.abs(band_avg - pooled)) < 0.02


# ======================================================================
# T4 -- NORTH-SOUTH (|beta|) SYMMETRY (verify, do NOT assume)
# ======================================================================


@_needs_injections
def test_T4_north_south_survival_symmetry(  # noqa: N802
    pdet_sky: SimulationDetectionProbability,
) -> None:
    """Injection survival is symmetric under beta -> -beta (justifies |beta| folding)."""
    sin_beta = np.cos(pdet_sky._qS_arr)  # signed sin(beta) = cos(qS)
    d_hor = pdet_sky._d_hor
    dl = np.linspace(0.05, 0.5, 12)

    def _surv(mask: np.ndarray) -> np.ndarray:
        h_sorted = np.sort(d_hor[mask])
        n = h_sorted.size
        return (n - np.searchsorted(h_sorted, dl, side="left")) / n

    max_asym = 0.0
    for lo, hi in [(0.0, 0.33), (0.33, 0.66), (0.66, 1.0)]:
        north = (sin_beta >= lo) & (sin_beta < hi)
        south = (sin_beta <= -lo) & (sin_beta > -hi)
        max_asym = max(max_asym, float(np.max(np.abs(_surv(north) - _surv(south)))))
    # Statistical (per-band Poisson); comfortably below the ~8% response modulation.
    assert max_asym < 0.02


# ======================================================================
# T5 -- BAND-COUNT CONVERGENCE
# ======================================================================


@_needs_injections
def test_T5_band_count_convergence(completeness: PixelCompleteness) -> None:  # noqa: N802
    """D(h) and beta_Gbar(h) are stable as n_sky_bands = 4 -> 6 -> 8."""
    d_vals = {}
    bg_vals = {}
    for nb in (4, 6, 8):
        pdet = SimulationDetectionProbability(_INJECTION_DIR, snr_threshold=20.0, n_sky_bands=nb)
        d_vals[nb] = precompute_completion_denominator(
            [_H], pdet, _OMEGA_M, _OMEGA_DE, completeness=completeness
        )[_H]
        bg_vals[nb] = precompute_missing_completion_denominator([_H], pdet, completeness)[_H]

    d_arr = np.array(list(d_vals.values()))
    bg_arr = np.array(list(bg_vals.values()))
    d_spread = (d_arr.max() - d_arr.min()) / d_arr.mean()
    bg_spread = (bg_arr.max() - bg_arr.min()) / bg_arr.mean()
    assert d_spread < 5e-3, f"D(h) band-count spread {d_spread:.2e} too large"
    assert bg_spread < 1e-2, f"beta_Gbar band-count spread {bg_spread:.2e} too large"


# ======================================================================
# T6 -- ANISOTROPIC CLOSURE (MANDATORY acceptance, + negative control, slow)
# ======================================================================


def _synthetic_sky_injections(dirpath: str, rng: np.random.Generator, n: int = 120_000) -> None:
    """Isotropic-sky injection pool whose horizon encodes a strong R(beta).

    ``R(u) = 1 + 0.45 (1 - 2u)`` (peak at the ecliptic plane u=|sin beta|=0,
    trough at the poles u=1); amplified vs the real catalog so the closure has
    discriminating power on the sky axis.  Sky is drawn ISOTROPICALLY (uniform in
    cos theta), so no catalog circularity enters the estimator (Route A).
    """
    z = rng.uniform(0.02, 1.2, n)
    q_s = np.arccos(rng.uniform(-1.0, 1.0, n))  # isotropic ecliptic colatitude
    u = np.abs(np.cos(q_s))  # |sin beta|
    d_l = dist_vectorized(z, h=_H)
    d_hor = 3.0 * (1.0 + 0.45 * (1.0 - 2.0 * u))  # R(u)
    snr = 20.0 * d_hor / d_l  # so d_hor = SNR d_L / thr recovers R(u)
    pd.DataFrame(
        {
            "z": z,
            "M": rng.uniform(1e5, 5e5, n),
            "phiS": rng.uniform(0.0, 2.0 * np.pi, n),
            "qS": q_s,
            "SNR": snr,
            "h_inj": _H,
            "luminosity_distance": d_l,
        }
    ).to_csv(os.path.join(dirpath, "injection_h_0p73_task_0.csv"), index=False)


class _CorrelatedSkyCompleteness:
    """Synthetic per-pixel completeness correlated with ecliptic latitude.

    ``f_k`` high near the ecliptic plane, low near the poles, so the missing
    fraction ``(1 - f_k)`` correlates with the (also latitude-dependent) response
    p_det -- inducing a NON-ZERO Cov[(1-f), p_det] that the isotropic factorized
    selection cannot capture (the negative control), but the per-pixel sum can.
    """

    def __init__(self, npix: int = 3072) -> None:
        cos_theta = (np.arange(npix) + 0.5) / npix * 2.0 - 1.0
        self._theta = np.arccos(cos_theta)
        self._u = np.abs(cos_theta)
        self._f_base = np.clip(0.9 - 0.8 * self._u, 0.0, 1.0)
        self.npix = npix

    def pixel_centers(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.npix), self._theta

    def f_pixels(self, z: np.ndarray, h: float) -> np.ndarray:
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        decline = np.clip(1.0 - 0.5 * z_arr, 0.05, 1.0)
        return np.clip(self._f_base[None, :] * decline[:, None], 0.0, 1.0)

    def f_bar(self, z: np.ndarray, h: float) -> np.ndarray | float:
        out = self.f_pixels(z, h).mean(axis=1)
        return float(out[0]) if np.ndim(z) == 0 else out


class _IsoShim:
    """Hides pixel_centers/f_pixels so beta_Gbar takes the isotropic f_bar path."""

    def __init__(self, comp: _CorrelatedSkyCompleteness) -> None:
        self._comp = comp

    def f_bar(self, z: np.ndarray, h: float) -> np.ndarray | float:
        return self._comp.f_bar(z, h)


@pytest.mark.slow
@_needs_injections
def test_T6_anisotropic_closure_debiases_H0() -> None:  # noqa: N802
    """Sky-aware selection recovers the true anisotropic-population H0; iso is biased.

    Positive: the sky-resolved beta_Gbar reproduces the TRUE anisotropic-population
    selection (dark hosts ~ (1-f_k) sky, real per-pixel response), so the inferred
    H0 matches the true-selection inference to <~0.1%.
    Negative control: the isotropic factorized selection deviates from truth by a
    clearly larger margin (reproduces the sky-selection bias the fix removes).
    """
    rng = np.random.default_rng(11)
    tmp = tempfile.mkdtemp()
    _synthetic_sky_injections(tmp, rng)
    pdet = SimulationDetectionProbability(tmp, snr_threshold=20.0, n_sky_bands=6)
    comp = _CorrelatedSkyCompleteness()

    hs = np.linspace(0.64, 0.82, 19)
    hs_list = hs.tolist()

    bg_sky = precompute_missing_completion_denominator(hs_list, pdet, cast(CompletenessModel, comp))
    bg_iso = precompute_missing_completion_denominator(
        hs_list, pdet, cast(CompletenessModel, _IsoShim(comp))
    )

    # TRUE selection: MC over dark-host sky ~ (1-f_k), response per real pixel.
    theta_pix = comp._theta
    pix_idx = np.arange(0, comp.npix, 48)

    def _d_true(h: float) -> float:
        z_max = dist_to_redshift(pdet.get_dl_max(h), h=h)

        def _integ(z: np.ndarray) -> np.ndarray:
            d_l = np.asarray(dist_vectorized(z, h=h), dtype=np.float64)
            f_pix = comp.f_pixels(z, h)[:, pix_idx]
            s = np.stack(
                [
                    np.asarray(
                        pdet.detection_probability_without_bh_mass_sky(
                            d_l, np.zeros_like(d_l), np.full_like(d_l, theta_pix[k]), h=h
                        )
                    )
                    for k in pix_idx
                ],
                axis=1,
            )
            val = ((1.0 - f_pix) * s).mean(axis=1)
            dVc = np.atleast_1d(np.asarray(comoving_volume_element(z, h=h), dtype=np.float64))  # noqa: N806
            return np.asarray(val * dVc / (1.0 + z))

        return float(fixed_quad(_integ, 1e-6, z_max, n=32)[0])

    d_true = {h: _d_true(h) for h in hs_list}

    # Selection SHAPE discrepancy (normalized at h_true) -- the H0-carrying axis.
    def _norm(tab: dict[float, float]) -> np.ndarray:
        v = np.array([tab[h] for h in hs_list])
        return np.asarray(v / v[int(np.argmin(np.abs(hs - _H)))])

    shape_err_sky = float(np.max(np.abs(_norm(bg_sky) - _norm(d_true))))
    shape_err_iso = float(np.max(np.abs(_norm(bg_iso) - _norm(d_true))))

    # --- single-host dark-siren H0 posterior (stable, H0-sensitive) ---
    def _gen_events(n_target: int = 2500) -> np.ndarray:
        f_ref = comp.f_pixels(np.array([0.3]), _H)[0]
        w = 1.0 - f_ref
        w /= w.sum()
        u_pix = comp._u
        idxp = np.arange(comp.npix)
        z_out: list[float] = []
        while len(z_out) < n_target:
            m = 30_000
            z = rng.uniform(0.05, 1.1, m)
            pp = np.asarray(comoving_volume_element(z, h=_H)) / (1.0 + z)
            pp /= pp.max()
            z = z[rng.uniform(0.0, 1.0, m) < pp]
            k = rng.choice(idxp, size=len(z), p=w)
            detected = dist_vectorized(z, h=_H) <= 3.0 * (1.0 + 0.45 * (1.0 - 2.0 * u_pix[k]))
            z_out.extend(z[detected].tolist())
        return np.array(z_out[:n_target])

    z_cat = _gen_events()
    s_frac = 0.10
    d_l_true = dist_vectorized(z_cat, h=_H)
    d_l_obs = d_l_true * (1.0 + rng.normal(0.0, s_frac, z_cat.size))
    sig = s_frac * d_l_true
    sz = 5e-4 * (1.0 + z_cat)

    def _num_i(h: float) -> np.ndarray:
        z_g = np.linspace(0.01, 1.2, 300)
        d_lz = dist_vectorized(z_g, h=h)
        n_term = np.exp(-0.5 * ((d_l_obs[:, None] - d_lz[None, :]) / sig[:, None]) ** 2)
        z_term = np.exp(-0.5 * ((z_g[None, :] - z_cat[:, None]) / sz[:, None]) ** 2)
        return np.asarray(np.trapezoid(n_term * z_term, z_g, axis=1))

    def _posterior_mean(sel: dict[float, float]) -> float:
        logp = np.array(
            [np.sum(np.log(_num_i(h) + 1e-300)) - z_cat.size * np.log(sel[h]) for h in hs_list]
        )
        logp -= logp.max()
        p = np.exp(logp)
        p /= np.trapezoid(p, hs)
        return float(np.trapezoid(hs * p, hs))

    mean_true = _posterior_mean(d_true)
    mean_sky = _posterior_mean(bg_sky)
    mean_iso = _posterior_mean(bg_iso)

    resid_sky = abs(mean_sky - mean_true) / _H
    resid_iso = abs(mean_iso - mean_true) / _H

    # POSITIVE: the sky-aware selection reproduces the true-selection shape (<~1%)
    # and H0 (<0.1%).
    assert shape_err_sky < 5e-3, f"sky-aware selection shape error {shape_err_sky:.4f}"
    assert resid_sky < 1e-3, f"sky-aware H0 residual vs truth {resid_sky:.4f}"

    # NEGATIVE control: the isotropic selection is clearly worse on the sky axis --
    # it has discriminating power (else the test could not witness the fix).
    assert shape_err_iso > 3.0 * shape_err_sky, (
        f"isotropic shape error {shape_err_iso:.4f} not clearly worse than "
        f"sky-aware {shape_err_sky:.4f}"
    )
    assert resid_iso > resid_sky, (
        f"isotropic H0 residual {resid_iso:.4f} not larger than sky-aware {resid_sky:.4f}"
    )
    # The sky-selection systematic is bounded (~1%), matching protocol Sec. 7.
    assert resid_iso < 0.02, f"isotropic H0 bias {resid_iso:.4f} outside the bounded band"


# ======================================================================
# T7 -- FRAME ASSERTION (BarycentricTrueEcliptic J2000 pole)
# ======================================================================


def test_T7_ecliptic_pole_is_barycentric_true_ecliptic_j2000() -> None:  # noqa: N802
    """The ecliptic pole used for the sky axis is BarycentricTrueEcliptic(J2000).

    Only the POLE matters under azimuthal symmetry (longitude/equinox offset is
    immaterial to the latitude beta = pi/2 - qS).  The north ecliptic pole
    (lat=+90 deg) maps to ICRS Dec ~ 90 - obliquity ~ 66.56 deg -- the standard
    J2000 ecliptic pole -- confirming the pixel_completeness frame matches the
    fastlisaresponse ecliptic convention.
    """
    import astropy.units as u
    from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord

    from master_thesis_code.galaxy_catalogue import pixel_completeness as pc

    # pixel_completeness builds its map in BarycentricTrueEcliptic(J2000).
    north_pole = SkyCoord(
        lon=0.0 * u.deg,
        lat=90.0 * u.deg,
        frame=BarycentricTrueEcliptic(equinox="J2000"),
    )
    icrs = north_pole.transform_to("icrs")
    # J2000 mean obliquity ~ 23.4393 deg => ecliptic pole Dec ~ 66.5607 deg.
    assert float(icrs.dec.deg) == pytest.approx(66.5607, abs=0.05)

    # The frozen m_th map is the SOLE source of f (audit R4): shape/consistency.
    m_th = np.load(pc.M_TH_CACHE_PATH)
    assert m_th.shape == (12 * pc.NSIDE**2,)


# ======================================================================
# T8 -- Sigma_global uses the SAME flat per-band p_det as D(h)/beta_Gbar
#       (guardrail: p_det(Omega) must be ONE shared object; the interpolated
#       accessor would NOT cancel in beta_G/Sigma_global -> rescales the
#       in-catalogue channel weight and reintroduces the sky bias).
# ======================================================================
@_needs_injections
def test_T8_global_selection_uses_flat_band_pdet_convention(  # noqa: N802
    pdet_sky: SimulationDetectionProbability,
) -> None:
    from master_thesis_code.emri_rate import R_eff_per_mbh
    from master_thesis_code.physical_relations import dist_to_redshift, dist_vectorized

    cat = _make_catalog(seed=7, n=4000, with_sky=True)
    sig = precompute_global_catalog_selection(
        [_H], cast(GalaxyCatalogueHandler, cat), pdet_sky, with_bh_mass=False
    )[_H]

    df = cat.reduced_galaxy_catalog
    z = df[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=float)
    M = df[InternalCatalogColumns.BH_MASS].to_numpy(dtype=float)  # noqa: N806
    theta = df[InternalCatalogColumns.THETA_S].to_numpy(dtype=float)
    z_max = dist_to_redshift(pdet_sky.get_dl_max(_H), h=_H)
    elig = (z < z_max) & np.isfinite(M) & (M > 0.0)
    z, M, theta = z[elig], M[elig], theta[elig]  # noqa: N806
    w = np.asarray(R_eff_per_mbh(M), dtype=float) / (1.0 + z)
    d_L = np.asarray(dist_vectorized(z, h=_H), dtype=float)  # noqa: N806

    # Hand-computed FLAT per-band Sigma_global (same edges + side="right").
    edges = np.asarray(pdet_sky.band_edges_sin_beta(), dtype=float)
    nb = edges.size - 1
    band = np.clip(np.searchsorted(edges, np.abs(np.cos(theta)), side="right") - 1, 0, nb - 1)
    s = np.asarray(pdet_sky.survival_per_band(d_L), dtype=float)  # (nb, N)
    sig_flat = float(np.sum(w * s[band, np.arange(band.size)]))

    # The INTERPOLATED convention (the pre-fix bug) genuinely differs at Nband=6.
    p_interp = np.asarray(
        pdet_sky.detection_probability_without_bh_mass_sky(d_L, np.zeros_like(theta), theta, h=_H),
        dtype=float,
    )
    sig_interp = float(np.sum(w * p_interp))

    assert sig == pytest.approx(sig_flat, rel=1e-12)  # uses flat per-band (matches D/beta_Gbar)
    assert abs(sig - sig_interp) / sig_flat > 1e-4  # and NOT the interpolated convention
