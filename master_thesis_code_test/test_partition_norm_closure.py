"""End-to-end closure test for the partition-norm likelihood (Task A).

A self-consistent mock pipeline: generate a galaxy population at a known H0, split
it into a catalog (kept with probability f(z)) and missing/dark galaxies, inject
rate-weighted EMRI events (in-catalog + dark, per Change 4b), give each a Gaussian
luminosity-distance measurement, apply a smooth detection probability, then run the
partition-norm inference over an H0 grid and assert the MAP recovers the injected H0.

Drives the REAL precompute functions Task A added
(``precompute_completion_denominator``, ``precompute_missing_completion_denominator``,
``precompute_global_catalog_selection``) and assembles the single Gray ratio
``p_i = (beta_G * L_cat + B_num) / D(h)``. Sky is a matched delta (not what Task A
changed), isolating the d_L/H0 dependence that drives the MAP. The galaxy-z PDF is
taken in the narrow limit (``N_g ~= p_GW(z_g)``, ``D_g ~= p_det(z_g)``), matching
``precompute_global_catalog_selection``.

These checks confirm the catalog/completion BALANCE is unbiased; the per-event
assembly (single ratio, limits) is covered by test_partition_norm_restructure.py.
Marked ``slow`` (a few seconds per realization). Reusable for Change 5: sweep
``f`` shapes / add an Omega dependence.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy.integrate import fixed_quad
from scipy.stats import norm

from master_thesis_code.bayesian_inference.bayesian_statistics import (
    precompute_completion_denominator,
    precompute_global_catalog_selection,
    precompute_missing_completion_denominator,
)
from master_thesis_code.emri_rate import R_eff_per_mbh, mbh_mass_function
from master_thesis_code.galaxy_catalogue.handler import InternalCatalogColumns
from master_thesis_code.physical_relations import (
    comoving_volume_element,
    dist,
    dist_to_redshift,
    dist_vectorized,
)

_Z_MIN = 1e-3
_Z_MAX = 0.5
_M_MIN = 1.0e4
_M_MAX = 1.0e7
_D_HOR_GPC = 1.2  # detection horizon (Gpc)
_H_MIN, _H_MAX = 0.60, 0.80

FloatArr = npt.NDArray[np.float64]
CompFunc = Callable[[FloatArr], FloatArr]


def _p_det_of_dl(d_L: FloatArr) -> FloatArr:
    """Smooth detection probability vs luminosity distance (Gpc), -> 0 past horizon."""
    d = np.asarray(d_L, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp((d - 0.7 * _D_HOR_GPC) / (0.08 * _D_HOR_GPC)))
    return np.asarray(np.where(d > _D_HOR_GPC, 0.0, p), dtype=np.float64)


class _MockPdet:
    """Sky-independent detection-probability stub (3D + 4D accessors)."""

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


class _ZCompleteness:
    """Completeness f(z) (H0-independent), shared by injection and inference."""

    def __init__(self, f_func: CompFunc) -> None:
        self._f = f_func

    def get_completeness_at_redshift(
        self, z: float | FloatArr, h: float = 0.0, **kw: object
    ) -> FloatArr:
        return np.clip(self._f(np.asarray(z, dtype=np.float64)), 0.0, 1.0)

    def f_bar(self, z: float | FloatArr, h: float = 0.0) -> FloatArr:
        # Omega-independent stub: the sky-average equals f(z) (Change 5.2/5.4).
        return self.get_completeness_at_redshift(z, h)

    def f_k(self, z: float | FloatArr, k: int, h: float = 0.0) -> FloatArr:
        # Omega-independent stub: every pixel shares the same f(z) (Change 5.3).
        return self.get_completeness_at_redshift(z, h)


class _ClosureCatalog:
    """Minimal catalog handler exposing ``reduced_galaxy_catalog`` (z, M columns)."""

    def __init__(self, z: FloatArr, M: FloatArr) -> None:
        self.reduced_galaxy_catalog = pd.DataFrame(
            {InternalCatalogColumns.REDSHIFT: z, InternalCatalogColumns.BH_MASS: M}
        )


def _sample_population(
    rng: np.random.Generator, n_gal: int, h_true: float
) -> tuple[FloatArr, FloatArr]:
    """Full population: z ~ (1/(1+z)) dVc/dz, log10 M ~ phi_MBH(M) * R_eff(M)."""
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
    return z, M


def _inject_events(
    rng: np.random.Generator,
    z_all: FloatArr,
    M_all: FloatArr,
    h_true: float,
    n_target: int,
    sigma_frac: float,
) -> list[dict[str, float]]:
    """Rate-weighted hosts -> consistent Gaussian d_L measurement -> detection filter."""
    w = np.asarray(R_eff_per_mbh(M_all), dtype=np.float64) / (1.0 + z_all)
    p = w / w.sum()
    events: list[dict[str, float]] = []
    tries = 0
    while len(events) < n_target and tries < 400 * n_target:
        tries += 1
        g = int(rng.choice(len(z_all), p=p))
        z_host = float(z_all[g])
        d_true = float(dist(z_host, h=h_true))  # Gpc
        # Reported sigma fixed per event; d_meas ~ N(d_true, sigma) -- the inference
        # uses the SAME sigma, so no sigma-mismatch (Eddington) bias by construction.
        sigma_dL = sigma_frac * d_true
        d_meas = d_true + sigma_dL * rng.standard_normal()
        if d_meas <= 0:
            continue
        if rng.uniform() < float(_p_det_of_dl(np.asarray([d_meas]))[0]):
            events.append({"d_meas": d_meas, "sigma_dL": sigma_dL})
    return events


def _event_log_likelihood(
    event: dict[str, float],
    catalog_z: FloatArr,
    catalog_M: FloatArr,
    completeness: _ZCompleteness,
    h: float,
    D_h: float,
    beta_G: float,
    global_denom: float,
) -> float:
    """log p_i(h) = log[(beta_G * L_cat + B_num) / D(h)] for one event (no-BH channel)."""
    d_meas = event["d_meas"]
    sigma_dL = event["sigma_dL"]

    def p_gw_of_z(z: FloatArr) -> FloatArr:
        d_model = np.asarray(dist_vectorized(z, h=h), dtype=np.float64)  # Gpc
        return np.asarray(norm.pdf(d_model, loc=d_meas, scale=sigma_dL), dtype=np.float64)

    # L_cat = sum_g w_g N_g / global_denom over candidate catalog galaxies. z(d,h)
    # increases with BOTH d and h, so the widest 5-sigma window over the H0 grid is
    # [z(d-5s, h_min), z(d+5s, h_max)]. N_g ~= p_GW(z_g) (narrow galaxy-z-PDF limit).
    z_lo = dist_to_redshift(max(d_meas - 5.0 * sigma_dL, 1e-4), h=_H_MIN)
    z_hi = dist_to_redshift(d_meas + 5.0 * sigma_dL, h=_H_MAX)
    cand = (catalog_z >= z_lo) & (catalog_z <= z_hi)
    cat_num_sum = 0.0
    if np.any(cand):
        zc = catalog_z[cand]
        wc = np.asarray(R_eff_per_mbh(catalog_M[cand]), dtype=np.float64) / (1.0 + zc)
        cat_num_sum = float(np.sum(wc * p_gw_of_z(zc)))
    L_cat = cat_num_sum / global_denom if global_denom > 0 else 0.0

    # B_num = INTEGRAL (1-f(z)) p_GW(z) (1/(1+z)) dVc/dz over the event 4-sigma window.
    bz_lo = max(dist_to_redshift(max(d_meas - 4.0 * sigma_dL, 1e-4), h=h), 1e-6)
    bz_hi = dist_to_redshift(d_meas + 4.0 * sigma_dL, h=h)

    def b_integrand(z: FloatArr) -> FloatArr:
        f_z = completeness.get_completeness_at_redshift(z, h)
        dVc = np.asarray(comoving_volume_element(z, h=h), dtype=np.float64)
        return (1.0 - f_z) * p_gw_of_z(z) * dVc / (1.0 + z)

    B_num = float(fixed_quad(b_integrand, bz_lo, bz_hi, n=50)[0])

    p_i = (beta_G * L_cat + B_num) / D_h if D_h > 0 else 0.0
    return float(np.log(p_i)) if p_i > 0 else -1e30


def _f_one(z: FloatArr) -> FloatArr:
    """Constant full completeness f(z) = 1 (pure catalog, no dark events)."""
    return np.ones_like(z)


def _f_declining(z: FloatArr) -> FloatArr:
    """Realistic GLADE-like completeness: ~1 nearby, declining to 0.05 by z~0.33."""
    return np.clip(1.0 - z / 0.35, 0.05, 1.0)


def _map_bias(
    *,
    h_true: float,
    f_func: CompFunc,
    sigma_frac: float = 0.015,
    n_gal: int = 10000,
    n_events: int = 300,
    seed: int = 0,
) -> float:
    """Run one closure realization; return MAP(H0) - h_true (parabola-refined).

    A realization whose MAP lands at a grid edge is under-constrained (a finite-N
    variance artifact, not bias); it returns the edge offset so the median over
    seeds absorbs it.
    """
    rng = np.random.default_rng(seed)
    completeness = _ZCompleteness(f_func)
    z_all, M_all = _sample_population(rng, n_gal, h_true)
    f_at_z = completeness.get_completeness_at_redshift(z_all)
    in_cat = rng.uniform(size=n_gal) < f_at_z
    catalog_z = z_all[in_cat]
    catalog_M = M_all[in_cat]
    catalog = _ClosureCatalog(catalog_z, catalog_M)
    events = _inject_events(rng, z_all, M_all, h_true, n_events, sigma_frac)

    pdet = _MockPdet()
    hs = [float(h) for h in np.round(np.arange(_H_MIN, _H_MAX + 1e-9, 0.02), 4)]
    D_tab = precompute_completion_denominator(hs, pdet, Omega_m=0.25, Omega_DE=0.75)  # type: ignore[arg-type]
    bGbar_tab = precompute_missing_completion_denominator(hs, pdet, completeness=completeness)  # type: ignore[arg-type]
    gd_tab = precompute_global_catalog_selection(hs, catalog, pdet, with_bh_mass=False)  # type: ignore[arg-type]

    logpost = np.zeros(len(hs))
    for i, h in enumerate(hs):
        beta_G = D_tab[h] - bGbar_tab[h]
        logpost[i] = sum(
            _event_log_likelihood(
                ev, catalog_z, catalog_M, completeness, h, D_tab[h], beta_G, gd_tab[h]
            )
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


def _median_bias(f_func: CompFunc, h_true: float, n_seeds: int = 7) -> float:
    """Median MAP bias over ``n_seeds`` realizations.

    The MEDIAN (not mean) is the unbiasedness statistic: occasional realizations
    with the mock's small N land the MAP near a grid edge (high variance, not a
    systematic), which the median is robust to. See the docstring of this module
    and the closure findings in .planning/derivation-partition-norm/.
    """
    return float(
        np.median([_map_bias(h_true=h_true, f_func=f_func, seed=s) for s in range(n_seeds)])
    )


@pytest.mark.slow
def test_closure_pure_catalog_recovers_h0() -> None:
    """f=1 (no completion): the global-denominator L_cat recovers the injected H0."""
    bias = _median_bias(_f_one, h_true=0.70)
    assert abs(bias) < 0.008, f"pure-catalog median MAP bias {bias:+.4f} exceeds 0.008"


@pytest.mark.slow
def test_closure_realistic_completeness_recovers_h0() -> None:
    """Realistic declining f(z) (GLADE-like): the mixture recovers the injected H0."""
    bias = _median_bias(_f_declining, h_true=0.70)
    assert abs(bias) < 0.008, f"realistic-completeness median MAP bias {bias:+.4f} exceeds 0.008"


@pytest.mark.slow
def test_closure_recovers_nondefault_h0() -> None:
    """Recovers a DIFFERENT injected H0 (0.65), proving no pull toward a default."""
    bias = _median_bias(_f_declining, h_true=0.65)
    assert abs(bias) < 0.008, f"H0=0.65 median MAP bias {bias:+.4f} exceeds 0.008"
