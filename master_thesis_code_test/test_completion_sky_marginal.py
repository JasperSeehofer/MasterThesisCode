"""Regression test for the completion-term B_num sky-normalisation de-rail fix.

[PHYSICS] The completion numerator marginalises the GW likelihood over the
UNKNOWN dark-host sky direction with an isotropic 1/(4pi) prior (a sky-averaged
probability), NOT the peak sky density. Eq. (32) in Gray et al. (2020),
arXiv:1908.06050.

This documents the numerical change: the new isotropic sky-marginal is
~2/(sigma_phi * sigma_theta) times SMALLER than the OLD peak-density evaluation
(~1600x at sigma_sky ~ 2 deg), bringing the completion term B_num from ~1000-5000x
domination down to ~parity with the in-catalogue term. That is the mechanism by
which the fix de-rails the H0 posterior while retaining the completeness correction.
"""

import numpy as np
from scipy.stats import multivariate_normal, norm


def _old_peak_sky_factor(mean: np.ndarray, cov: np.ndarray) -> float:
    """OLD behaviour: 3D GW density evaluated at the event sky-peak (x == mean)."""
    return float(multivariate_normal(mean=mean, cov=cov).pdf(mean))


def _new_marginal_sky_factor(mean: np.ndarray, cov: np.ndarray) -> float:
    """NEW behaviour: isotropic sky-marginal = (sin θ_det/4π) · N(d_L_frac; mean[2], √cov[2,2]).

    The sin(θ_det) is the solid-angle Jacobian: the Fisher Gaussian is a density in
    the bare (φ_S, q_S) coordinates, so the dΩ = sinθ dθ dφ marginal picks up sinθ
    at the narrow beam position (G2a derivation note, Eq. 10).
    """
    sigma_dLfrac = float(np.sqrt(cov[2, 2]))
    return float(
        norm.pdf(mean[2], loc=mean[2], scale=sigma_dLfrac) * np.sin(mean[1]) / (4.0 * np.pi)
    )


def test_completion_sky_marginal_reduces_magnitude() -> None:
    # Representative EMRI event: sigma_sky ~ 2 deg (0.035 rad), sigma_dL/dL ~ 3.7%.
    sigma_sky = 2.0 / 180.0 * np.pi
    sigma_dLfrac = 0.037
    mean = np.array([0.5, 1.2, 1.0])  # [phi, theta, d_L_fraction]; d_L_frac mean = 1
    cov = np.diag([sigma_sky**2, sigma_sky**2, sigma_dLfrac**2])

    old = _old_peak_sky_factor(mean, cov)
    new = _new_marginal_sky_factor(mean, cov)

    assert old > 0.0 and new > 0.0  # sign preserved
    assert new < old  # the fix reduces the completion magnitude
    ratio = old / new
    # Analytic identity for a factorised Gaussian:
    # old/new = 2 / (sin(theta_det) * sigma_phi * sigma_theta).
    expected_ratio = 2.0 / (np.sin(mean[1]) * sigma_sky * sigma_sky)
    assert np.isclose(ratio, expected_ratio, rtol=0.05)
    assert ratio > 1000.0  # ~1640x at sigma_sky = 2 deg -> completion WAS dominating


def test_sky_marginal_carries_solid_angle_jacobian() -> None:
    # [PHYSICS] G2a fix: the isotropic marginal scales as sin(theta_det) — maximal
    # for equatorial events, suppressed toward the poles (the coordinate-space
    # (phi, q) density concentrates area near the poles; the physical per-solid-angle
    # prior does not).
    sigma_dLfrac = 0.037
    cov = np.diag([0.035**2, 0.035**2, sigma_dLfrac**2])
    equatorial = _new_marginal_sky_factor(np.array([0.5, np.pi / 2, 1.0]), cov)
    midlat = _new_marginal_sky_factor(np.array([0.5, np.pi / 6, 1.0]), cov)
    assert np.isclose(midlat / equatorial, np.sin(np.pi / 6), rtol=1e-9)
    # theta = pi/2 reproduces the pre-Jacobian 1/(4pi) normalisation exactly.
    bare = float(norm.pdf(1.0, loc=1.0, scale=sigma_dLfrac) / (4.0 * np.pi))
    assert np.isclose(equatorial, bare, rtol=1e-12)


def test_completion_vanishes_for_complete_catalogue() -> None:
    # Limiting case: f -> 1 (complete catalogue) -> (1 - f) -> 0 -> B_num -> 0,
    # unchanged by the sky-normalisation fix.
    f = 1.0
    sigma_dLfrac = 0.037
    p_gw = norm.pdf(1.0, loc=1.0, scale=sigma_dLfrac) / (4.0 * np.pi)
    dVc = 1.0
    z = 0.1
    integrand = (1.0 - f) * p_gw * dVc / (1.0 + z)
    assert integrand == 0.0


def test_sky_marginal_is_finite_as_localisation_sharpens() -> None:
    # sigma_sky -> 0: the OLD peak density diverges (~1/sigma_sky^2); the NEW isotropic
    # marginal stays FINITE (a perfectly-localised GW still has only a 1/(4pi) prior
    # chance of pointing at a given dark-host direction).
    sigma_dLfrac = 0.037
    new_values = []
    for sigma_sky in (0.05, 0.02, 0.005, 0.001):
        cov = np.diag([sigma_sky**2, sigma_sky**2, sigma_dLfrac**2])
        mean = np.array([0.5, 1.2, 1.0])
        new_values.append(_new_marginal_sky_factor(mean, cov))
        old = _old_peak_sky_factor(mean, cov)
        assert old > _new_marginal_sky_factor(mean, cov)
    # NEW is independent of sigma_sky (finite, constant); OLD blows up.
    assert np.allclose(new_values, new_values[0], rtol=1e-9)
