"""G2d: Eddington-in-M host-mass rate prior (exact moment-matched form).

The volume_deconv/volume_global kernels replace the bare host-mass Gaussian
N(M; M_g, sigma_M) with the rate-weighted prior N * R_eff / Z_M, implemented
as the exact posterior-mean shift (moment-matched Gaussian, analytic marginal
preserved). NB: R_eff is NOT monotone — the kappa_cap low-mass roll-off makes
the shift POSITIVE near 1e5 Msun. docs/derivations/G2d_host_mass_rate_prior.md.
"""

import numpy as np
import pytest
from scipy.stats import norm

from master_thesis_code.bayesian_inference.bayesian_statistics import (
    eddington_shifted_host_mass,
)
from master_thesis_code.emri_rate import R_eff_per_mbh


def _exact_posterior_mean(M_g: float, sigma: float) -> float:
    """Independent quadrature of the mean of N(M; M_g, sigma^2) * R_eff(M) / Z_M."""
    lo, hi = max(M_g - 6 * sigma, 1e3), M_g + 6 * sigma
    M = np.linspace(lo, hi, 20001)
    w = norm.pdf(M, loc=M_g, scale=sigma) * np.asarray(R_eff_per_mbh(M), dtype=np.float64)
    return float(np.trapezoid(M * w, M) / np.trapezoid(w, M))


def test_shift_vanishes_as_sigma_to_zero() -> None:
    M_g = 3e5
    assert eddington_shifted_host_mass(M_g, 0.0) == pytest.approx(M_g)
    assert eddington_shifted_host_mass(M_g, 1e-4 * M_g) == pytest.approx(M_g, rel=1e-6)


def test_shift_follows_local_rate_slope_sign() -> None:
    """Falling R_eff (high M) pulls the mass DOWN; the kappa_cap roll-off
    (rising R_eff at low M) pulls it UP — the exact form captures both."""
    assert eddington_shifted_host_mass(8e5, 0.4 * 8e5) < 8e5
    assert eddington_shifted_host_mass(1e5, 0.55 * 1e5) > 1e5


@pytest.mark.parametrize("sigma_rel", [0.2, 0.55, 0.76, 1.0])
def test_matches_independent_exact_quadrature(sigma_rel: float) -> None:
    """Helper mean == independent fine-grid quadrature of N*R_eff/Z (<0.5% of M_g)."""
    for M_g in (1e5, 3e5, 8e5):
        exact = _exact_posterior_mean(M_g, sigma_rel * M_g)
        got = eddington_shifted_host_mass(M_g, sigma_rel * M_g)
        assert abs(got - exact) / M_g < 5e-3
