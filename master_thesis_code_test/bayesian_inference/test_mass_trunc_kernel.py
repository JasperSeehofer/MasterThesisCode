"""Limiting-case + physics gates for the mass_trunc host-mass kernel (EXP-45).

The ``mass_trunc`` normalization mode replaces the linear-Gaussian G2d moment
match in the 2D (with-BH-mass) channel with the truncated lognormal x R_eff
host-mass prior on ``[M_MIN, M_MAX]`` (module helpers ``_mass_trunc_*`` in
``bayesian_statistics``). These tests pin the analytic limits the
``/physics-change`` protocol requires, at the pure-helper level (the full-pipeline
scalar==batch bit-parity and golden regression live in ``test_kernel_parity`` /
``test_kernel_batch_equivalence``).

References:
    Reines & Volonteri (2015), arXiv:1508.06274 (lognormal mass error);
    Babak et al. (2017), arXiv:1703.09722 (R_eff population weight);
    results/mass_kernel_truncation_20260713/FINDINGS.md (motivation).
"""

import numpy as np
import pytest

import master_thesis_code.bayesian_inference.bayesian_statistics as bs
from master_thesis_code.datamodels.parameter_space import ParameterSpace
from master_thesis_code.emri_rate import R_eff_per_mbh


def _pm_density(M: np.ndarray, host_M: float, sigma_lnM: float, Z_M: float) -> np.ndarray:
    """Normalised prior density in M (0 outside the window), for reference checks."""
    inside = (M >= bs._MASS_TRUNC_M_MIN) & (M <= bs._MASS_TRUNC_M_MAX)
    w = bs._mass_trunc_lnM_weight(np.where(inside, M, bs._MASS_TRUNC_M_MIN), host_M, sigma_lnM)
    return np.where(inside, w / (M * Z_M), 0.0)


def test_mass_window_matches_parameter_space_bounds() -> None:
    """Drift guard: the truncation window is the EMRI ParameterSpace.M bound."""
    ps = ParameterSpace()
    assert bs._MASS_TRUNC_M_MIN == pytest.approx(ps.M.lower_limit)
    assert bs._MASS_TRUNC_M_MAX == pytest.approx(ps.M.upper_limit)


def test_sigma_lnM_recovers_from_linear_error() -> None:
    """sigma_lnM = host_M_error / host_M (invert handler's linearisation)."""
    host_M = np.array([3.0e5, 4.5e6])
    host_M_error = np.array([0.6 * 3.0e5, 0.5 * 4.5e6])
    got = bs._mass_trunc_sigma_lnM(host_M, host_M_error)
    np.testing.assert_allclose(got, [0.6, 0.5])
    # invalid error -> floored (spec-mass limit), never negative / nan
    assert bs._mass_trunc_sigma_lnM(3e5, 0.0) == bs._MASS_TRUNC_SIGMA_LNM_FLOOR


@pytest.mark.parametrize("host_M", [1.5e4, 3.0e5, 4.5e6, 7.0e6])
@pytest.mark.parametrize("sigma_lnM", [0.05, 0.3, 0.6, 0.8])
def test_prior_normalises_to_unity(host_M: float, sigma_lnM: float) -> None:
    """int_{M_MIN}^{M_MAX} p_M(M) dM == 1 (independent fine-grid quadrature)."""
    Z_M = bs._mass_trunc_log_normalisation(host_M, sigma_lnM).item()
    u = np.linspace(np.log(bs._MASS_TRUNC_M_MIN), np.log(bs._MASS_TRUNC_M_MAX), 400001)
    M = np.exp(u)
    integral = np.trapezoid(_pm_density(M, host_M, sigma_lnM, Z_M) * M, u)  # dM = M d lnM
    assert integral == pytest.approx(1.0, rel=2e-4)


def test_prior_is_zero_outside_window() -> None:
    """Truncation: the prior density vanishes below M_MIN and above M_MAX."""
    host_M, sigma_lnM = 3.0e5, 0.6
    Z_M = bs._mass_trunc_log_normalisation(host_M, sigma_lnM).item()
    outside = np.array([bs._MASS_TRUNC_M_MIN * 0.5, bs._MASS_TRUNC_M_MAX * 2.0])
    assert np.all(_pm_density(outside, host_M, sigma_lnM, Z_M) == 0.0)


def test_normalisation_scale_invariant_to_reff(monkeypatch: pytest.MonkeyPatch) -> None:
    """A global rescale of R_eff cancels in p_M (only relative weighting matters)."""
    host_M, sigma_lnM = 1.0e6, 0.5
    Z_M = bs._mass_trunc_log_normalisation(host_M, sigma_lnM).item()
    M = np.array([2e5, 1e6, 5e6])
    p1 = _pm_density(M, host_M, sigma_lnM, Z_M)

    # scale R_eff by a constant -> both weight and Z_M scale identically -> p_M unchanged
    def scaled_R_eff(mass: float | np.ndarray) -> np.ndarray:
        return 7.3 * np.asarray(R_eff_per_mbh(mass), dtype=np.float64)

    monkeypatch.setattr(bs, "R_eff_per_mbh", scaled_R_eff)
    Z2 = bs._mass_trunc_log_normalisation(host_M, sigma_lnM).item()
    p2 = _pm_density(M, host_M, sigma_lnM, Z2)
    np.testing.assert_allclose(p1, p2, rtol=1e-12)


def test_mz_integral_sharp_gw_limit() -> None:
    """Design regime (sharp GW M_z, broad prior): the mass marginal collapses onto
    the prior sampled at the GW-peak mass,
    ``mz(z) -> p_M(M*(z)) * det_M/(1+z)``,  ``M*(z) = mu_cond * det_M/(1+z)``.
    This is the limit the Gauss-Hermite-on-the-GW-peak quadrature is built for --
    real EMRI redshifted-mass errors are far below the ~0.6 dex catalogue prior."""
    K = 40
    mu_cond = np.linspace(0.85, 1.15, K)
    sigma_cond = 1.0e-3  # GW MUCH sharper than the sigma_lnM = 0.6 prior
    det_M, host_M, sigma_lnM = 5.0e6, 4.0e6, 0.6
    opz = 1.0 + np.linspace(0.30, 0.50, K)
    Z_M = bs._mass_trunc_log_normalisation(host_M, sigma_lnM).item()
    mz = bs._mass_trunc_mz_integral(mu_cond, sigma_cond, opz, det_M, host_M, sigma_lnM, Z_M)
    m_star = mu_cond * det_M / opz  # GW-peak rest-frame mass at each z
    ref = _pm_density(m_star, host_M, sigma_lnM, Z_M) * det_M / opz  # p_M(M*) |dM/da|
    assert np.all(np.isfinite(mz))
    np.testing.assert_allclose(mz, ref, rtol=1e-4, atol=1e-8 * ref.max())


def test_mz_integral_spec_mass_robustness() -> None:
    """sigma_lnM -> 0 (delta prior) is OUTSIDE the GH design regime, but must stay
    finite and non-negative -- no NaN from the peak-aware Z_M (the guard against the
    volume_trunc-style aliasing blow-up)."""
    K = 40
    mu_cond = np.linspace(0.85, 1.15, K)
    opz = 1.0 + np.linspace(0.30, 0.50, K)
    sig0 = bs._MASS_TRUNC_SIGMA_LNM_FLOOR
    Z0 = bs._mass_trunc_log_normalisation(4.0e6, sig0).item()
    assert np.isfinite(Z0) and Z0 > 0.0
    mz = bs._mass_trunc_mz_integral(mu_cond, 0.05, opz, 5.0e6, 4.0e6, sig0, Z0)
    assert np.all(np.isfinite(mz))
    assert np.all(mz >= 0.0)


def test_mz_integral_zero_when_gw_mass_outside_window() -> None:
    """If the GW peak mass is far outside [M_MIN, M_MAX], the truncated prior gives
    a vanishing mass marginal (the untruncated Gaussian would leak probability)."""
    K = 20
    det_M, host_M, sigma_lnM = 5.0e6, 3.0e5, 0.6
    Z_M = bs._mass_trunc_log_normalisation(host_M, sigma_lnM).item()
    opz = 1.0 + np.linspace(0.30, 0.50, K)
    # mu_cond such that a*det_M/(1+z) >> M_MAX for all nodes -> M well above the window
    mu_cond = np.full(K, 100.0)
    mz = bs._mass_trunc_mz_integral(mu_cond, 0.02, opz, det_M, host_M, sigma_lnM, Z_M)
    assert np.all(mz >= 0.0)
    assert np.all(mz < 1e-30)


def test_mz_integral_scalar_batch_bit_identical() -> None:
    """The core mass marginal is bit-identical for a scalar host and its batch row
    (the guarantee that lets scalar/batch pipeline entry points agree)."""
    K = 50
    mu_cond = np.linspace(0.8, 1.2, K)
    opz = 1.0 + np.linspace(0.30, 0.50, K)
    det_M, sigma_cond = 5.0e6, 0.03
    host_M = np.array([3.0e5, 4.5e6, 2.0e4])
    sig = bs._mass_trunc_sigma_lnM(host_M, np.array([0.6, 0.5, 0.7]) * host_M)
    Z = bs._mass_trunc_log_normalisation(host_M, sig)
    scalar = bs._mass_trunc_mz_integral(
        mu_cond, sigma_cond, opz, det_M, float(host_M[0]), float(sig[0]), float(Z[0])
    )
    batch = bs._mass_trunc_mz_integral(
        np.broadcast_to(mu_cond, (3, K)).copy(),
        sigma_cond,
        np.broadcast_to(opz, (3, K)).copy(),
        det_M,
        host_M,
        sig,
        Z,
    )
    assert np.array_equal(scalar, batch[0])
