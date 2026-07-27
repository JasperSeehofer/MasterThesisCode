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


def _gaussian_product_mz(
    mu_cond: np.ndarray,
    sigma_cond: float,
    opz: np.ndarray,
    det_M: float,
    host_M: float,
    sigma_lnM: float,
) -> np.ndarray:
    """Analytic Gaussian-product mass marginal (Eq. 14.31), independent reference."""
    mu_gal = host_M * opz / det_M
    sigma_gal = sigma_lnM * mu_gal
    s2 = sigma_cond**2 + sigma_gal**2
    return np.exp(-0.5 * (mu_cond - mu_gal) ** 2 / s2) / np.sqrt(2.0 * np.pi * s2)


def test_mz_integral_spec_mass_limit_recovers_gaussian_product() -> None:
    """RATIFY-M3 crossover, sigma_lnM -> 0 (spec-mass limit, derivation §3.7 case 1):
    the bare GW-centred GH quadrature aliases the now-narrow prior (it returned
    exactly 0 at the floor pre-crossover); with the crossover the kernel falls back
    to the analytic Gaussian product, recovering the current default path's
    sigma_gal -> 0 limit N(mu_cond; mu_gal, sigma_cond) exactly."""
    K = 40
    mu_cond = np.linspace(0.85, 1.15, K)
    opz = 1.0 + np.linspace(0.30, 0.50, K)
    sig0 = bs._MASS_TRUNC_SIGMA_LNM_FLOOR
    Z0 = bs._mass_trunc_log_normalisation(4.0e6, sig0).item()
    assert np.isfinite(Z0) and Z0 > 0.0
    mz = bs._mass_trunc_mz_integral(mu_cond, 0.05, opz, 5.0e6, 4.0e6, sig0, Z0)
    ref = _gaussian_product_mz(mu_cond, 0.05, opz, 5.0e6, 4.0e6, sig0)
    assert np.all(np.isfinite(mz))
    np.testing.assert_allclose(mz, ref, rtol=1e-12)
    # and the limit itself: sigma_gal ~ 1e-6 -> N(mu_cond; mu_gal, sigma_cond)
    mu_gal = 4.0e6 * opz / 5.0e6
    lim = np.exp(-0.5 * (mu_cond - mu_gal) ** 2 / 0.05**2) / np.sqrt(2.0 * np.pi * 0.05**2)
    np.testing.assert_allclose(mz, lim, rtol=1e-6)


def test_mz_integral_crossover_continuity_at_threshold() -> None:
    """RATIFY-M3: mz is continuous (to the O(sigma_lnM) family difference) across
    the crossover threshold sigma_gal = K * sigma_cond — the C0-continuity minimum
    bar. Interior host, so truncation is negligible on both sides."""
    K = 30
    mu_cond = np.linspace(0.95, 1.05, K)
    opz = np.full(K, 1.4)
    det_M, host_M, sigma_cond = 7.0e5, 5.0e5, 0.01
    a_gal = host_M * 1.4 / det_M  # = 1.0
    sig_thr = bs._MASS_TRUNC_GH_CROSSOVER_K * sigma_cond / a_gal
    for eps in (0.999, 1.001):
        sig = sig_thr * eps
        Z = bs._mass_trunc_log_normalisation(host_M, sig).item()
        mz = bs._mass_trunc_mz_integral(mu_cond, sigma_cond, opz, det_M, host_M, sig, Z)
        ref = _gaussian_product_mz(mu_cond, sigma_cond, opz, det_M, host_M, sig)
        # both sides agree with the Gaussian product to the family difference
        np.testing.assert_allclose(mz, ref, rtol=0.15)
    below = bs._mass_trunc_mz_integral(
        mu_cond,
        sigma_cond,
        opz,
        det_M,
        host_M,
        sig_thr * 0.999,
        bs._mass_trunc_log_normalisation(host_M, sig_thr * 0.999).item(),
    )
    above = bs._mass_trunc_mz_integral(
        mu_cond,
        sigma_cond,
        opz,
        det_M,
        host_M,
        sig_thr * 1.001,
        bs._mass_trunc_log_normalisation(host_M, sig_thr * 1.001).item(),
    )
    np.testing.assert_allclose(below, above, rtol=0.15)


def test_mz_integral_operative_regime_untouched_by_crossover() -> None:
    """Catalogue regime (sigma_lnM ~ 0.6 >> K*sigma_cond/a_gal): the crossover must
    NOT fire — mz equals the brute-force quadrature of the truncated LN x R_eff
    prior against the GW Gaussian (implementation-independent reference)."""
    K = 8
    mu_cond = np.linspace(0.9, 1.1, K)
    sigma_cond = 0.01
    det_M, host_M, sigma_lnM = 5.0e6, 4.0e6, 0.6
    opz = 1.0 + np.linspace(0.30, 0.50, K)
    Z_M = bs._mass_trunc_log_normalisation(host_M, sigma_lnM).item()
    mz = bs._mass_trunc_mz_integral(mu_cond, sigma_cond, opz, det_M, host_M, sigma_lnM, Z_M)
    # brute force: int N(a; mu_cond, sigma_cond) p_M(a det_M/(1+z)) det_M/(1+z) da
    for j in range(K):
        a = np.linspace(mu_cond[j] - 8 * sigma_cond, mu_cond[j] + 8 * sigma_cond, 20001)
        M = a * det_M / opz[j]
        gw = np.exp(-0.5 * ((a - mu_cond[j]) / sigma_cond) ** 2) / (
            np.sqrt(2.0 * np.pi) * sigma_cond
        )
        ref_j = np.trapezoid(gw * _pm_density(M, host_M, sigma_lnM, Z_M) * det_M / opz[j], a)
        assert mz[j] == pytest.approx(ref_j, rel=1e-6)


def test_mz_integral_broad_mismatched_host_keeps_fat_tail() -> None:
    """IMPLEMENTATION CORRECTION guard (derivation §3.3): a BROAD prior
    (sigma_lnM ~ 0.7) on a mass-mismatched host (a_gal << 1) must stay on the GH
    path — its fat lognormal tail at the GW peak is the correct physics; the
    Gaussian fallback would return exp(-thousands). The linearized width alone
    (sigma_gal = 0.0037 < K*sigma_cond) would misfire here."""
    K = 10
    mu_cond = np.linspace(0.95, 1.05, K)
    sigma_cond = 0.01
    det_M, host_M, sigma_lnM = 3.0e6, 1.5e4, 0.667  # a_gal ~ 0.0055
    opz = np.full(K, 1.10)
    Z_M = bs._mass_trunc_log_normalisation(host_M, sigma_lnM).item()
    mz = bs._mass_trunc_mz_integral(mu_cond, sigma_cond, opz, det_M, host_M, sigma_lnM, Z_M)
    # reference: sharp-GW limit — the prior tail density at the GW-peak mass
    m_star = mu_cond * det_M / opz
    ref = _pm_density(m_star, host_M, sigma_lnM, Z_M) * det_M / opz
    assert np.all(ref > 0.0)  # the fat tail is nonzero at these ~8 sigma_lnM
    # rtol: GH integrates the tail over the finite sigma_cond window; the
    # reference is the sigma_cond -> 0 pointwise limit (~1% apart here). The
    # Gaussian fallback would be ~40 orders of magnitude low, not ~1%.
    np.testing.assert_allclose(mz, ref, rtol=2e-2)


def test_resolve_host_mass_kernel_auto_preserves_bundling() -> None:
    """'auto' == the historical bundling: trunc_lognormal iff mass_trunc mode."""
    assert bs.resolve_host_mass_kernel("auto", "mass_trunc", "auto") == "trunc_lognormal"
    for mode in ("volume_deconv", "absolute_marginal", "generator_marginal", "global"):
        assert bs.resolve_host_mass_kernel("auto", mode, "auto") == "gaussian"


def test_resolve_host_mass_kernel_explicit_override() -> None:
    """The ratified real-data combination is expressible: absolute_marginal
    normalization x volume_deconv z-kernel x trunc_lognormal mass kernel."""
    assert (
        bs.resolve_host_mass_kernel("trunc_lognormal", "absolute_marginal", "auto")
        == "trunc_lognormal"
    )
    assert bs.resolve_host_mass_kernel("gaussian", "mass_trunc", "auto") == "gaussian"


def test_resolve_host_mass_kernel_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown host_mass_kernel"):
        bs.resolve_host_mass_kernel("lognormal", "mass_trunc", "auto")


def test_resolve_host_mass_kernel_point_z_guard() -> None:
    """Prior-consistency guard (derivation §3.3): a point-resolving host-z numerator
    with the trunc_lognormal mass kernel would give N_g and D_g DIFFERENT mass
    priors (counted-once-in-M violation) — must raise, in every route to it."""
    # explicit point z with the bundled mass_trunc kernel
    with pytest.raises(ValueError, match="prior-inconsistent"):
        bs.resolve_host_mass_kernel("auto", "mass_trunc", "point")
    # generator_marginal auto-resolves z to point
    with pytest.raises(ValueError, match="prior-inconsistent"):
        bs.resolve_host_mass_kernel("trunc_lognormal", "generator_marginal", "auto")
    # explicit point z with an explicit trunc mass kernel
    with pytest.raises(ValueError, match="prior-inconsistent"):
        bs.resolve_host_mass_kernel("trunc_lognormal", "absolute_marginal", "point")
    # the gaussian mass kernel composes freely with point (production default)
    assert bs.resolve_host_mass_kernel("auto", "generator_marginal", "auto") == "gaussian"


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
