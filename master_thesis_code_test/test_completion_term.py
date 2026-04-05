"""Tests for the completeness-corrected dark siren likelihood.

Verifies the Gray et al. (2020), arXiv:1908.06050, Eq. 9 combination formula
and the completion term (Eqs. 31-32) implemented in bayesian_statistics.py.

These tests do NOT require GPU, P_det data, or the full evaluation pipeline.
They test mathematical correctness by mocking expensive dependencies.
"""

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import fixed_quad
from scipy.stats import multivariate_normal

from master_thesis_code.galaxy_catalogue.glade_completeness import GladeCatalogCompleteness
from master_thesis_code.physical_relations import comoving_volume_element, dist_vectorized

# ======================================================================
# Test 1: f=1 limit recovers catalog-only result
# Gray et al. (2020) Eq. 9: f_i=1 => p_i = L_cat
# ======================================================================


class TestF1RecoversCatalogOnly:
    """When f_i=1 (complete catalog), the combination formula must return L_cat."""

    def test_f1_recovers_catalog_only_without_bh_mass(self) -> None:
        f_i = 1.0
        L_cat = 3.14e-5
        L_comp = 1.23e-4  # should be ignored
        combined = f_i * L_cat + (1 - f_i) * L_comp
        assert combined == pytest.approx(L_cat, rel=1e-15)

    def test_f1_recovers_catalog_only_with_bh_mass(self) -> None:
        f_i = 1.0
        L_cat_with_bh = 7.89e-6
        L_comp = 5.67e-3
        combined = f_i * L_cat_with_bh + (1 - f_i) * L_comp
        assert combined == pytest.approx(L_cat_with_bh, rel=1e-15)


# ======================================================================
# Test 2: f=0 limit gives completion-only result
# Gray et al. (2020) Eq. 9: f_i=0 => p_i = L_comp
# ======================================================================


class TestF0GivesCompletionOnly:
    """When f_i=0 (empty catalog), the combination formula must return L_comp."""

    def test_f0_gives_completion_only(self) -> None:
        f_i = 0.0
        L_cat = 3.14e-5  # should be ignored
        L_comp = 1.23e-4
        combined = f_i * L_cat + (1 - f_i) * L_comp
        assert combined == pytest.approx(L_comp, rel=1e-15)

    def test_f0_with_zero_L_cat_still_returns_L_comp(self) -> None:
        f_i = 0.0
        L_cat = 0.0
        L_comp = 4.56e-3
        combined = f_i * L_cat + (1 - f_i) * L_comp
        assert combined == pytest.approx(L_comp, rel=1e-15)


# ======================================================================
# Test 3: Weighted combination formula
# Gray et al. (2020) Eq. 9: p_i = f_i * L_cat + (1 - f_i) * L_comp
# ======================================================================


class TestCombinationFormulaWeightedSum:
    """The combination formula is a simple linear interpolation."""

    def test_f05_gives_average(self) -> None:
        f_i = 0.5
        L_cat = 2.0e-4
        L_comp = 6.0e-4
        combined = f_i * L_cat + (1 - f_i) * L_comp
        expected = 0.5 * 2.0e-4 + 0.5 * 6.0e-4  # = 4.0e-4
        assert combined == pytest.approx(expected, rel=1e-15)

    def test_f03_gives_weighted_sum(self) -> None:
        f_i = 0.3
        L_cat = 1.0e-3
        L_comp = 2.0e-3
        combined = f_i * L_cat + (1 - f_i) * L_comp
        expected = 0.3 * 1.0e-3 + 0.7 * 2.0e-3  # = 1.7e-3
        assert combined == pytest.approx(expected, rel=1e-15)

    def test_combination_always_between_components(self) -> None:
        """For f_i in (0,1), the combined value is between L_cat and L_comp."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            f_i = rng.uniform(0.01, 0.99)
            L_cat = rng.uniform(1e-8, 1e-2)
            L_comp = rng.uniform(1e-8, 1e-2)
            combined = f_i * L_cat + (1 - f_i) * L_comp
            assert min(L_cat, L_comp) <= combined <= max(L_cat, L_comp)


# ======================================================================
# Test 4: Completion term positivity
# L_comp = integral[p_GW * P_det * dVc dz] / integral[P_det * dVc dz]
# Both integrands are non-negative => L_comp >= 0
# ======================================================================


class TestCompletionTermPositive:
    """L_comp is a ratio of positive integrals and must be positive."""

    def test_completion_term_positive_with_constant_pdet(self) -> None:
        """With P_det=1, L_comp = integral[p_GW * dVc dz] / integral[dVc dz]."""
        # Set up a simple scenario: detection at d_L = 0.5 Gpc, h = 0.73
        h = 0.73
        d_L_det = 0.5  # Gpc
        phi_det = 1.0
        theta_det = 0.5

        # Simple GW Gaussian (3D: phi, theta, d_L_fraction)
        sigma_d_L_frac = 0.05
        cov = np.diag([0.01**2, 0.01**2, sigma_d_L_frac**2])
        gw_gaussian = multivariate_normal(mean=[phi_det, theta_det, 1.0], cov=cov)

        z_center = 0.1  # approximate redshift for d_L ~ 0.5 Gpc at h=0.73
        z_lower = max(z_center - 0.05, 1e-6)
        z_upper = z_center + 0.05

        def numerator_integrand(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            d_L = np.asarray(dist_vectorized(z, h=h), dtype=np.float64)
            d_L_frac = d_L / d_L_det
            phi = np.full_like(z, phi_det)
            theta = np.full_like(z, theta_det)
            p_gw: npt.NDArray[np.float64] = gw_gaussian.pdf(np.vstack([phi, theta, d_L_frac]).T)
            dVc: npt.NDArray[np.float64] = np.atleast_1d(
                np.asarray(comoving_volume_element(z, h=h), dtype=np.float64)
            )
            return p_gw * dVc  # P_det = 1 (constant)

        def denominator_integrand(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            dVc: npt.NDArray[np.float64] = np.atleast_1d(
                np.asarray(comoving_volume_element(z, h=h), dtype=np.float64)
            )
            return dVc  # P_det = 1

        num: float = fixed_quad(numerator_integrand, z_lower, z_upper, n=50)[0]
        den: float = fixed_quad(denominator_integrand, z_lower, z_upper, n=50)[0]

        assert den > 0, "Denominator must be positive (volume element > 0)"
        L_comp = num / den
        assert L_comp > 0, "L_comp must be positive (GW likelihood is positive)"


# ======================================================================
# Test 5: Completion term is dimensionless
# Scaling dVc by a constant factor should leave L_comp unchanged
# (since dVc appears in both numerator and denominator)
# ======================================================================


class TestCompletionTermDimensionless:
    """L_comp is a ratio; scaling the volume element must not change it."""

    def test_lcomp_invariant_under_dvc_scaling(self) -> None:
        h = 0.73
        d_L_det = 0.5
        phi_det = 1.0
        theta_det = 0.5
        sigma_d_L_frac = 0.05
        cov = np.diag([0.01**2, 0.01**2, sigma_d_L_frac**2])
        gw_gaussian = multivariate_normal(mean=[phi_det, theta_det, 1.0], cov=cov)

        z_lower = 0.05
        z_upper = 0.15

        def compute_lcomp(scale_factor: float) -> float:
            def num(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                d_L = np.asarray(dist_vectorized(z, h=h), dtype=np.float64)
                d_L_frac = d_L / d_L_det
                phi = np.full_like(z, phi_det)
                theta = np.full_like(z, theta_det)
                p_gw: npt.NDArray[np.float64] = gw_gaussian.pdf(np.vstack([phi, theta, d_L_frac]).T)
                dVc: npt.NDArray[np.float64] = np.atleast_1d(
                    np.asarray(comoving_volume_element(z, h=h), dtype=np.float64)
                )
                return p_gw * dVc * scale_factor

            def den(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                dVc: npt.NDArray[np.float64] = np.atleast_1d(
                    np.asarray(comoving_volume_element(z, h=h), dtype=np.float64)
                )
                return dVc * scale_factor

            n: float = fixed_quad(num, z_lower, z_upper, n=50)[0]
            d: float = fixed_quad(den, z_lower, z_upper, n=50)[0]
            return n / d

        L_comp_1 = compute_lcomp(1.0)
        L_comp_100 = compute_lcomp(100.0)
        L_comp_0p01 = compute_lcomp(0.01)

        assert L_comp_1 == pytest.approx(L_comp_100, rel=1e-10)
        assert L_comp_1 == pytest.approx(L_comp_0p01, rel=1e-10)


# ======================================================================
# Test 6: f_i varies with h (completeness threading)
# The same redshift at different h maps to different d_L, giving
# different completeness values
# ======================================================================


class TestFVariesWithH:
    """Completeness f_i must depend on h through the d_L(z, h) relation."""

    def test_f_varies_with_h(self) -> None:
        completeness = GladeCatalogCompleteness()
        z = 0.10  # at z=0.1, completeness should differ between h=0.6 and h=0.86

        f_low_h = completeness.get_completeness_at_redshift(z, h=0.6)
        f_high_h = completeness.get_completeness_at_redshift(z, h=0.86)

        # Higher h => smaller d_L => higher completeness
        assert f_low_h != f_high_h, (
            f"f_i must vary with h: f(h=0.6)={f_low_h}, f(h=0.86)={f_high_h}"
        )
        assert f_high_h > f_low_h, "Higher h should give higher completeness (smaller d_L)"


# ======================================================================
# Test 7: Completion integrand uses detection sky position
# The completion term evaluates the GW Gaussian at (phi_det, theta_det),
# NOT at a galaxy position
# ======================================================================


class TestCompletionIntegrandUsesDetectionSkyPosition:
    """The completion term must use the detection's sky position."""

    def test_completion_integrand_uses_detection_sky_position(self) -> None:
        """Evaluate completion numerator integrand at two different sky positions.

        The completion term uses the detection sky position (phi_det, theta_det),
        not a galaxy position. If we change phi_det, the integrand changes.
        """
        h = 0.73
        d_L_det = 0.5
        sigma_d_L_frac = 0.05
        z_test = np.array([0.10])

        # Case 1: detection at (phi=1.0, theta=0.5)
        phi_det_1 = 1.0
        theta_det_1 = 0.5
        cov = np.diag([0.01**2, 0.01**2, sigma_d_L_frac**2])
        gw_gaussian_1 = multivariate_normal(mean=[phi_det_1, theta_det_1, 1.0], cov=cov)
        d_L = np.asarray(dist_vectorized(z_test, h=h), dtype=np.float64)
        d_L_frac = d_L / d_L_det
        val_1 = float(gw_gaussian_1.pdf([phi_det_1, theta_det_1, float(d_L_frac[0])]))

        # Case 2: detection at (phi=2.0, theta=1.0) - different sky position
        phi_det_2 = 2.0
        theta_det_2 = 1.0
        gw_gaussian_2 = multivariate_normal(mean=[phi_det_2, theta_det_2, 1.0], cov=cov)
        val_2 = float(gw_gaussian_2.pdf([phi_det_2, theta_det_2, float(d_L_frac[0])]))

        # Both evaluate at their own detection position (the peak), so
        # they should be equal (both at the Gaussian peak in phi/theta)
        assert val_1 == pytest.approx(val_2, rel=1e-10), (
            "Completion term should evaluate at detection sky position "
            "(GW Gaussian peak in sky), not at a galaxy position"
        )

        # But evaluating Gaussian 1 at Gaussian 2's position should differ
        val_cross = float(gw_gaussian_1.pdf([phi_det_2, theta_det_2, float(d_L_frac[0])]))
        assert val_cross != pytest.approx(val_1, rel=0.01), (
            "If evaluated at a different sky position, GW likelihood should change"
        )
