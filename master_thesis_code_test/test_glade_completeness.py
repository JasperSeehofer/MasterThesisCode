"""Tests for GLADE+ completeness function and comoving volume element.

Covers:
- Completeness fraction bounds (contract: claim-fz-range, test-fz-bounds)
- Dalya et al. (2022) reference values (contract: claim-fz-correct, test-fz-dalya)
- h-dependence of redshift interface (contract: claim-h-dependence, test-fz-h-dependence)
- Vectorized input/output (contract: claim-vectorized, test-fz-vectorized)
- Comoving volume element dimensions and scaling (contract: claim-dVc, test-dVc-*)
- Backward compatibility of legacy interface
- Edge cases

References
----------
Dalya et al. (2022), arXiv:2110.06184, Section 3.
Hogg (1999), arXiv:astro-ph/9905116, Eq. (28).
Gray et al. (2020), arXiv:1908.06050, Sec. II.3.1.
"""

import numpy as np
import pytest

from master_thesis_code.constants import GPC_TO_MPC, SPEED_OF_LIGHT_KM_S
from master_thesis_code.galaxy_catalogue.glade_completeness import GladeCatalogCompleteness
from master_thesis_code.physical_relations import comoving_volume_element, dist

# ======================================================================
# Completeness fraction tests
# ======================================================================


class TestCompletenessFractionBounds:
    """Contract: claim-fz-range, test-fz-bounds."""

    def test_completeness_fraction_bounds(self) -> None:
        """f(z) in [0, 1] for z in np.linspace(0, 0.25, 200) at h=0.73."""
        gc = GladeCatalogCompleteness()
        z_arr = np.linspace(0.0, 0.25, 200)
        frac = gc.get_completeness_at_redshift(z_arr, h=0.73)
        assert isinstance(frac, np.ndarray)
        assert np.all(frac >= 0.0), f"Found negative values: {frac[frac < 0]}"
        assert np.all(frac <= 1.0), f"Found values > 1: {frac[frac > 1]}"


class TestCompletenessFractionAtZero:
    """f(z=0) must be 1.0 for any h."""

    def test_completeness_fraction_at_zero_default_h(self) -> None:
        gc = GladeCatalogCompleteness()
        assert gc.get_completeness_at_redshift(0.0) == 1.0

    @pytest.mark.parametrize("h", [0.6, 0.73, 0.86])
    def test_completeness_fraction_at_zero_various_h(self, h: float) -> None:
        gc = GladeCatalogCompleteness()
        assert gc.get_completeness_at_redshift(0.0, h=h) == 1.0


class TestCompletenessMonotonic:
    """f(z) should be approximately non-increasing.

    The raw digitized data has a small non-monotonicity around 200-240 Mpc
    where completeness rises slightly (43.0 -> 43.6%). We allow a tolerance
    of 0.01 (1 percentage point) for this.
    """

    def test_completeness_fraction_monotonic(self) -> None:
        gc = GladeCatalogCompleteness()
        z_arr = np.linspace(0.001, 0.20, 200)
        frac = gc.get_completeness_at_redshift(z_arr, h=0.73)
        assert isinstance(frac, np.ndarray)
        # Allow small tolerance for digitization non-monotonicity
        diffs = np.diff(frac)
        violations = diffs[diffs > 0.01]
        assert len(violations) == 0, (
            f"Monotonicity violated by > 0.01 at {len(violations)} points: "
            f"max increase = {violations.max():.4f}"
        )


class TestCompletenessDalyaReferencePoints:
    """Contract: claim-fz-correct, test-fz-dalya.

    Note: The digitized data in the code represents a specific completeness
    curve from Dalya et al. (2022) Fig. 2. The numerical values are
    determined by the digitized data -- these tests verify the interpolation
    and interface work correctly at physically meaningful redshifts.

    At h=0.73: z=0.029 maps to d_L ~ 122 Mpc, z=0.11 maps to d_L ~ 491 Mpc.
    """

    def test_completeness_at_z029(self) -> None:
        """At z=0.029 (d_L~122 Mpc), completeness ~ 50% from digitized data."""
        gc = GladeCatalogCompleteness()
        f = gc.get_completeness_at_redshift(0.029, h=0.73)
        # The digitized data gives ~50% at d_L~122 Mpc
        assert 0.40 <= f <= 0.60, f"f(z=0.029) = {f:.4f}, expected ~0.50"

    def test_completeness_at_z011(self) -> None:
        """At z=0.11 (d_L~491 Mpc), completeness should be well below 50%."""
        gc = GladeCatalogCompleteness()
        f = gc.get_completeness_at_redshift(0.11, h=0.73)
        assert f < 0.50, f"f(z=0.11) = {f:.4f}, expected < 0.50"

    def test_completeness_decreasing_with_redshift(self) -> None:
        """Completeness at z=0.029 > completeness at z=0.11."""
        gc = GladeCatalogCompleteness()
        f_low = gc.get_completeness_at_redshift(0.029, h=0.73)
        f_high = gc.get_completeness_at_redshift(0.11, h=0.73)
        assert f_low > f_high


class TestCompletenessHDependence:
    """Contract: claim-h-dependence, test-fz-h-dependence.

    Higher h => H0 is larger => d_L = c(1+z)/H0 * integral is SMALLER
    at the same z => the catalog is MORE complete at that (shorter) distance.

    So f(z, h=0.86) > f(z, h=0.6) at fixed z.
    """

    def test_completeness_h_dependence(self) -> None:
        gc = GladeCatalogCompleteness()
        f_low_h = gc.get_completeness_at_redshift(0.05, h=0.6)
        f_high_h = gc.get_completeness_at_redshift(0.05, h=0.86)
        assert f_low_h != f_high_h, "f should differ for different h"
        # Higher h -> smaller d_L -> higher completeness
        assert f_high_h > f_low_h, (
            f"f(z=0.05, h=0.86) = {f_high_h:.4f} should be > "
            f"f(z=0.05, h=0.6) = {f_low_h:.4f} "
            "(higher h -> smaller d_L -> higher completeness)"
        )

    def test_h_dependence_via_distance(self) -> None:
        """Verify the mechanism: higher h -> smaller d_L at same z."""
        d_low_h = dist(0.05, h=0.6) * GPC_TO_MPC
        d_high_h = dist(0.05, h=0.86) * GPC_TO_MPC
        assert d_high_h < d_low_h, (
            f"d_L(z=0.05, h=0.86)={d_high_h:.1f} Mpc should be < "
            f"d_L(z=0.05, h=0.6)={d_low_h:.1f} Mpc"
        )


class TestCompletenessVectorized:
    """Contract: claim-vectorized, test-fz-vectorized."""

    def test_completeness_vectorized_shape(self) -> None:
        gc = GladeCatalogCompleteness()
        z_arr = np.array([0.01, 0.05, 0.10, 0.15])
        result = gc.get_completeness_at_redshift(z_arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == z_arr.shape

    def test_completeness_vectorized_matches_scalar(self) -> None:
        gc = GladeCatalogCompleteness()
        z_arr = np.array([0.01, 0.05, 0.10, 0.15])
        vec_result = gc.get_completeness_at_redshift(z_arr)
        assert isinstance(vec_result, np.ndarray)
        for i, zi in enumerate(z_arr):
            scalar_result = gc.get_completeness_at_redshift(float(zi))
            assert isinstance(scalar_result, float)
            assert vec_result[i] == pytest.approx(scalar_result, rel=1e-10)


class TestCompletenessEdgeCases:
    """Edge cases: negative z, very large z."""

    def test_negative_distance(self) -> None:
        """Negative distance returns 1.0 (complete)."""
        gc = GladeCatalogCompleteness()
        assert gc.get_completeness_fraction(-10.0) == 1.0

    def test_very_large_redshift(self) -> None:
        """z=1.0 should give a small positive value (flat extrapolation)."""
        gc = GladeCatalogCompleteness()
        f = gc.get_completeness_at_redshift(1.0, h=0.73)
        assert isinstance(f, float)
        assert f > 0.0, "Should not be zero (flat extrapolation beyond max distance)"
        assert not np.isnan(f), "Should not be NaN"
        # Should be the last digitized value / 100
        assert f == pytest.approx(0.2134, rel=0.01)

    def test_zero_distance(self) -> None:
        """Distance = 0 returns completeness = 1.0."""
        gc = GladeCatalogCompleteness()
        assert gc.get_completeness_fraction(0.0) == 1.0


class TestBackwardCompat:
    """Legacy get_completeness() returns percent (not fraction)."""

    def test_backward_compat_get_completeness(self) -> None:
        gc = GladeCatalogCompleteness()
        result = gc.get_completeness(200)
        # Should return ~44 in percent units
        assert 43.0 <= result <= 45.0, f"get_completeness(200) = {result:.2f}, expected ~44%"

    def test_backward_compat_returns_zero_beyond_max(self) -> None:
        """Legacy interface returns 0 beyond max distance (original behavior)."""
        gc = GladeCatalogCompleteness()
        result = gc.get_completeness(800)
        assert result == 0.0


# ======================================================================
# Comoving volume element tests
# ======================================================================


class TestComovingVolumeElementPositive:
    """Contract: claim-dVc, test-dVc-dimensions."""

    def test_comoving_volume_element_positive(self) -> None:
        """dVc/dz > 0 for z > 0."""
        z_arr = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
        result = comoving_volume_element(z_arr, h=0.73)
        assert isinstance(result, np.ndarray)
        assert np.all(result > 0.0)


class TestComovingVolumeElementZSquaredScaling:
    """Contract: claim-dVc, test-dVc-limit.

    At z << 1: dVc/dz/dOmega ~ (c/H0)^3 * z^2.
    """

    def test_z_squared_scaling(self) -> None:
        """dVc(z=0.002) / dVc(z=0.001) ~ 4.0 within 5%."""
        dVc_002 = comoving_volume_element(0.002, h=0.73)
        dVc_001 = comoving_volume_element(0.001, h=0.73)
        assert isinstance(dVc_002, float)
        assert isinstance(dVc_001, float)
        ratio = dVc_002 / dVc_001
        assert ratio == pytest.approx(4.0, rel=0.05), (
            f"Ratio dVc(0.002)/dVc(0.001) = {ratio:.4f}, expected ~4.0"
        )


class TestComovingVolumeElementAtZZero:
    """dVc/dz(z=0) = 0 because d_com(z=0) = 0."""

    def test_volume_element_at_z_zero(self) -> None:
        dVc = comoving_volume_element(0.0, h=0.73)
        assert isinstance(dVc, float)
        assert dVc == pytest.approx(0.0, abs=1e-6)


class TestComovingVolumeElementDimensions:
    """Spot-check dVc/dz at z=0.01 against analytical low-z formula.

    At z << 1: dVc/dz/dOmega ~ (c/H0)^3 * z^2
    where c/H0 in Mpc = SPEED_OF_LIGHT_KM_S / (h * 100) Mpc.
    """

    def test_volume_element_dimensions_spot_check(self) -> None:
        z = 0.01
        h = 0.73
        dVc = comoving_volume_element(z, h=h)
        assert isinstance(dVc, float)

        # Analytical low-z approximation
        c_over_H0_mpc = SPEED_OF_LIGHT_KM_S / (h * 100.0)  # Mpc
        expected = c_over_H0_mpc**3 * z**2

        # Should agree to within ~1% at z=0.01
        assert dVc == pytest.approx(expected, rel=0.02), (
            f"dVc/dz(z=0.01) = {dVc:.2e}, analytical = {expected:.2e}"
        )

    def test_volume_element_vectorized(self) -> None:
        """Array input produces array output of same shape."""
        z_arr = np.array([0.01, 0.05, 0.1])
        result = comoving_volume_element(z_arr, h=0.73)
        assert isinstance(result, np.ndarray)
        assert result.shape == z_arr.shape
        # Each value should match scalar call
        for i, zi in enumerate(z_arr):
            scalar = comoving_volume_element(float(zi), h=0.73)
            assert result[i] == pytest.approx(scalar, rel=1e-10)
