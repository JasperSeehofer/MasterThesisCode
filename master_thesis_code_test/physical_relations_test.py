import numpy as np
import pytest

from master_thesis_code.constants import OMEGA_DE, OMEGA_M, W_0, W_A, H
from master_thesis_code.physical_relations import (
    convert_redshifted_mass_to_true_mass,
    convert_true_mass_to_redshifted_mass,
    dist,
    dist_derivative,
    dist_to_redshift,
    dist_to_redshift_error_proagation,
    dist_vectorized,
    get_redshift_outer_bounds,
    hubble_function,
)


@pytest.mark.parametrize("redshift, expected_distance", [(1.0, 6.5)])
def test_dist(redshift: float, expected_distance: float) -> None:
    result = round(dist(redshift), 1)
    assert result == expected_distance


@pytest.mark.parametrize(
    "distance, distance_error, h_min, h_max, Omega_m_min, Omega_m_max, expected_bounds",
    [
        (0.5, 0.05, 0.6, 0.83, 0.04, 0.5, (0.0, 0.5)),  # example parameters
    ],
)
def test_get_redshift_outer_bounds(
    distance: float,
    distance_error: float,
    h_min: float,
    h_max: float,
    Omega_m_min: float,
    Omega_m_max: float,
    expected_bounds: tuple,
) -> None:
    result_lower_bound, result_upper_bound = get_redshift_outer_bounds(
        distance, distance_error, h_min, h_max, Omega_m_min, Omega_m_max
    )
    best_guess = dist_to_redshift(distance, H, OMEGA_M, OMEGA_DE, W_0, W_A)
    estimated_lower_bound = dist_to_redshift(
        distance - 2 * distance_error, H, OMEGA_M, OMEGA_DE, W_0, W_A
    )
    estimated_upper_bound = dist_to_redshift(
        distance + 2 * distance_error, H, OMEGA_M, OMEGA_DE, W_0, W_A
    )
    assert result_upper_bound >= estimated_upper_bound
    assert result_lower_bound <= estimated_lower_bound


def test_dist_at_zero_redshift() -> None:
    """Fundamental analytical limit: dist(0) == 0.0."""
    result = dist(0)
    assert result == 0.0


def test_dist_monotonically_increasing() -> None:
    assert dist(0.5) < dist(1.0) < dist(2.0)


def test_dist_returns_float() -> None:
    assert isinstance(dist(1.0), float)


def test_hubble_function_at_zero() -> None:
    """Normalization: hubble_function(0) == 1.0 for flat LCDM with Omega_m + Omega_de = 1."""
    result = hubble_function(0)
    assert isinstance(result, float)
    assert abs(result - 1.0) < 1e-10


def test_hubble_function_positive() -> None:
    for z in [0.5, 1.0, 2.0]:
        result = hubble_function(z)
        assert isinstance(result, float)
        assert result > 0


def test_dist_to_redshift_at_zero() -> None:
    """dist_to_redshift(0.0) should return 0.0."""
    result = dist_to_redshift(0.0)
    assert abs(result) < 1e-6


@pytest.mark.parametrize("z", [0.5, 1.0, 2.0])
def test_dist_round_trip(z: float) -> None:
    """dist_to_redshift(dist(z)) should recover z within 1e-5."""
    d = dist(z)
    z_recovered = dist_to_redshift(d)
    assert abs(z_recovered - z) < 1e-5


def test_dist_vectorized_shape() -> None:
    zs = np.array([0.1, 0.5, 1.0, 2.0])
    result = dist_vectorized(zs)
    assert len(result) == len(zs)


def test_dist_vectorized_matches_scalar() -> None:
    zs = np.array([0.1, 0.5, 1.0, 2.0])
    result = dist_vectorized(zs)
    for i, z in enumerate(zs):
        assert abs(float(result[i]) - dist(z)) < 1e-10


@pytest.mark.parametrize("z", [0.5, 1.0])
def test_dist_derivative_positive(z: float) -> None:
    """The luminosity distance is monotonically increasing, so its derivative must be positive."""
    assert dist_derivative(z) > 0


def test_convert_redshifted_mass_to_true_mass() -> None:
    """M_z / (1+z) == M: algebraic identity."""
    M = 1e5
    z = 0.5
    M_z = M * (1 + z)
    M_err = 0.0
    z_err = 0.0
    M_recovered, _ = convert_redshifted_mass_to_true_mass(M_z, M_err, z, z_err)
    assert abs(M_recovered - M) < 1e-10


def test_convert_true_mass_to_redshifted_mass() -> None:
    """M * (1+z) == M_z: algebraic identity."""
    M = 1e5
    z = 0.5
    M_err = 0.0
    z_err = 0.0
    M_z, _ = convert_true_mass_to_redshifted_mass(M, M_err, z, z_err)
    assert abs(M_z - M * (1 + z)) < 1e-10


def test_mass_conversion_round_trip() -> None:
    """Two-way round-trip consistency."""
    M = 1e5
    z = 0.3
    M_z, _ = convert_true_mass_to_redshifted_mass(M, 0.0, z, 0.0)
    M_back, _ = convert_redshifted_mass_to_true_mass(M_z, 0.0, z, 0.0)
    assert abs(M_back - M) < 1e-10


def test_dist_to_redshift_error_propagation_positive() -> None:
    """Error propagation result should be a positive float."""
    result = dist_to_redshift_error_proagation(distance=1.0, distance_error=0.1)
    assert isinstance(result, float)
    assert result > 0


def test_dist_varies_with_hubble_constant() -> None:
    """Higher H₀ → smaller luminosity distance for the same redshift (d_L ~ c/H₀)."""
    z = 1.0
    d_low_h = dist(z, h=0.70)
    d_high_h = dist(z, h=0.73)

    assert d_low_h > 0
    assert d_high_h > 0
    # A higher Hubble constant compresses the distance scale: d_L ∝ 1/H₀
    assert d_high_h < d_low_h


def test_dist_hubble_scaling_approximate() -> None:
    """dist is approximately proportional to 1/H₀ at fixed redshift (exact for small z)."""
    z = 0.1
    h1 = 0.70
    h2 = 0.73
    d1 = dist(z, h=h1)
    d2 = dist(z, h=h2)
    # Expected ratio: d1/d2 ≈ h2/h1
    expected_ratio = h2 / h1
    actual_ratio = d1 / d2
    # Tolerance of 2% to account for non-linear corrections at z=0.1
    assert abs(actual_ratio - expected_ratio) < 0.02
