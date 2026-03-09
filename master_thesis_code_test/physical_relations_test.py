import pytest

from master_thesis_code.constants import OMEGA_DE, OMEGA_M, W_0, W_A, H
from master_thesis_code.physical_relations import dist, dist_to_redshift, get_redshift_outer_bounds


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
