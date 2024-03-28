import pytest
from master_thesis_code.physical_relations import dist


@pytest.mark.parametrize("redshift, expected_distance", [(1.0, 6.5)])
def test_dist(redshift: float, expected_distance: float) -> None:
    result = round(dist(redshift), 1)
    assert result == expected_distance
