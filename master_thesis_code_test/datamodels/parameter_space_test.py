import math

import numpy as np
import pytest

from master_thesis_code.datamodels.parameter_space import (
    Parameter,
    ParameterSpace,
    log_uniform,
    polar_angle_distribution,
    uniform,
)

EXPECTED_PARAMETER_KEYS = {
    "M",
    "mu",
    "a",
    "p0",
    "e0",
    "x0",
    "luminosity_distance",
    "qS",
    "phiS",
    "qK",
    "phiK",
    "Phi_phi0",
    "Phi_theta0",
    "Phi_r0",
}


@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.0, 1.0),
        (-5.0, 5.0),
        (100.0, 200.0),
        (0.0, 2 * np.pi),
        (-1.0, 1.0),
    ],
)
def test_uniform_in_bounds(lower: float, upper: float) -> None:
    for _ in range(100):
        value = uniform(lower, upper)
        assert lower <= value <= upper


@pytest.mark.parametrize(
    "lower, upper",
    [
        (1e4, 1e7),
        (1.0, 1e2),
        (1e-3, 1.0),
    ],
)
def test_log_uniform_in_bounds(lower: float, upper: float) -> None:
    for _ in range(100):
        value = log_uniform(lower, upper)
        assert lower <= value <= upper


def test_polar_angle_distribution_in_range() -> None:
    for _ in range(100):
        value = polar_angle_distribution(0.0, np.pi)
        assert 0.0 <= value <= np.pi


def test_parameter_space_default_construction() -> None:
    ps = ParameterSpace()
    assert ps is not None
    assert isinstance(ps.M, Parameter)
    assert isinstance(ps.mu, Parameter)
    assert isinstance(ps.a, Parameter)
    assert isinstance(ps.luminosity_distance, Parameter)


def test_randomize_parameter_stays_in_bounds() -> None:
    ps = ParameterSpace()
    for _ in range(50):
        ps.randomize_parameter(ps.M)
        assert ps.M.lower_limit <= ps.M.value <= ps.M.upper_limit


def test_randomize_parameters_all_within_bounds() -> None:
    ps = ParameterSpace()
    for _ in range(10):
        ps.randomize_parameters()
        for symbol, value in ps._parameters_to_dict().items():
            param = getattr(ps, symbol)
            assert param.lower_limit <= value <= param.upper_limit, (
                f"Parameter '{symbol}' value {value} out of bounds "
                f"[{param.lower_limit}, {param.upper_limit}]"
            )


def test_parameters_to_dict_has_14_keys() -> None:
    ps = ParameterSpace()
    assert len(ps._parameters_to_dict()) == 14


def test_parameters_to_dict_has_correct_keys() -> None:
    ps = ParameterSpace()
    keys = set(ps._parameters_to_dict().keys())
    assert keys == EXPECTED_PARAMETER_KEYS


def test_parameters_to_dict_values_are_floats() -> None:
    ps = ParameterSpace()
    ps.randomize_parameters()
    for key, value in ps._parameters_to_dict().items():
        assert isinstance(value, float), f"Parameter '{key}' value {value!r} is not a float"


def test_no_nan_after_randomize() -> None:
    ps = ParameterSpace()
    ps.randomize_parameters()
    for key, value in ps._parameters_to_dict().items():
        assert not math.isnan(value), f"Parameter '{key}' is NaN after randomization"


def test_set_host_galaxy_parameters_updates_fields() -> None:
    """set_host_galaxy_parameters() must update dist, qS, phiS, and M on the ParameterSpace."""
    from master_thesis_code.galaxy_catalogue.handler import HostGalaxy
    from master_thesis_code.physical_relations import dist

    phi_s = 1.23
    q_s = 0.45
    z = 0.3
    M = 5e5

    host = HostGalaxy.from_attributes(
        phiS=phi_s,
        qS=q_s,
        z=z,
        z_error=0.001,
        M=M,
        M_error=1e4,
    )

    ps = ParameterSpace()
    ps.set_host_galaxy_parameters(host)

    assert ps.phiS.value == phi_s
    assert ps.qS.value == q_s
    assert ps.M.value == M
    expected_dist = dist(z)
    assert abs(ps.luminosity_distance.value - expected_dist) < 1e-10


def test_set_host_galaxy_parameters_overwrites_previous_values() -> None:
    """Calling set_host_galaxy_parameters() twice updates to the latest host's values."""
    from master_thesis_code.galaxy_catalogue.handler import HostGalaxy

    host1 = HostGalaxy.from_attributes(phiS=0.1, qS=0.2, z=0.1, z_error=0.001, M=1e5, M_error=1e3)
    host2 = HostGalaxy.from_attributes(phiS=2.5, qS=1.0, z=0.5, z_error=0.001, M=9e5, M_error=1e4)

    ps = ParameterSpace()
    ps.set_host_galaxy_parameters(host1)
    ps.set_host_galaxy_parameters(host2)

    assert ps.phiS.value == 2.5
    assert ps.qS.value == 1.0
    assert ps.M.value == 9e5
