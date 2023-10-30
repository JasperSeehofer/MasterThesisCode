import pytest
from master_thesis_code.datamodels.parameter_space import ParameterSpace

@pytest.fixture
def self():
    return ParameterSpace()

def test_parameters_to_dict_keys(self) -> None:
    assert list(self._parameters_to_dict().keys()) == [
        "M",
        "mu",
        "a",
        "p0",
        "e0",
        "x0",
        "dist",
        "qS",
        "phiS",
        "qK",
        "phiK",
        "Phi_theta0",
        "Phi_phi0",
        "Phi_r0",
    ]