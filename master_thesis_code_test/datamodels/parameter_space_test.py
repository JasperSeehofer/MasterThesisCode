import pytest
import master_thesis_code

from master_thesis_code.datamodels.parameter_space import ParameterSpace, Parameter, parameters_configuration

@pytest.fixture
def self() -> ParameterSpace:
    return ParameterSpace()

def test_parameters_to_dict_keys(self: ParameterSpace) -> None:
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
    
@pytest.mark.parametrize("parameter_info", parameters_configuration)
def test_randomize_parameter(self: ParameterSpace, parameter_info: Parameter) -> None:
    lower_limit = parameter_info.lower_limit
    upper_limit = parameter_info.upper_limit
    self.randomize_parameter(parameter_info=parameter_info)
    parameter_value = getattr(self, parameter_info.symbol)
    assert parameter_value >= lower_limit
    assert parameter_value <= upper_limit
