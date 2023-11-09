import pytest
from master_thesis_code.datamodels.parameter_space import ParameterSpace
from master_thesis_code.LISA_configuration import LISAConfiguration


@pytest.fixture(name="self")
def self_fixture() -> LISAConfiguration:
    parameter_space = ParameterSpace()
    return LISAConfiguration(parameter_space=parameter_space,dt=10)
