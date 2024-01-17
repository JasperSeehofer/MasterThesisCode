import pytest
from master_thesis_code.datamodels.parameter_space import ParameterSpace
from master_thesis_code.LISA_configuration import LISAConfiguration


@pytest.fixture(name="self")
def self_fixture() -> LISAConfiguration:
    parameter_space = ParameterSpace()
    return LISAConfiguration(parameter_space=parameter_space, dt=10)

"""
Test cases:
- F_+ and F_x correctly implemented.
- measurement 1 & 2 
- Easy SNR check
- power spectral density
"""

# check F_+

