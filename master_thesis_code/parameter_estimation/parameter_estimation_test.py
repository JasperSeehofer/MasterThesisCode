import pytest
import numpy as np
from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation

@pytest.fixture
def self() -> ParameterEstimation:
    return ParameterEstimation(wave_generation_type="FastSchwarzschildEccentricFlux")

@pytest.fixture
def waveform() -> np.ndarray[float]:
    return []

def finite_differences_test(self, waveform: np.ndarray[float]) -> None:
    pass