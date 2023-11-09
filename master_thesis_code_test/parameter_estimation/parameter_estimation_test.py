import pytest
import numpy as np
from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation

@pytest.fixture(name="self")
def self_fixture() -> ParameterEstimation:
    return ParameterEstimation(wave_generation_type="FastSchwarzschildEccentricFlux")

@pytest.fixture(name="waveform")
def waveform_fixture() -> np.ndarray[float]:
    return np.zeros(10)

@pytest.fixture("linear_neighbouring_waveform")
def linear_neigbouring_waveform() -> np.ndarray:
    pass

def example_function(x: float, a: float, b: float) -> float:
    return a*x + np.cos(b*x)

def example_function_derivative_b(x: float, b: float) -> float:
    return -1*np.sin(b*x)*x

def test_finite_differences_constant(self, waveform: np.ndarray[float], neighouring_waveform: np.ndarray[float]) -> None:
    derivative = self.finite_differences()


