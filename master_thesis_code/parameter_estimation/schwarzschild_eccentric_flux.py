import numpy as np
from datamodels.parameter_space import ParameterSpace

parameter_space = ParameterSpace()

class SchwarzschildParameterEstimation():
    parameter_space: ParameterSpace
    waveform: callable

    def __init__(self):
        self.parameter_space = ParameterSpace().assume_schwarzschild()