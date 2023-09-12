from dataclasses import dataclass, field
from constants import INFINITY
import numpy as np
import random
import typing

@dataclass
class Parameter():
    """Main class for parameters."""
    symbol: str
    unit: str
    lower_limit: float
    upper_limit: float


pn5_parameters_configuration: list[Parameter] = [
        Parameter(symbol="M", unit="solar masses", lower_limit=1e5, upper_limit=1e6),  # mass of the MBH (massive black hole) in solar masses
        Parameter(symbol="mu", unit="solar masses", lower_limit=1e1, upper_limit=1e2),  # mass of the CO (compact object) in solar masses
        Parameter(symbol="a", unit="dimensionless", lower_limit=0., upper_limit=1000000000),  # dimensionless spin of the MBH
        Parameter(symbol="a2_vec", unit="xxx", lower_limit=0., upper_limit=INFINITY),  # 3-dimensional spin angular momentum of the CO
        Parameter(symbol="p_0", unit="xxx", lower_limit=0., upper_limit=INFINITY),  # Kepler-orbit parameter: separation
        Parameter(symbol="e_0", unit="dimensionless", lower_limit=0., upper_limit=INFINITY),  # Kepler-orbit parameter: eccentricity
        Parameter(symbol="x_I0", unit="dimensionless", lower_limit=-1., upper_limit=1),  # Kepler-orbit parameter: x_I0=cosI (I is the inclination)
        Parameter(symbol="d_L", unit="xxx", lower_limit=0., upper_limit=INFINITY),  # luminosity distance
        Parameter(symbol="theta_S", unit="radian", lower_limit=0., upper_limit=2*np.pi),  # polar sky-localization (solar system barycenter frame)
        Parameter(symbol="phi_S", unit="radian", lower_limit=0., upper_limit=np.pi),  # azimuthal sky-localization (solar system barycenter frame)
        Parameter(symbol="theta_K", unit="radian", lower_limit=0., upper_limit=2*np.pi),  # polar orientation of spin angular momentum 
        Parameter(symbol="phi_K", unit="radian", lower_limit=0., upper_limit=np.pi),  # azimuthal orientation of spin angular momentum 
        Parameter(symbol="Phi_theta0", unit="radian", lower_limit=0., upper_limit=2*np.pi),  # polar phase
        Parameter(symbol="Phi_phi0", unit="radian", lower_limit=0, upper_limit=np.pi),  # azimuthal phase
        Parameter(symbol="Phi_r0", unit="radian", lower_limit=0, upper_limit=INFINITY),  # radial phase
    ]

schwarzschild_parameters_configuration: list[Parameter] = [
        Parameter(symbol="M", unit="solar masses", lower_limit=1e6, upper_limit=1e7),  # mass of the MBH (massive black hole) in solar masses
        Parameter(symbol="mu", unit="solar masses", lower_limit=1e1, upper_limit=1e2),  # mass of the CO (compact object) in solar masses
        Parameter(symbol="p_0", unit="xxx", lower_limit=10., upper_limit=20),  # Kepler-orbit parameter: separation
        Parameter(symbol="e_0", unit="dimensionless", lower_limit=0.2, upper_limit=0.6),  # Kepler-orbit parameter: eccentricity
        Parameter(symbol="theta", unit="radian", lower_limit=-np.pi/2, upper_limit=np.pi/2),  # polar viewing angle in source frame 
        Parameter(symbol="phi", unit="radian", lower_limit=0.01, upper_limit=2*np.pi),  # azimuthal viewing angle in source frame 
    ]

@dataclass
class ParameterSpace():
    """
    Dataclass to manage the parameter space of a simulation.
    """
    parameters_configuration: list[Parameter] = field(default_factory=list)
    parameters_to_marginalize: list[str] = field(default_factory=list)

    def randomize_parameter(self, parameter_info: Parameter) -> None:
        upper_limit = parameter_info.upper_limit
        lower_limit = parameter_info.lower_limit
        parameter_value = lower_limit + (upper_limit - lower_limit)*random.random()
        setattr(self, parameter_info.symbol, parameter_value)
            

    def randomize_parameters(self) -> None:
        for parameter_symbol in vars(self).keys():
            parameter_info = next((parameter for parameter in self.parameters_configuration if parameter.symbol == parameter_symbol), None)

            if parameter_info is not None:
                self.randomize_parameter(parameter_info=parameter_info)

class SchwarzschildParameterSpace(ParameterSpace):
    M: float  # mass of the MBH (massive black hole) in solar masses
    mu: float  # mass of the CO (compact object) in solar masses
    p_0: float  # Kepler-orbit parameter: separation
    e_0: float  # Kepler-orbit parameter: eccentricity
    theta: float # polar viewing angle in source frame
    phi: float # azimuthal viewing angle in source frame

    def __init__(self, parameters_to_marginalize: list[str] = []):
        self.M = 0.
        self.mu = 0.
        self.p_0 = 0.
        self.e_0 = 0.
        self.theta = 0.
        self.phi = 0.

        self.parameters_to_marginalize = parameters_to_marginalize
        self.parameters_configuration = schwarzschild_parameters_configuration
        self.randomize_parameters()

    def _parameters_to_dict(self) -> dict:
        return {
            "M": self.M,
            "mu": self.mu,
            "p0": self.p_0,
            "e0": self.e_0,
            "theta": self.theta,
            "phi": self.phi,
        }

class PN5ParameterSpace(ParameterSpace):
    M: float  # mass of the MBH (massive black hole) in solar masses
    mu: float  # mass of the CO (compact object) in solar masses
    a: float  # dimensionless spin of the MBH
    a2_vec: tuple[float]  # 3-dimensional spin angular momentum of the CO
    p_0: float  # Kepler-orbit parameter: separation
    e_0: float  # Kepler-orbit parameter: eccentricity
    x_I0: float  # Kepler-orbit parameter: x_I0=cosI (I is the inclination)
    d_L: float  # luminosity distance
    theta_S: float  # polar sky-localization (solar system barycenter frame)
    phi_S: float  # azimuthal sky-localization (solar system barycenter frame)
    theta_K: float  # polar orientation of spin angular momentum 
    phi_K: float  # azimuthal orientation of spin angular momentum 
    Phi_theta0: float  # polar phase
    Phi_phi0: float  # azimuthal phase
    Phi_r0: float  # radial phase

    def __init__(self, parameters_to_marginalize: list[str] = []):
        self.M = 0.
        self.mu = 0.
        self.a = 0.
        self.a2_vec = 0.
        self.p_0 = 0.
        self.e_0 = 0.
        self.x_I0 = 0.
        self.d_L = 0.
        self.theta_S = 0.
        self.phi_S = 0.
        self.theta_K = 0.
        self.phi_K = 0.
        self.Phi_theta0 = 0.
        self.Phi_phi0 = 0.
        self.Phi_r0 = 0.
        self.parameters_to_marginalize = parameters_to_marginalize
        self.parameters_configuration = pn5_parameters_configuration
        self.randomize_parameters()
