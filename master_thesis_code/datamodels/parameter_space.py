from dataclasses import dataclass, field
from master_thesis_code.constants import INFINITY
import numpy as np
import random


@dataclass
class Parameter():
    """Main class for parameters."""
    symbol: str
    unit: str
    lower_limit: float
    upper_limit: float
    derivative_epsilon: float = 1e-6


parameters_configuration: list[Parameter] = [
        Parameter(symbol="M", unit="solar masses", lower_limit=1e5, upper_limit=1e7),  # mass of the MBH (massive black hole) in solar masses
        Parameter(symbol="mu", unit="solar masses", lower_limit=1, upper_limit=1e2),  # mass of the CO (compact object) in solar masses
        Parameter(symbol="a", unit="dimensionless", lower_limit=0., upper_limit=1),  # dimensionless spin of the MBH
        Parameter(symbol="p0", unit="meters", lower_limit=10., upper_limit=16.),  # Kepler-orbit parameter: separation
        Parameter(symbol="e0", unit="dimensionless", lower_limit=0.2, upper_limit=0.7),  # Kepler-orbit parameter: eccentricity
        Parameter(symbol="x0", unit="dimensionless", lower_limit=-1., upper_limit=1),  # Kepler-orbit parameter: x_I0=cosI (I is the inclination)
        Parameter(symbol="dist", unit="Gpc", lower_limit=0.1, upper_limit=2),  # luminosity distance
        Parameter(symbol="qS", unit="radian", lower_limit=-np.pi/2, upper_limit=np.pi/2),  # Sky location polar angle in ecliptic coordinates.
        Parameter(symbol="phiS", unit="radian", lower_limit=0., upper_limit=2*np.pi),  # Sky location azimuthal angle in ecliptic coordinates.
        Parameter(symbol="qK", unit="radian", lower_limit=-np.pi/2, upper_limit=np.pi/2),  # Initial BH spin polar angle in ecliptic coordinates.
        Parameter(symbol="phiK", unit="radian", lower_limit=0., upper_limit=2*np.pi),  # Initial BH spin azimuthal angle in ecliptic coordinates.
        Parameter(symbol="Phi_theta0", unit="radian", lower_limit=0., upper_limit=2*np.pi),  # initial polar phase
        Parameter(symbol="Phi_phi0", unit="radian", lower_limit=0., upper_limit=2*np.pi),  # initial azimuthal phase
        Parameter(symbol="Phi_r0", unit="radian", lower_limit=0., upper_limit=2*np.pi),  # initial radial phase
    ]


@dataclass
class ParameterSpace():
    """
    Dataclass to manage the parameter space of a simulation.
    """
    M: float  # mass of the MBH (massive black hole) in solar masses
    mu: float  # mass of the CO (compact object) in solar masses
    a: float  # dimensionless spin of the MBH
    p0: float  # Kepler-orbit parameter: separation
    e0: float  # Kepler-orbit parameter: eccentricity
    x0: float  # Kepler-orbit parameter: x_I0=cosI (I is the inclination)
    dist: float  # luminosity distance
    qS: float  # polar sky-localization (solar system barycenter frame)
    phiS: float  # azimuthal sky-localization (solar system barycenter frame)
    qK: float  # polar orientation of spin angular momentum 
    phiK: float  # azimuthal orientation of spin angular momentum 
    Phi_theta0: float  # polar phase
    Phi_phi0: float  # azimuthal phase
    Phi_r0: float  # radial phase

    parameters_configuration: list[Parameter] = field(default_factory=list)
    parameters_to_marginalize: list[str] = field(default_factory=list)

    def __init__(self, parameters_to_marginalize: list[str] = []):
        self.M = 0.
        self.mu = 0.
        self.a = 0.
        self.p0 = 0.
        self.e0 = 0.
        self.x0 = 0.
        self.dist = 0.
        self.qS = 0.
        self.phiS = 0.
        self.qK = 0.
        self.phiK = 0.
        self.Phi_theta0 = 0.
        self.Phi_phi0 = 0.
        self.Phi_r0 = 0.
        self.parameters_to_marginalize = parameters_to_marginalize
        self.parameters_configuration = parameters_configuration
        self.randomize_parameters()


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

    def _parameters_to_dict(self) -> dict:
        return {
            "M": self.M,
            "mu": self.mu,
            "a": self.a,
            "p0": self.p0,
            "e0": self.e0,
            "x0": self.x0,
            "dist": self.dist,
            "qS": self.qS,
            "phiS": self.phiS,
            "qK": self.qK,
            "phiK": self.phiK,
            "Phi_theta0": self.Phi_theta0,
            "Phi_phi0": self.Phi_phi0,
            "Phi_r0": self.Phi_r0,
        }
