from dataclasses import dataclass, field
from master_thesis_code.constants import INFINITY
import numpy as np
import random
from master_thesis_code.galaxy_catalogue.handler import HostGalaxy


def uniform(lower_limit: float, upper_limit: float) -> float:
    return np.random.uniform(lower_limit, upper_limit)


def log_uniform(lower_limit: float, upper_limit: float) -> float:
    lower_limit = np.log10(lower_limit)
    upper_limit = np.log10(upper_limit)
    uniform_log = uniform(lower_limit, upper_limit)
    return 10**uniform_log


def polar_angle_distribution(lower_limit: float, upper_limit: float) -> float:
    return np.arccos(np.random.uniform(-1.0, 1.0))


@dataclass
class Parameter:
    """Main class for parameters."""

    symbol: str
    unit: str
    lower_limit: float
    upper_limit: float
    value: float = 0.0
    derivative_epsilon: float = 1e-6
    is_fixed: bool = False
    randomize_by_distribution: callable = uniform


@dataclass
class ParameterSpace:
    """
    Dataclass to manage the parameter space of a simulation.
    """

    M: Parameter = Parameter(
        symbol="M", unit="solar masses", lower_limit=1e4, upper_limit=1e7, randomize_by_distribution=log_uniform
    )  # mass of the MBH (massive black hole) in solar masses

    mu: Parameter = Parameter(
        symbol="mu", unit="solar masses", lower_limit=1, upper_limit=1e2
    )  # mass of the CO (compact object) in solar masses
    a: Parameter = Parameter(
        symbol="a", unit="dimensionless", lower_limit=0.0, upper_limit=1
    )  # dimensionless spin of the MBH
    p0: Parameter = Parameter(
        symbol="p0", unit="meters", lower_limit=10.0, upper_limit=16.0
    )  # Kepler-orbit parameter: separation
    e0: Parameter = Parameter(
        symbol="e0", unit="dimensionless", lower_limit=0.05, upper_limit=0.7
    )  # Kepler-orbit parameter: eccentricity
    x0: Parameter = Parameter(
        symbol="x0", unit="dimensionless", lower_limit=-1.0, upper_limit=1.0
    )  # Kepler-orbit parameter: x_I0=cosI (I is the inclination)
    dist: Parameter = Parameter(
        symbol="dist", unit="Gpc", lower_limit=0.1, upper_limit=7
    )  # luminosity distance
    qS: Parameter = Parameter(
        symbol="qS",
        unit="radian",
        lower_limit=0.0,
        upper_limit=np.pi,
        randomize_by_distribution=polar_angle_distribution,
    )  # Sky location polar angle in ecliptic coordinates.
    phiS: Parameter = Parameter(
        symbol="phiS", unit="radian", lower_limit=0.0, upper_limit=2 * np.pi
    )  # Sky location azimuthal angle in ecliptic coordinates.
    qK: Parameter = Parameter(
        symbol="qK",
        unit="radian",
        lower_limit=0.0,
        upper_limit=np.pi,
        randomize_by_distribution=polar_angle_distribution,
    )  # Initial BH spin polar angle in ecliptic coordinates.
    phiK: Parameter = Parameter(
        symbol="phiK", unit="radian", lower_limit=0.0, upper_limit=2 * np.pi
    )  # Initial BH spin azimuthal angle in ecliptic coordinates.
    Phi_phi0: Parameter = Parameter(
        symbol="Phi_phi0", unit="radian", lower_limit=0.0, upper_limit=2 * np.pi
    )  # initial azimuthal phase
    Phi_theta0: Parameter = Parameter(
        symbol="Phi_theta0", unit="radian", lower_limit=0.0, upper_limit=2 * np.pi
    )  # initial polar phase
    Phi_r0: Parameter = Parameter(
        symbol="Phi_r0", unit="radian", lower_limit=0.0, upper_limit=2 * np.pi
    )  # initial radial phase

    def randomize_parameter(self, parameter: Parameter) -> None:
        parameter.value = parameter.randomize_by_distribution(
            parameter.lower_limit, parameter.upper_limit
        )
        setattr(self, parameter.symbol, parameter)

    def randomize_parameters(self) -> None:
        for parameter in vars(self).values():
            if isinstance(parameter, Parameter) and not parameter.is_fixed:
                self.randomize_parameter(parameter=parameter)

    def set_host_galaxy_parameters(self, host_galaxy: HostGalaxy) -> None:
        self.M.value = host_galaxy.M
        self.phiS.value = host_galaxy.phiS
        self.qS.value = host_galaxy.qS
        self.dist.value = host_galaxy.dist

    def _parameters_to_dict(self) -> dict:
        return {
            "M": self.M.value,
            "mu": self.mu.value,
            "a": self.a.value,
            "p0": self.p0.value,
            "e0": self.e0.value,
            "x0": self.x0.value,
            "dist": self.dist.value,
            "qS": self.qS.value,
            "phiS": self.phiS.value,
            "qK": self.qK.value,
            "phiK": self.phiK.value,
            "Phi_phi0": self.Phi_phi0.value,
            "Phi_theta0": self.Phi_theta0.value,
            "Phi_r0": self.Phi_r0.value,
        }
