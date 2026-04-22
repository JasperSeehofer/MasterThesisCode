from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from master_thesis_code.galaxy_catalogue.handler import HostGalaxy
from master_thesis_code.physical_relations import dist


def uniform(lower_limit: float, upper_limit: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(lower_limit, upper_limit))


def log_uniform(lower_limit: float, upper_limit: float, rng: np.random.Generator) -> float:
    lower_limit = np.log10(lower_limit)
    upper_limit = np.log10(upper_limit)
    uniform_log = uniform(lower_limit, upper_limit, rng)
    return float(10**uniform_log)


def polar_angle_distribution(
    lower_limit: float, upper_limit: float, rng: np.random.Generator
) -> float:
    return float(np.arccos(rng.uniform(-1.0, 1.0)))


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
    randomize_by_distribution: Callable[[float, float, np.random.Generator], float] = uniform


@dataclass
class ParameterSpace:
    """
    Dataclass to manage the parameter space of a simulation.
    """

    # Per-parameter derivative_epsilon: Vallisneri (2008) arXiv:gr-qc/0703086 Eq. (A11)
    # Optimal step size for 5-point stencil (p=4): h* ≈ ε_machine^(1/4) × |x| ≈ 3.3e-4 × |x|
    # Each epsilon is chosen to be ~3e-4 × (representative parameter value).
    # Vallisneri (2008) arXiv:gr-qc/0703086 Eq. (A11) — per-param epsilon

    M: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="M",
            unit="solar masses",
            lower_limit=1e4,
            upper_limit=1e7,
            randomize_by_distribution=log_uniform,
            derivative_epsilon=1.0,  # ~3e-4 × 3e3 SM (log-uniform midpoint ~3e3 SM)
        )
    )  # mass of the MBH (massive black hole) in solar masses

    mu: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="mu",
            unit="solar masses",
            lower_limit=1,
            upper_limit=1e2,
            derivative_epsilon=0.01,  # ~3e-4 × 30 SM (midpoint ~30 SM)
        )
    )  # mass of the CO (compact object) in solar masses
    a: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="a",
            unit="dimensionless",
            lower_limit=0.0,
            upper_limit=1,
            derivative_epsilon=1e-3,  # ~3e-4 × 0.5 (dimensionless [0, 1])
        )
    )  # dimensionless spin of the MBH
    p0: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="p0",
            unit="meters",
            lower_limit=10.0,
            upper_limit=16.0,
            derivative_epsilon=1e-3,  # ~3e-4 × 13 (midpoint; dimensionless semi-latus rectum)
        )
    )  # Kepler-orbit parameter: separation
    e0: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="e0",
            unit="dimensionless",
            lower_limit=0.05,
            upper_limit=0.7,
            derivative_epsilon=1e-4,  # ~3e-4 × 0.35 ≈ 1e-4 (dimensionless [0.05, 0.7])
        )
    )  # Kepler-orbit parameter: eccentricity
    x0: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="x0",
            unit="dimensionless",
            lower_limit=-1.0,
            upper_limit=1.0,
            derivative_epsilon=1e-4,  # symmetric around 0; use half-range scale 1e-4
        )
    )  # Kepler-orbit parameter: x_I0=cosI (I is the inclination)
    luminosity_distance: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="luminosity_distance",
            unit="Gpc",
            lower_limit=0.0,
            upper_limit=7,
            derivative_epsilon=1e-4,  # ~3e-4 × 1 Gpc ≈ 3e-4; use 1e-4 Gpc (= 0.1 Mpc)
        )
    )  # luminosity distance
    qS: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="qS",
            unit="radian",
            lower_limit=0.0,
            upper_limit=np.pi,
            randomize_by_distribution=polar_angle_distribution,
            derivative_epsilon=1e-4,  # ~3e-4 × π/2 ≈ 5e-4; use 1e-4 rad
        )
    )  # Sky location polar angle in ecliptic coordinates.
    phiS: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="phiS",
            unit="radian",
            lower_limit=0.0,
            upper_limit=2 * np.pi,
            derivative_epsilon=1e-4,  # ~3e-4 × π ≈ 1e-3; use 1e-4 rad
        )
    )  # Sky location azimuthal angle in ecliptic coordinates.
    qK: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="qK",
            unit="radian",
            lower_limit=0.0,
            upper_limit=np.pi,
            randomize_by_distribution=polar_angle_distribution,
            derivative_epsilon=1e-4,  # same as qS
        )
    )  # Initial BH spin polar angle in ecliptic coordinates.
    phiK: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="phiK",
            unit="radian",
            lower_limit=0.0,
            upper_limit=2 * np.pi,
            derivative_epsilon=1e-4,  # same as phiS
        )
    )  # Initial BH spin azimuthal angle in ecliptic coordinates.
    Phi_phi0: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="Phi_phi0",
            unit="radian",
            lower_limit=0.0,
            upper_limit=2 * np.pi,
            derivative_epsilon=1e-4,  # ~3e-4 × π ≈ 1e-3; use 1e-4 rad
        )
    )  # initial azimuthal phase
    Phi_theta0: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="Phi_theta0",
            unit="radian",
            lower_limit=0.0,
            upper_limit=2 * np.pi,
            derivative_epsilon=1e-4,  # same as Phi_phi0
        )
    )  # initial polar phase
    Phi_r0: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="Phi_r0",
            unit="radian",
            lower_limit=0.0,
            upper_limit=2 * np.pi,
            derivative_epsilon=1e-4,  # same as Phi_phi0
        )
    )  # initial radial phase

    def randomize_parameter(self, parameter: Parameter, rng: np.random.Generator) -> None:
        parameter.value = parameter.randomize_by_distribution(
            parameter.lower_limit, parameter.upper_limit, rng
        )
        setattr(self, parameter.symbol, parameter)

    def randomize_parameters(self, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        for parameter in vars(self).values():
            if isinstance(parameter, Parameter) and not parameter.is_fixed:
                self.randomize_parameter(parameter=parameter, rng=rng)

    def set_host_galaxy_parameters(self, host_galaxy: HostGalaxy, h: float) -> None:
        self.M.value = host_galaxy.M
        self.phiS.value = host_galaxy.phiS
        self.qS.value = host_galaxy.qS
        # h_inj threaded explicitly per PE-01 (Phase 37); dark siren PE self-consistency at h_inj
        # (Gray et al. 2020, Laghi et al. 2021). h has no default — calling without h raises TypeError (SC-2).
        self.luminosity_distance.value = dist(host_galaxy.z, h=h)  # SC-1: h_inj threaded

    def _parameters_to_dict(self) -> dict:
        return {
            "M": self.M.value,
            "mu": self.mu.value,
            "a": self.a.value,
            "p0": self.p0.value,
            "e0": self.e0.value,
            "x0": self.x0.value,
            "luminosity_distance": self.luminosity_distance.value,
            "qS": self.qS.value,
            "phiS": self.phiS.value,
            "qK": self.qK.value,
            "phiK": self.phiK.value,
            "Phi_phi0": self.Phi_phi0.value,
            "Phi_theta0": self.Phi_theta0.value,
            "Phi_r0": self.Phi_r0.value,
        }
