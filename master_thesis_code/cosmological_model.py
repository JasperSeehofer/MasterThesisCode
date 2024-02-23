from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np
from master_thesis_code.datamodels.parameter_space import ParameterSpace, Parameter


@dataclass
class CosmologicalParameter:
    upper_limit: float
    lower_limit: float
    unit_info: str
    prior_info: str


@dataclass
class PossibleHost:
    redshift: float
    redshift_uncertainty: float


@dataclass
class Detection:
    d_L: float
    d_L_uncertainty: float
    WL_uncertainty: float


class Model1CrossCheck:
    """cross check of Model M1 in PHYSICAL REVIEW D 95, 103012 (2017)"""

    parameter_space: ParameterSpace
    emri_rate: int = 294  # 1/yr
    snr_threshold: int = 20

    def __init__(self) -> None:
        self.parameter_space = ParameterSpace()
        self._apply_model_assumptions()

    def _apply_model_assumptions(self) -> None:

        self.parameter_space.M.lower_limit = 10**4
        self.parameter_space.M.upper_limit = 10**7

        self.parameter_space.a.value = 0.98
        self.parameter_space.a.is_fixed = True

        self.parameter_space.mu.value = 10
        self.parameter_space.mu.is_fixed = True

        self.parameter_space.e0.upper_limit = 0.2

        self.parameter_space.dist.upper_limit = 4.5

    def MBH_distribution(mass: float) -> float:
        return 0.005 * (mass / 3e6) ** (-0.3)  # 1/ Mpc^-3


class LamCDMScenario(CosmologicalParameter):
    """https://arxiv.org/pdf/2102.01708.pdf"""

    h: CosmologicalParameter
    Omega_m: CosmologicalParameter

    def __init__(self) -> None:
        self.h = CosmologicalParameter(
            upper_limit=0.86, lower_limit=0.6, unit_info="s*Mpc/km", prior="uniform"
        )
        self.Omega_m = CosmologicalParameter(
            upper_limit=0.04, lower_limit=0.5, unit_info="s*Mpc/km", prior="uniform"
        )


class BayesianStatistics:

    def p_Di(
        self,
        z_gw: np.array,
        detection: Detection,
        possible_host_galaxies: List[PossibleHost],
    ):
        integrant = 0.0
        for possible_host in possible_host_galaxies:
            integrant += (
                self.weight(possible_host)
                / possible_host.redshift_uncertainty
                / np.sqrt(
                    detection.d_L_uncertainty**2 + possible_host.WL_uncertainty**2
                )
                * np.exp(
                    -1
                    / 2
                    * (
                        (possible_host.redshift - z_gw) ** 2
                        / possible_host.redshift_uncertainty**2
                        + (detection.d_L - self.d_zgw(z_gw)) ** 2
                        / (detection.d_L_uncertainty**2 + detection.WL_uncertainty**2)
                    )
                )
            )

        return np.trapz(integrant, z_gw) / 2 / np.pi

    def d_zgw(self, z_gw: float) -> float:
        return 1  # Eq. 2.5

    def sum_in_p_Di(possible_host_galaxies: List[PossibleHost]) -> float:
        return 1

    def weight(possible_host: PossibleHost) -> float:
        return 1.0  # TBD
