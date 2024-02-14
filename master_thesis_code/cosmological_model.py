from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np


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
