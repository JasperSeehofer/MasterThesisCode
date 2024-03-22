from dataclasses import dataclass
from typing import List
import pandas as pd
import os
import numpy as np
import logging
from master_thesis_code.datamodels.parameter_space import (
    ParameterSpace,
    Parameter,
    uniform,
)
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from master_thesis_code.constants import C, H0
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
)

_LOGGER = logging.getLogger()


@dataclass
class CosmologicalParameter(Parameter):
    fiducial_value: float = 1.0


@dataclass
class Detection:
    d_L: float
    d_L_uncertainty: float
    WL_uncertainty: float


@dataclass
class ParameterSample:
    M: float
    a: float
    dist: float
    mu: float = 10


# setup distribution of MBH spin
min_a, max_a = 0, 0.998
median_a = 0.98
mean_a = median_a
std_a = 0.05
a_distribution = truncnorm(
    (min_a - mean_a) / std_a, (max_a - mean_a) / std_a, loc=mean_a, scale=std_a
)

# coefficients of polynomial fit of dN/dz for different mass bins
merger_distribution_coefficients = {
    0: [
        -94138538.96193656,
        962369408.6975077,
        -3578439441.007358,
        5185569151.868952,
        136402179.6970964,
        -5943613356.609655,
        -3095452047.664805,
        14366833862.29217,
        281549370.2295778,
    ],
    1: [
        -121373875.11104208,
        1445799310.7124693,
        -6789811160.974133,
        15499973013.857445,
        -16287443134.169672,
        4260623123.77606,
        448851767.47119844,
        7196325833.826655,
        392346838.9761119,
    ],
    2: [
        247775058.37853566,
        -3216041245.326129,
        17221325721.312645,
        -49199088856.52833,
        81032523270.16118,
        -76909638891.85504,
        36993487023.340546,
        -3395035047.189672,
        935723800.2450081,
    ],
    3: [
        242799.05947105083,
        89582514.52046189,
        -1091533038.55458,
        5564104340.280578,
        -15711586884.180557,
        26545162214.60701,
        -25878717919.48227,
        11984641274.884602,
        19999528.190069355,
    ],
    4: [
        31727829.680760894,
        -386428090.92766726,
        1923642431.9585557,
        -5016108535.359479,
        7268737080.670669,
        -5669600556.885939,
        1925511804.844566,
        311360824.27473867,
        58391055.04627399,
    ],
}


def polynomial(x, a, b, c, d, e, f, g, h, i) -> float:
    if isinstance(x, (int, float)):
        if x > 3:
            x = 3.0
    else:
        x = np.array([value if value <= 3 else 3.0 for value in x])  # end of fit range

    return (
        a * x**9
        + b * x**8
        + c * x**7
        + d * x**6
        + e * x**5
        + f * x**4
        + g * x**3
        + h * x**2
        + i * x
    )


def MBH_spin_distribution(lower_limit: float, upper_limit: float) -> float:
    """https://iopscience.iop.org/article/10.1088/0004-637X/762/2/68/pdf"""
    return a_distribution.rvs(1)[0]


class Model1CrossCheck:
    """cross check of Model M1 in PHYSICAL REVIEW D 95, 103012 (2017)"""

    parameter_space: ParameterSpace
    emri_rate: int = 294  # 1/yr
    snr_threshold: int = 20

    def __init__(self) -> None:
        self.parameter_space = ParameterSpace()
        self._apply_model_assumptions()

    def _apply_model_assumptions(self) -> None:

        self.parameter_space.M.lower_limit = 10 ** (4.5)
        self.parameter_space.M.upper_limit = 10 ** (6.5)

        self.parameter_space.a.value = 0.98
        self.parameter_space.a.is_fixed = True

        self.parameter_space.mu.value = 10
        self.parameter_space.mu.is_fixed = True

        self.parameter_space.e0.upper_limit = 0.2

        self.parameter_space.dist.upper_limit = 6.8

    def emri_sample_distribution(self, M: float, redshift: float) -> float:
        return self.dN_dz_of_mass(M, redshift) * self.R_emri(M)

    @staticmethod
    def dN_dz_of_mass(mass: float, redshift: float) -> float:
        mass_bin = np.log10(mass)
        if mass_bin < 4.5:
            return polynomial(redshift, *merger_distribution_coefficients[0])
        elif mass_bin < 5.0:
            fraction = (mass_bin - 4.5) / 0.5
            return (1 - fraction) * polynomial(
                redshift, *merger_distribution_coefficients[0]
            ) + fraction * polynomial(redshift, *merger_distribution_coefficients[1])
        elif mass_bin < 5.5:
            fraction = (mass_bin - 5.0) / 0.5
            return (1 - fraction) * polynomial(
                redshift, *merger_distribution_coefficients[1]
            ) + fraction * polynomial(redshift, *merger_distribution_coefficients[2])
        elif mass_bin < 6.0:
            fraction = (mass_bin - 5.5) / 0.5
            return (1 - fraction) * polynomial(
                redshift, *merger_distribution_coefficients[2]
            ) + fraction * polynomial(redshift, *merger_distribution_coefficients[3])
        else:  # mass_bin >= 6.25
            fraction = (mass_bin - 4.5) / 0.5
            fraction = min(fraction, 1.0)
            return (1 - fraction) * polynomial(
                redshift, *merger_distribution_coefficients[3]
            ) + fraction * polynomial(redshift, *merger_distribution_coefficients[4])

    @staticmethod
    def R_emri(M: float) -> float:
        if M < 1.2e5:
            return 10 ** ((1.02445) * np.log10(M / 1.2e5) + np.log10(33.1))
        elif M < 2.5e5:
            return 10 ** ((0.4689) * np.log10(M / 2.5e5) + np.log10(46.7))
        else:
            return 10 ** ((-0.2475) * np.log10(M / 2.9e7) + np.log10(14.4))

    def get_parameter_samples(self, n_samples: int) -> List[ParameterSample]:
        pass  # TBD

    def visualize_emri_distribution(self) -> None:
        # ensure directory is given
        figures_directory = f"saved_figures/cosmological_model/"
        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)

        masses = np.logspace(4, 7, 100)
        redshifts = np.linspace(0, 5, 1000)
        # EMRI rate
        plt.figure(figsize=(10, 6))
        plt.plot(masses, [self.R_emri(mass) for mass in masses])
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("MBH mass in solar masses")
        plt.ylabel("EMRI rate R in 1/Gyr")
        plt.savefig(f"{figures_directory}emri_rate_R.png")
        plt.close()

        # plot dN/dz for different mass bins
        plt.figure(figsize=(10, 6))
        for mass_bin, coefficients in merger_distribution_coefficients.items():
            plt.plot(
                redshifts, polynomial(redshifts, *coefficients), label=f"{mass_bin}"
            )
        plt.yscale("log")
        plt.xlabel("redshift")
        plt.ylabel("dN/dz")
        plt.legend()
        plt.savefig(f"{figures_directory}dN_dz_mass_bins.png")
        plt.close()

        # plot EMRI distribution
        redshifts, masses = np.meshgrid(redshifts, masses)
        dN_dz_distribution = np.vectorize(self.dN_dz_of_mass)(masses, redshifts)
        distribution = np.vectorize(self.emri_sample_distribution)(masses, redshifts)

        plt.contourf(redshifts, masses, dN_dz_distribution, cmap="viridis")
        plt.colorbar()
        plt.yscale("log")
        plt.xlabel("redshift")
        plt.ylabel("mass")
        plt.savefig(f"{figures_directory}dN_dz_distribution.png")
        plt.close()

        # Create a contour plot
        plt.contourf(redshifts, masses, distribution, cmap="viridis")
        plt.colorbar()
        plt.yscale("log")
        plt.xlabel("redshift")
        plt.ylabel("mass")
        plt.savefig(f"{figures_directory}emri_distribution.png")
        plt.close()


class LamCDMScenario:
    """https://arxiv.org/pdf/2102.01708.pdf"""

    h: CosmologicalParameter
    Omega_m: CosmologicalParameter
    w_0: float = -1.0
    w_a: float = 0.0

    def __init__(self) -> None:
        self.h = CosmologicalParameter(
            symbol="h",
            upper_limit=0.86,
            lower_limit=0.6,
            unit="s*Mpc/km",
            randomize_by_distribution=uniform,
            fiducial_value=0.73,
        )
        self.Omega_m = CosmologicalParameter(
            symbol="Omega_m",
            upper_limit=0.04,
            lower_limit=0.5,
            unit="s*Mpc/km",
            randomize_by_distribution=uniform,
            fiducial_value=0.25,
        )


class DarkEnergyScenario:
    w_0: CosmologicalParameter
    w_a: CosmologicalParameter
    h: float = 0.73
    Omega_m: float = 0.25
    Omega_DE: float = 0.75

    def __init__(self) -> None:
        self.w_0 = CosmologicalParameter(
            symbol="w_0",
            unit="xxx",
            lower_limit=-3.0,
            upper_limit=-0.3,
            randomize_by_distribution=uniform,
            fiducial_value=-1.0,
        )
        self.w_a = CosmologicalParameter(
            symbol="w_a",
            unit="xxx",
            lower_limit=-1.0,
            upper_limit=1.0,
            randomize_by_distribution=uniform,
            fiducial_value=0.0,
        )

    def de_equation(self, z) -> float:
        return self.w_0 + z / (1 + z) / self.w_a


class BayesianStatistics:
    cramer_rao_bounds: pd.DataFrame
    cosmological_model: LamCDMScenario = LamCDMScenario()

    def __init__(self) -> None:
        self.cramer_rao_bounds = pd.read_csv(
            "./simulations/cramer_rao_bounds_unbiased.csv"
        )

    def evaluate(self, galaxy_catalog: GalaxyCatalogueHandler) -> None:
        _LOGGER.info("Evaluating Bayesian statistics...")
        posteriors = []
        posteriors_with_bh_mass = []
        h_samples = np.linspace(
            self.cosmological_model.h.lower_limit,
            self.cosmological_model.h.upper_limit,
            20,
        )
        Omega_m = self.cosmological_model.Omega_m.fiducial_value
        Omega_DE = 1 - Omega_m
        w_0 = self.cosmological_model.w_0
        w_a = self.cosmological_model.w_a
        for h in h_samples:
            posterior, posterior_with_bh_mass = self.p_D(
                h=h,
                Omega_m=Omega_m,
                Omega_DE=Omega_DE,
                w_0=w_0,
                w_a=w_a,
                galaxy_catalog=galaxy_catalog,
            )
            _LOGGER.info(f"posterior comupted for h = {h}")
            posteriors.append(posterior)
            posteriors_with_bh_mass.append(posterior_with_bh_mass)

        fig = plt.figure(figsize=(16, 9))
        plt.scatter(h_samples, posteriors, label="without BH mass")
        plt.scatter(h_samples, posteriors_with_bh_mass, label="with BH mass")
        plt.xlabel("Hubble constant h")
        plt.ylabel("Posterior")
        plt.legend()
        plt.savefig("saved_figures/bayesian_statistics.png")
        plt.close()

    def p_D(
        self,
        h: float,
        Omega_m: float,
        Omega_DE: float,
        w_0: float,
        w_a: float,
        galaxy_catalog: GalaxyCatalogueHandler,
    ) -> tuple[float, float]:
        result = 1
        result_with_bh_mass = 1
        for index, detection in self.cramer_rao_bounds.iterrows():
            parameters = {
                "dist": detection["dist"],
                "dist_error": np.sqrt(detection["delta_dist_delta_dist"]),
                "phi": detection["phiS"],
                "phi_error": np.sqrt(detection["delta_phiS_delta_phiS"]),
                "theta": detection["qS"],
                "theta_error": np.sqrt(detection["delta_qS_delta_qS"]),
                "M_z": detection["M"],
                "M_z_error": np.sqrt(detection["delta_M_delta_M"]),
            }

            if not self.use_detection():
                continue

            possible_hosts, possible_hosts_with_bh_mass = (
                galaxy_catalog.get_possible_hosts(**parameters)
            )
            _LOGGER.info(
                f"possible hosts found {len(possible_hosts)}/{len(possible_hosts_with_bh_mass)}..."
            )
            result *= self.p_Di(
                detection=detection,
                possible_host_galaxies=possible_hosts,
                h=h,
                Omega_m=Omega_m,
                Omega_DE=Omega_DE,
                w_0=w_0,
                w_a=w_a,
            )
            _LOGGER.info("posterior computed for detection without bh mass...")
            result_with_bh_mass *= self.p_Di(
                detection=detection,
                possible_host_galaxies=possible_hosts_with_bh_mass,
                evaluate_with_bh_mass=True,
                h=h,
                Omega_m=Omega_m,
                Omega_DE=Omega_DE,
                w_0=w_0,
                w_a=w_a,
            )
            _LOGGER.info("posterior computed for detection...")
        return result, result_with_bh_mass

    def p_Di(
        self,
        h: float,
        Omega_m: float,
        Omega_DE: float,
        w_0: float,
        w_a: float,
        detection: pd.Series,
        possible_host_galaxies: List[HostGalaxy],
        evaluate_with_bh_mass: bool = False,
    ) -> float:
        z_gws = np.linspace(0, 10, 100)  # TODO
        integrant = np.zeros(len(z_gws))
        d_L = detection["dist"]
        d_L_uncertainty = detection["delta_dist_delta_dist"]

        weight: callable = self.weight
        if evaluate_with_bh_mass:
            weight: callable = self.weight_with_bh_mass

        for possible_host in possible_host_galaxies:

            """WL_uncertainty = (
                d_L * 0.066 * (1 - (1 + possible_host.z) ** (-0.25) / 0.25) ** (1.8)
            )"""  # TODO check if correct
            integrant += [
                (
                    weight(possible_host)
                    / possible_host.z_error
                    / np.sqrt(d_L_uncertainty**2)
                    * np.exp(
                        -1
                        / 2
                        * (
                            (possible_host.z - z_gw) ** 2 / possible_host.z_error**2
                            + (
                                d_L
                                - self.d_zgw(
                                    z_gw=z_gw,
                                    h=h,
                                    Omega_m=Omega_m,
                                    Omega_DE=Omega_DE,
                                    w_0=w_0,
                                    w_a=w_a,
                                )
                            )
                            ** 2
                            / (d_L_uncertainty**2)
                        )
                    )
                )
                for z_gw in z_gws
            ]

        return np.trapz(integrant, z_gws) / 2 / np.pi

    def d_zgw(
        self,
        z_gw: float,
        h: float,
        Omega_m: float,
        Omega_DE: float,
        w_0: float,
        w_a: float,
    ) -> float:
        zs = np.linspace(0, z_gw, 100)
        return (
            C
            * (1 + z_gw)
            / h
            * np.trapz(
                1 / self.E(z=zs, Omega_m=Omega_m, Omega_DE=Omega_DE, w_0=w_0, w_a=w_a),
                zs,
            )
        )

    def E(
        self, z: float, Omega_m: float, Omega_DE: float, w_0: float, w_a: float
    ) -> float:
        return np.sqrt(Omega_m * (1 + z) ** 3 + Omega_DE * self.g(z, w_0=w_0, w_a=w_a))

    def g(self, z: float, w_0: float, w_a: float) -> float:
        return (1 + z) ** (3 * (1 + w_0 + w_a)) * np.exp(-3.0 * w_a * z / (1 + z))

    def weight(self, possible_host: HostGalaxy) -> float:
        return 1.0  # TBD

    def weight_with_bh_mass(self, possible_host: HostGalaxy) -> float:
        return 1.0  # TBD

    def error_volume(self) -> float:
        return 1.0

    def use_detection(
        self,
        d_L: float,
        d_L_error: float,
        phi: float,
        phi_error: float,
        theta: float,
        theta_error: float,
    ) -> bool:
        sky_localization_uncertainty = self._sky_localization_uncertainty(
            phi_error=phi_error, theta=theta, theta_error=theta_error
        )
        distance_relative_error = d_L_error / d_L

        if (distance_relative_error < 0.1) and (sky_localization_uncertainty < 0.2):
            return True
        return False

    @staticmethod
    def _sky_localization_uncertainty(
        phi_error: float, theta: float, theta_error: float
    ) -> float:
        return (
            2 * np.pi * np.abs(np.sin(theta)) * np.sqrt(phi_error**2 * theta_error**2)
        )
