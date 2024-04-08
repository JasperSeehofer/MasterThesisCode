from dataclasses import dataclass
from typing import List, Dict
import json
import pandas as pd
import os
import math
import numpy as np
import logging
import matplotlib.pyplot as plt
import time
from scipy.stats import multivariate_normal, truncnorm
from statistics import NormalDist
import multiprocessing as mp
from master_thesis_code.datamodels.parameter_space import (
    ParameterSpace,
    Parameter,
    uniform,
)

from master_thesis_code.constants import C, H0
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
)
from master_thesis_code.physical_relations import (
    dist,
    convert_true_mass_to_redshifted_mass_with_distance,
    get_redshift_outer_bounds,
    dist_to_redshift,
)

_LOGGER = logging.getLogger()

DEFAULT_GALAXY_Z_ERROR = 0.0015
GALAXY_WEIGHTS = "galaxy_weights"


@dataclass
class CosmologicalParameter(Parameter):
    fiducial_value: float = 1.0


@dataclass
class Detection:
    d_L: float
    d_L_uncertainty: float
    phi: float
    phi_error: float
    theta: float
    theta_error: float
    M: float
    M_uncertainty: float
    WL_uncertainty: float = 0.0

    def __init__(self, parameters: pd.Series) -> None:
        self.d_L = parameters["dist"]
        self.d_L_uncertainty = np.sqrt(parameters["delta_dist_delta_dist"])
        self.phi = parameters["phiS"]
        self.phi_error = np.sqrt(parameters["delta_phiS_delta_phiS"])
        self.theta = parameters["qS"]
        self.theta_error = np.sqrt(parameters["delta_qS_delta_qS"])
        self.M = parameters["M"]
        self.M_uncertainty = np.sqrt(parameters["delta_M_delta_M"])
        self.theta_phi_covariance = parameters["delta_phiS_delta_qS"]
        self.M_phi_covariance = parameters["delta_phiS_delta_M"]
        self.M_theta_covariance = parameters["delta_qS_delta_M"]


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
    detection: Detection
    cosmological_model: LamCDMScenario
    h: float
    Omega_m: float
    Omega_DE: float
    w_0: float
    w_a: float
    h_values: List = []
    posterior_data: Dict[int, List[float]] = {}
    posterior_data_with_bh_mass: Dict[int, List[float]] = {}

    def __init__(self) -> None:
        self.full_cramer_rao_bounds = pd.read_csv(
            "./simulations/cramer_rao_bounds_unbiased.csv"
        )
        _LOGGER.debug(f"cramer_bounds: {self.full_cramer_rao_bounds.describe()}")
        self.cramer_rao_bounds = self.full_cramer_rao_bounds.sample(5)
        _LOGGER.info(f"Loaded {len(self.cramer_rao_bounds)} detections...")
        self.cosmological_model = LamCDMScenario()
        self.h = self.cosmological_model.h.fiducial_value
        self.Omega_m = self.cosmological_model.Omega_m.fiducial_value
        self.Omega_DE = 1 - self.Omega_m
        self.w_0 = self.cosmological_model.w_0
        self.w_a = self.cosmological_model.w_a

    def visualize(self, galaxy_catalog: GalaxyCatalogueHandler) -> None:
        with open("simulations/bayesian_statistics_posteriors.json", "r") as file:
            self.posterior_data = json.load(file)
        with open(
            "simulations/bayesian_statistics_posteriors_with_bh_mass.json", "r"
        ) as file:
            self.posterior_data_with_bh_mass = json.load(file)

        h_samples = np.linspace(
            self.cosmological_model.h.lower_limit,
            self.cosmological_model.h.upper_limit,
            20,
        )
        posteriors = np.ones(len(h_samples))
        posteriors_with_bh_mass = np.ones(len(h_samples))

        plt.figure(figsize=(16, 9))
        for detection_index, posterior in self.posterior_data.items():
            if len(posterior) < len(h_samples):
                continue
            plt.plot(
                h_samples,
                posterior / np.max(posterior),
                label=f"detection: {detection_index}",
            )
            plt.xlabel("Hubble constant h")
            plt.ylabel("Posterior")
            plt.legend()
        plt.savefig(f"saved_figures/bayesian_statistics_event_posteriors.png")
        plt.close()

        plt.figure(figsize=(16, 9))
        for detection_index, posterior in self.posterior_data_with_bh_mass.items():
            if len(posterior) < len(h_samples):
                continue
            plt.plot(
                h_samples,
                posterior / np.max(posterior),
                label=f"detection {detection_index}",
            )
            plt.xlabel("Hubble constant h")
            plt.ylabel("Posterior")
            plt.legend()
        plt.savefig(
            f"saved_figures/bayesian_statistics_event_posteriors_with_bh_mass.png"
        )
        plt.close()

        for index, posterior in self.posterior_data.items():
            if len(posterior) < len(h_samples):
                continue
            posteriors *= np.array(posterior)
        for index, posterior in self.posterior_data_with_bh_mass.items():
            if len(posterior) < len(h_samples):
                continue
            posteriors_with_bh_mass *= np.array(posterior)

        posteriors = posteriors / np.max(posteriors)
        posteriors_with_bh_mass = posteriors_with_bh_mass / np.max(
            posteriors_with_bh_mass
        )

        fig = plt.figure(figsize=(16, 9))
        plt.scatter(h_samples, posteriors, label="without BH mass")
        plt.scatter(h_samples, posteriors_with_bh_mass, label="with BH mass")
        plt.xlabel("Hubble constant h")
        plt.ylabel("Posterior")
        plt.legend()
        plt.savefig("saved_figures/bayesian_statistics.png")
        plt.close()

        self.visualize_galaxy_weights(galaxy_catalog)

    def visualize_galaxy_weights(self, galaxy_catalog: GalaxyCatalogueHandler) -> None:
        _LOGGER.info("Visualizing galaxy weights...")
        # visualize galaxy weights
        weight_data = self.posterior_data_with_bh_mass[GALAXY_WEIGHTS]

        for detection_index, host_galaxy_weights in weight_data.items():
            _LOGGER.debug(f"Visualizing galaxy weights for detection {detection_index}")
            # sort by weight and use first 1000 galaxies
            detection = Detection(
                self.full_cramer_rao_bounds.iloc[int(detection_index)]
            )
            host_galaxy_weights = host_galaxy_weights[0]
            _LOGGER.debug(f"found {len(host_galaxy_weights)} host galaxies...")

            host_galaxies = [
                galaxy_catalog.get_host_galaxy_by_index(int(index))
                for index, _ in host_galaxy_weights
            ]
            host_galaxies_phi = np.array([galaxy.phiS for galaxy in host_galaxies])
            host_galaxies_theta = np.array([galaxy.qS for galaxy in host_galaxies])
            unweighted_likelihoods = np.array(
                [weights[0] for _, weights in host_galaxy_weights]
            )
            weights = np.array([weights[1] for _, weights in host_galaxy_weights])
            weights_bh_mass = np.array(
                [weights[2] for _, weights in host_galaxy_weights]
            )

            # create 2D plot for phi and qS
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.set_title(
                f"Galaxy skylocalization weight for detection {detection_index}"
            )
            # plot lines for detection (true values)
            ax.axvline(detection.phi, color="r", linestyle="--")
            ax.axhline(detection.theta, color="r", linestyle="--")
            ax.set_xlabel("phiS")
            ax.set_ylabel("qS")
            scatter = ax.scatter(
                host_galaxies_phi,
                host_galaxies_theta,
                c=weights,
                cmap="viridis",
            )
            plt.colorbar(scatter, label="weight")
            plt.savefig(
                f"saved_figures/galaxy_weights_detection_{detection_index}_phi_theta.png",
                dpi=300,
            )
            plt.close()

            # create 2D plot for phi and qS with colormap for mass weight
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.set_title(f"Galaxy mass weight for detection {detection_index}")
            # plot lines for detection (true values)
            ax.axvline(detection.phi, color="r", linestyle="--")
            ax.axhline(detection.theta, color="r", linestyle="--")
            ax.set_xlabel("phiS")
            ax.set_ylabel("qS")

            scatter = ax.scatter(
                host_galaxies_phi,
                host_galaxies_theta,
                c=weights_bh_mass,
                cmap="viridis",
            )
            plt.colorbar(scatter, label="mass weight")
            plt.savefig(
                f"saved_figures/galaxy_mass_weights_detection_{detection_index}_phi_theta.png",
                dpi=300,
            )
            plt.close()

            # create plot for M weights
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(111)
            ax.set_title(f"Galaxy mass weights for detection {detection_index}")
            # plot lines for detection (true values)
            ax.axvline(detection.M, color="r", linestyle="--")
            ax.set_xlabel("M")
            ax.set_ylabel("mass weight")
            ax.scatter(
                [galaxy.M for galaxy in host_galaxies],
                weights_bh_mass,
            )
            plt.savefig(
                f"saved_figures/galaxy_mass_weights_detection_{detection_index}_mass.png",
                dpi=300,
            )
            plt.close()

            # plot weighted likelihood
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(111)
            ax.set_title(f"Galaxy weighted likelihood for detection {detection_index}")
            # plot lines for detection (true values)
            ax.vlines(
                dist_to_redshift(detection.d_L),
                0,
                np.max(unweighted_likelihoods * weights),
                color="r",
                linestyle="--",
            )
            ax.set_xlabel("z")
            ax.set_ylabel("likelihood")
            ax.scatter(
                [galaxy.z for galaxy in host_galaxies],
                unweighted_likelihoods * weights,
                label="likelihood",
            )
            plt.savefig(
                f"saved_figures/galaxy_weighted_likelihood_detection_{detection_index}.png",
                dpi=300,
            )
            plt.close()

            # create 3d plot for phi, qS, M with colormap for product of weight and mass weight
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_title(f"Galaxy likelihood for detection {detection_index}")
            # plot lines for detection (true values)

            ax.set_xlabel("phiS")
            ax.set_ylabel("qS")
            ax.set_zlabel("M")

            scatter = ax.scatter(
                host_galaxies_phi,
                host_galaxies_theta,
                [galaxy.M for galaxy in host_galaxies],
                c=weights * weights_bh_mass,
                cmap="viridis",
            )
            ax.scatter(detection.phi, detection.theta, detection.M, color="r")
            plt.colorbar(scatter, label="weighted likelihood")
            plt.savefig(
                f"saved_figures/galaxy_weights_detection_{detection_index}_phi_theta_mass.png",
                dpi=300,
            )
            plt.close()

    def evaluate(self, galaxy_catalog: GalaxyCatalogueHandler, h_value: float) -> None:
        _LOGGER.info(f"Computing posteriors for h = {h_value}...")
        if (h_value < self.cosmological_model.h.lower_limit) or (
            h_value > self.cosmological_model.h.upper_limit
        ):
            raise ValueError("Hubble constant out of bounds.")

        self.h = h_value
        _LOGGER.debug(
            f"Found {len(os.sched_getaffinity(0))} / {os.cpu_count()} (available / system) cpus."
        )
        cpu_count = os.cpu_count()
        if len(os.sched_getaffinity(0)) < cpu_count:
            try:
                os.sched_setaffinity(0, range(cpu_count))
            except OSError:
                _LOGGER.info("Could not set affinity")
        _LOGGER.debug(
            f"After trying to set affinity available cpus: {len(os.sched_getaffinity(0))}"
        )
        with mp.get_context("spawn").Pool(len(os.sched_getaffinity(0)) - 4) as pool:
            self.p_D(
                galaxy_catalog=galaxy_catalog,
                pool=pool,
            )
        _LOGGER.info(f"posteriors comupted for h = {self.h}")

        if not os.path.isdir("simulations/posteriors"):
            os.makedirs("simulations/posteriors")
        if not os.path.isdir("simulations/posteriors_with_bh_mass"):
            os.makedirs("simulations/posteriors_with_bh_mass")

        with open(
            f"simulations/posteriors/h_{str(np.round(self.h,3)).replace('.', '_')}.json",
            "r",
        ) as file:
            try:
                posteriors_existing_data = json.load(file)
            except json.decoder.JSONDecodeError:
                posteriors_existing_data = {}
            
        with open(
            f"simulations/posteriors/h_{str(np.round(self.h,3)).replace('.', '_')}.json",
            "w",
        ) as file:
            # update existing data
            data = posteriors_existing_data | self.posterior_data
            json.dump(data | {"h": self.h}, file)

        with open(
            f"simulations/posteriors_with_bh_mass/h_{str(np.round(self.h,3)).replace('.', '_')}.json",
            "r",
        ) as file:
            try:
                posteriors_with_bh_mass_existing_data = json.load(file)
            except json.decoder.JSONDecodeError:
                posteriors_existing_data = {}
        with open(
            f"simulations/posteriors_with_bh_mass/h_{str(np.round(self.h,3)).replace('.', '_')}.json",
            "w",
        ) as file:
            # update existing data
            
            data = posteriors_with_bh_mass_existing_data | self.posterior_data_with_bh_mass
            json.dump(data | {"h": self.h}, file)

    def p_D(
        self,
        galaxy_catalog: GalaxyCatalogueHandler,
        pool: mp.Pool,
    ) -> None:
        count = 0
        self.posterior_data_with_bh_mass[GALAXY_WEIGHTS] = {}
        for index, detection in self.cramer_rao_bounds.iterrows():
            _LOGGER.info(f"Progess: detections: {count}/{len(self.cramer_rao_bounds)}")
            count += 1
            try:
                self.posterior_data[index]
            except KeyError:
                self.posterior_data[index] = []
                self.posterior_data_with_bh_mass[index] = []
            detection["M"] = convert_true_mass_to_redshifted_mass_with_distance(
                detection["M"], detection["dist"]
            )
            self.detection = Detection(detection)
            z_min, z_max = get_redshift_outer_bounds(
                distance=self.detection.d_L,
                distance_error=self.detection.d_L_uncertainty,
                h_min=self.cosmological_model.h.lower_limit,
                h_max=self.cosmological_model.h.upper_limit,
                Omega_m_min=self.cosmological_model.Omega_m.lower_limit,
                Omega_m_max=self.cosmological_model.Omega_m.upper_limit,
            )

            if not self.use_detection():
                _LOGGER.debug("detection skipped...")
                continue

            possible_hosts = galaxy_catalog.get_possible_hosts(
                z_min=z_min,
                z_max=z_max,
                phi=self.detection.phi,
                phi_error=self.detection.phi_error,
                theta=self.detection.theta,
                theta_error=self.detection.theta_error,
                M_z=self.detection.M,
                M_z_error=self.detection.M_uncertainty,
            )

            if possible_hosts is None:
                _LOGGER.debug("no possible hosts found...")
                continue
            possible_hosts, possible_hosts_with_bh_mass = possible_hosts
            _LOGGER.info(
                f"possible hosts found {len(possible_hosts)}/{len(possible_hosts_with_bh_mass)}..."
            )
            event_likelihood, event_likelihood_with_bh_mass = self.p_Di(
                possible_host_galaxies=possible_hosts,
                possible_host_galaxies_with_bh_mass=possible_hosts_with_bh_mass,
                detection_index=index,
                pool=pool,
            )

            self.posterior_data[index].append(event_likelihood)
            self.posterior_data_with_bh_mass[index].append(
                event_likelihood_with_bh_mass
            )
            _LOGGER.debug(
                f"event likelihood: {event_likelihood}\nevent likelihood with bh mass: {event_likelihood_with_bh_mass}"
            )
            _LOGGER.debug("posteriors computed for detection...")

    def p_Di(
        self,
        possible_host_galaxies: List[HostGalaxy],
        possible_host_galaxies_with_bh_mass: List[HostGalaxy],
        detection_index: int,
        pool: mp.Pool,
    ) -> float:
        # start parallel computation
        _LOGGER.info(f"start parallel computation with: {pool}")
        start = time.time()
        chunksize = math.ceil(len(possible_host_galaxies) / pool._processes)
        chunksize_with_bh_mass = math.ceil(
            len(possible_host_galaxies_with_bh_mass) / pool._processes
        )
        results_with_bh_mass = pool.starmap(
            single_host_likelihood,
            [
                (
                    possible_host,
                    self.detection,
                    self.h,
                    self.Omega_m,
                    self.Omega_DE,
                    self.w_0,
                    self.w_a,
                    True,
                )
                for possible_host in possible_host_galaxies_with_bh_mass
            ],
            chunksize=chunksize_with_bh_mass,
        )

        results = pool.starmap(
            single_host_likelihood,
            [
                (
                    possible_host,
                    self.detection,
                    self.h,
                    self.Omega_m,
                    self.Omega_DE,
                    self.w_0,
                    self.w_a,
                    False,
                )
                for possible_host in possible_host_galaxies
            ],
            chunksize=chunksize,
        )
        end = time.time()
        _LOGGER.info(f"parallel computing took: {end - start}s")

        integral = 0.0
        weight_sum = 0.0
        integral_with_bh_mass = 0.0
        weight_sum_with_bh_mass = 0.0

        weights = list(
            zip(
                [
                    galaxy.catalog_index
                    for galaxy in possible_host_galaxies_with_bh_mass
                ],
                results_with_bh_mass,
            )
        )

        self.posterior_data_with_bh_mass[GALAXY_WEIGHTS][detection_index] = weights

        for result in results_with_bh_mass:
            # for evaluation without mass
            integral += result[0] * result[1]
            weight_sum += result[1]
            # for evaluation with miss
            integral_with_bh_mass += result[0] * result[2] * result[1]
            weight_sum_with_bh_mass += result[2] * result[1]

        for result in results:
            integral += result[0] * result[1]
            weight_sum += result[1]

        if weight_sum == 0:
            return 0, 0
        elif weight_sum_with_bh_mass == 0:
            return integral / weight_sum, 0
        return integral / weight_sum, integral_with_bh_mass / weight_sum_with_bh_mass

    def use_detection(self) -> bool:
        sky_localization_uncertainty = self._sky_localization_uncertainty(
            phi_error=self.detection.phi_error,
            theta=self.detection.theta,
            theta_error=self.detection.theta_error,
        )
        distance_relative_error = self.detection.d_L_uncertainty / self.detection.d_L

        if (distance_relative_error < 0.1) and (sky_localization_uncertainty < 0.1):
            return True
        _LOGGER.info(
            f"Detection skipped: distance_relative_error {distance_relative_error}, sky_localization_uncertainty {sky_localization_uncertainty}"
        )
        return False

    @staticmethod
    def _sky_localization_uncertainty(
        phi_error: float, theta: float, theta_error: float
    ) -> float:
        return (
            2
            * np.pi
            * np.abs(np.sin(theta))
            * np.sqrt(phi_error**2 * theta_error**2)  # TODO no covariance used
        )


def single_host_likelihood(
    possible_host: HostGalaxy,
    detection: Detection,
    h: float,
    Omega_m: float,
    Omega_de: float,
    w_0: float,
    w_a: float,
    evaluate_with_bh_mass: bool,
) -> list[float]:
    start = time.time()
    z_gws = np.linspace(0, 1, 10000)
    if evaluate_with_bh_mass:
        # compute weight using the bh mass
        norm_dist_measurement = NormalDist(
            mu=detection.M, sigma=detection.M_uncertainty
        )
        norm_dist_galaxy = NormalDist(mu=possible_host.M, sigma=possible_host.M_error)
        current_mass_weight = norm_dist_measurement.overlap(norm_dist_galaxy)

    # compute weight
    multivariate_normal_distribution = multivariate_normal(
        mean=[detection.phi, detection.theta],
        cov=[
            [detection.phi_error**2, detection.theta_phi_covariance],
            [detection.theta_phi_covariance, detection.theta_error**2],
        ],
    )
    current_weight = multivariate_normal_distribution.pdf(
        [possible_host.phiS, possible_host.qS]
    )

    """WL_uncertainty = (
        d_L * 0.066 * (1 - (1 + possible_host.z) ** (-0.25) / 0.25) ** (1.8)
    )"""  # TODO check if correct

    distances = np.vectorize(dist, excluded=["h", "Omega_m", "Omega_de", "w_0", "w_a"])(
        z_gws,
        h=h,
        Omega_m=Omega_m,
        Omega_de=Omega_de,
        w_0=w_0,
        w_a=w_a,
    )
    gaussian = np.exp(
        -1
        / 2
        * (
            (possible_host.z - z_gws) ** 2 / possible_host.z_error**2
            + (detection.d_L - distances) ** 2 / detection.d_L_uncertainty**2
        )
    )
    """
    _LOGGER.debug(
        "gaussian computed with:\n"
        f"zgw: [{z_gws[0]}, {z_gws[-1]}]\n"
        f"host redshift: {possible_host.z}\n"
        f"host redshift error: {possible_host.z_error}\n"
        f"detection distance: {detection.d_L}\n"
        f"distance error: {detection.d_L_uncertainty}\n"
        f"distances: [{distances[0]}, {distances[-1]} ]\n"
    )
    _LOGGER.debug(
        f"integrating with:\ngaussian: max={max(gaussian)} edges=[{gaussian[0]}, {gaussian[-1]}]"
    )
    """

    result = np.trapz(gaussian, z_gws)
    end = time.time()
    if evaluate_with_bh_mass:
        return [result, current_weight, current_mass_weight]
    return [result, current_weight]


def child_process_init() -> None:
    print(f"started child process: {mp.current_process().name}", flush=True)
