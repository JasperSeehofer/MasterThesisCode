from dataclasses import dataclass
from typing import List, Dict
import json
import pandas as pd
import os
import math
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from scipy.stats import multivariate_normal, truncnorm, gaussian_kde
from scipy.optimize import curve_fit

# import statsmodels.api as sm
from statistics import NormalDist
import multiprocessing as mp
from master_thesis_code.datamodels.parameter_space import (
    ParameterSpace,
    Parameter,
    uniform,
)
import emcee

from master_thesis_code.constants import (
    C,
    H0,
    H,
    H_MIN,
    CRAMER_RAO_BOUNDS_OUTPUT_PATH,
    PREPARED_CRAMER_RAO_BOUNDS_PATH,
    RADIAN_TO_DEGREE,
    KM_TO_M,
    GPC_TO_MPC,
)
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
    ParameterSample,
)
from master_thesis_code.physical_relations import (
    dist,
    convert_true_mass_to_redshifted_mass_with_distance,
    get_redshift_outer_bounds,
    dist_to_redshift,
)

_LOGGER = logging.getLogger()

DEFAULT_GALAXY_Z_ERROR = 0.0015
GALAXY_LIKELIHOODS = "galaxy_likelihoods"


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
    theta_phi_covariance: float
    M_phi_covariance: float
    M_theta_covariance: float
    d_L_M_covariance: float
    d_L_theta_covariance: float
    d_L_phi_covariance: float
    host_galaxy_index: int
    snr: float
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
        self.d_L_M_covariance = parameters["delta_dist_delta_M"]
        self.d_L_theta_covariance = parameters["delta_qS_delta_dist"]
        self.d_L_phi_covariance = parameters["delta_phiS_delta_dist"]
        self.host_galaxy_index = parameters["host_galaxy_index"]

    def get_skylocalization_error(self) -> float:
        return _sky_localization_uncertainty(
            self.phi_error, self.theta, self.theta_error, self.theta_phi_covariance
        )

    def convert_to_best_guess_parameters(self) -> None:
        while True:
            self.phi = np.random.normal(self.phi, self.phi_error)
            if 0 <= self.phi < 2 * np.pi:
                break
        while True:
            self.theta = np.random.normal(self.theta, self.theta_error)
            if 0 <= self.theta <= np.pi:
                break
        while True:
            self.d_L = np.random.normal(self.d_L, self.d_L_uncertainty)
            if 0 <= self.d_L:
                break

        while True:
            self.M = np.random.normal(
                self.M * (1 + dist_to_redshift(self.d_L)), self.M_uncertainty
            )
            if 1e4 <= self.M <= 1e7:
                break


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
        self.setup_emri_events_sampler()

    def _apply_model_assumptions(self) -> None:

        self.parameter_space.M.lower_limit = 10 ** (4.5)
        self.parameter_space.M.upper_limit = 10 ** (6.0)

        self.parameter_space.a.value = 0.98
        self.parameter_space.a.is_fixed = True

        self.parameter_space.mu.value = 10
        self.parameter_space.mu.is_fixed = True

        self.parameter_space.e0.upper_limit = 0.2

        self.max_redshift = 2.0
        self.parameter_space.dist.upper_limit = dist(redshift=self.max_redshift)

    def emri_distribution(self, M: float, redshift: float) -> float:
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

    def _log_probability(self, M: float, redshift: float) -> float:
        if (
            not self.parameter_space.M.lower_limit
            < M
            < self.parameter_space.M.upper_limit
        ):
            return -np.inf
        if not 0 < redshift < self.max_redshift:
            return -np.inf
        return np.log(self.emri_distribution(M, redshift))

    def setup_emri_events_sampler(self) -> None:
        # use emcee to sample the distribution

        log_probability = lambda x: self._log_probability(10 ** x[0], x[1])

        ndim = 2
        nwalkers = 100
        burn_in_steps = 1000
        p0_mass = np.random.rand(nwalkers, 1) * (
            np.log10(self.parameter_space.M.upper_limit)
            - np.log10(self.parameter_space.M.lower_limit)
        ) + np.log10(self.parameter_space.M.lower_limit)
        p0_redshift = np.random.rand(nwalkers, 1) * self.max_redshift
        p0 = np.column_stack((p0_mass, p0_redshift))
        _LOGGER.info(
            f"Setup emcee MCMC with {nwalkers} walkers and {burn_in_steps} burn in steps..."
        )
        self._emri_event_sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
        )

        # let the walkers burn in
        pos, prob, state = self._emri_event_sampler.run_mcmc(p0, burn_in_steps)
        self._sample_positions = pos
        self._emri_event_sampler.reset()

        _LOGGER.info("Burn in complete...")

    def sample_emri_events(self, number_of_samples: int) -> List[ParameterSample]:
        _LOGGER.info("Sampling EMRI events...")
        pos, prob, state = self._emri_event_sampler.run_mcmc(
            initial_state=self._sample_positions, nsteps=number_of_samples
        )
        samples = self._emri_event_sampler.get_chain(flat=True)
        self._sample_positions = pos
        self._emri_event_sampler.reset()
        return_samples = [
            ParameterSample(
                M=10 ** sample[0], redshift=sample[1], a=MBH_spin_distribution(0, 1)
            )
            for sample in samples
        ]
        _LOGGER.info(f"Sampling complete (number of samples ({len(return_samples)})).")

        return return_samples

    def visualize_emri_distribution_sampling(self, number_of_samples: int) -> None:
        samples = self.sample_emri_events(number_of_samples)
        masses = [sample.M for sample in samples]
        redshifts = [sample.redshift for sample in samples]

        # make a 2d contour plot of the distribution
        mass_bins = np.geomspace(
            self.parameter_space.M.lower_limit, self.parameter_space.M.upper_limit, 40
        )
        redshift_bins = np.linspace(
            0, dist_to_redshift(self.parameter_space.dist.upper_limit), 40
        )
        plt.figure(figsize=(10, 6))
        plt.hist2d(redshifts, masses, bins=[redshift_bins, mass_bins], cmap="viridis")
        plt.colorbar()
        plt.yscale("log")
        plt.xlabel("redshift")
        plt.ylabel("mass")
        plt.savefig("saved_figures/cosmological_model/emri_distribution_sampling.png")
        plt.close()

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
        distribution = np.vectorize(self.emri_distribution)(masses, redshifts)

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
    h_values_with_bh_mass: List = []
    galaxy_weights = {}
    posterior_data: Dict[int, List[float]] = {}
    posterior_data_with_bh_mass: Dict[int, List[float]] = {}

    def __init__(self) -> None:
        self.cramer_rao_bounds = pd.read_csv(PREPARED_CRAMER_RAO_BOUNDS_PATH)
        self.true_cramer_rao_bounds = pd.read_csv(CRAMER_RAO_BOUNDS_OUTPUT_PATH)
        _LOGGER.info(f"Loaded {len(self.cramer_rao_bounds)} detections...")
        self.cosmological_model = LamCDMScenario()
        self.h = self.cosmological_model.h.fiducial_value
        self.Omega_m = self.cosmological_model.Omega_m.fiducial_value
        self.Omega_DE = 1 - self.Omega_m
        self.w_0 = self.cosmological_model.w_0
        self.w_a = self.cosmological_model.w_a

    def visualize(self, galaxy_catalog: GalaxyCatalogueHandler) -> None:

        posteriors_directory = "simulations/posteriors"
        posteriors_with_bh_mass_directory = "simulations/posteriors_with_bh_mass"

        posteriors_files = [
            file for file in os.listdir(posteriors_directory) if file.endswith(".json")
        ]
        posteriors_with_bh_mass_files = [
            file
            for file in os.listdir(posteriors_with_bh_mass_directory)
            if file.endswith(".json")
        ]

        posteriors_data = {}
        for file in posteriors_files:

            with open(f"{posteriors_directory}/{file}", "r") as file:
                try:
                    h_data = dict(json.load(file))
                    h = str(h_data.pop("h"))
                    posteriors_data[h] = h_data
                except json.JSONDecodeError as e:
                    _LOGGER.error(f"Error reading file {file}: {e}")
                    continue

        posteriors_with_bh_mass_data = {}
        for file in posteriors_with_bh_mass_files:
            with open(f"{posteriors_with_bh_mass_directory}/{file}", "r") as file:
                try:
                    h_data = dict(json.load(file))
                except json.JSONDecodeError as e:
                    _LOGGER.error(f"Error reading file {file}: {e}")
                    continue
                h = str(h_data.pop("h"))
                self.galaxy_weights[h] = h_data.pop(GALAXY_LIKELIHOODS)
                posteriors_with_bh_mass_data[h] = h_data

        # extract h_values and posteriors for each detection
        for h, data in posteriors_data.items():
            self.h_values.append(float(h))
            for detection_index, posterior in data.items():
                try:
                    self.posterior_data[int(detection_index)].extend(posterior)
                except KeyError:
                    self.posterior_data[int(detection_index)] = posterior

        for h, data in posteriors_with_bh_mass_data.items():
            self.h_values_with_bh_mass.append(float(h))
            for detection_index, posterior in data.items():
                try:
                    self.posterior_data_with_bh_mass[int(detection_index)].extend(
                        posterior
                    )
                except KeyError:
                    self.posterior_data_with_bh_mass[int(detection_index)] = posterior

        # drop all posteriors with less samples than h_values or if they are all zero
        self.posterior_data_with_bh_mass = {
            detection_index: posterior
            for detection_index, posterior in self.posterior_data_with_bh_mass.items()
            if (
                (len(posterior) == len(self.h_values_with_bh_mass))
                and (np.max(posterior) > 0)
            )
        }
        self.posterior_data = {
            detection_index: posterior
            for detection_index, posterior in self.posterior_data.items()
            if (
                (len(posterior) == len(self.h_values))
                and (np.max(posterior) > 0)
                and (detection_index in list(self.posterior_data_with_bh_mass.keys()))
            )
        }

        """
        # skylocalization error checkpoint (threshold < 0.001)
        self.posterior_data = {
            detection_index: posterior
            for detection_index, posterior in self.posterior_data.items()
            if Detection(
                self.cramer_rao_bounds.iloc[int(detection_index)]
            ).get_skylocalization_error()
            < 0.0006
        }
        self.posterior_data_with_bh_mass = {
            detection_index: posterior
            for detection_index, posterior in self.posterior_data_with_bh_mass.items()
            if Detection(
                self.cramer_rao_bounds.iloc[int(detection_index)]
            ).get_skylocalization_error()
            < 0.0006
        }
        """
        _LOGGER.info(
            f"After filtering:\n h = {self.h_values}\n h_bh_mass = {self.h_values_with_bh_mass} #detections = {len(self.posterior_data)}\n #detections with bh mass = {len(self.posterior_data_with_bh_mass)}"
        )

        # create detection objects
        detections = [
            Detection(self.cramer_rao_bounds.iloc[int(index)])
            for index in self.posterior_data.keys()
        ]

        distances = [dist_to_redshift(detection.d_L) for detection in detections]

        # plot redshift distribution of detections
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.hist(
            distances,
            bins=np.linspace(0, max(distances), int(max(distances) * 100)),
            histtype="step",
            color="b",
            label="detections",
        )
        ax.set_xlabel("redshift")
        ax.set_ylabel("count")
        plt.savefig("saved_figures/detection_redshift_distribution.png", dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(16, 9))
        fig.suptitle("Redshift distribution of subsets of detections")
        # look for bias in distances of detections by taking subsets of the data
        for count in range(30):
            distances_subset = np.random.choice(
                distances, int(len(distances) / 2), replace=False
            )
            # get hist as points for better visualization
            hist, bins = np.histogram(
                distances_subset,
                bins=np.linspace(0, max(distances), int(max(distances) * 100)),
            )
            ax.plot(
                bins[:-1],
                hist,
            )
        ax.set_xlabel("redshift")
        ax.set_ylabel("count")
        plt.savefig(
            f"saved_figures/detection_redshift_distribution_subset.png",
            dpi=300,
        )
        plt.close()

        # define colormap for skylocalization coloring
        sky_localization_error_min = min(
            [detection.get_skylocalization_error() for detection in detections]
        )
        sky_localization_error_max = max(
            [detection.get_skylocalization_error() for detection in detections]
        )
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(
            vmin=sky_localization_error_min, vmax=sky_localization_error_max
        )

        """
        # sort h_values, posteriors and posteriors with bh mass by h value
        zipped = list(zip(self.h_values, self.posterior_data.items()))
        zipped.sort(key=lambda x: x[0])
        self.h_values, posterior_data_sorted = zip(*zipped)

        zipped_with_bh_mass = list(zip(self.h_values_with_bh_mass, self.posterior_data_with_bh_mass.items()))
        zipped_with_bh_mass.sort(key=lambda x: x[0])
        self.h_values_with_bh_mass, posterior_data_with_bh_mass_sorted = zip(*zipped_with_bh_mass)
        """
        posterior_data_sorted = self.posterior_data.items()
        posterior_data_with_bh_mass_sorted = self.posterior_data_with_bh_mass.items()

        fig, ax = plt.subplots(figsize=(16, 9))
        # plot line for true value
        ax.axvline(H, color="b", linestyle="--")
        for detection_index, posterior in posterior_data_sorted:
            detection = Detection(self.cramer_rao_bounds.iloc[int(detection_index)])
            color = cmap(norm(detection.get_skylocalization_error()))

            zipped = list(zip(self.h_values, posterior))
            zipped.sort(key=lambda x: x[0])
            h_values, posterior = zip(*zipped)

            ax.plot(
                h_values,
                posterior / np.max(posterior),
                label=f"detection: {detection_index}",
                color=color,
            )
        ax.set_xlabel("Hubble constant h")
        ax.set_ylabel("Posterior")
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            label="skylocalization error",
        )
        plt.savefig(f"saved_figures/bayesian_statistics_event_posteriors.png", dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(16, 9))
        # plot line for true value
        ax.axvline(H, color="b", linestyle="--")
        for detection_index, posterior in posterior_data_with_bh_mass_sorted:
            detection = Detection(self.cramer_rao_bounds.iloc[int(detection_index)])
            color = cmap(norm(detection.get_skylocalization_error()))

            zipped = list(zip(self.h_values_with_bh_mass, posterior))
            zipped.sort(key=lambda x: x[0])
            h_values_with_bh_mass, posterior = zip(*zipped)

            ax.plot(
                h_values_with_bh_mass,
                posterior / np.max(posterior),
                label=f"detection {detection_index}",
                color=color,
            )
        ax.set_xlabel("Hubble constant h")
        ax.set_ylabel("Posterior")
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            label="skylocalization error",
        )
        plt.savefig(
            f"saved_figures/bayesian_statistics_event_posteriors_with_bh_mass.png",
            dpi=300,
        )
        plt.close()

        # look at random subset of detections and their posterior
        fig, ax = plt.subplots(3, 2, figsize=(16, 9), height_ratios=[3, 1, 1])

        # create color list with 10 different colors
        NUMBER_OF_SUBSETS = 20
        NUMBER_OF_DETECTIONS = 50
        fig.suptitle(
            f"Posterior distribution of Hubble constant h using {NUMBER_OF_SUBSETS} subsets of {NUMBER_OF_DETECTIONS} detections"
        )
        subset_y_positions = np.linspace(0.3, 1.3, NUMBER_OF_SUBSETS)
        # create a colormap
        cmap = plt.cm.get_cmap(
            "viridis", NUMBER_OF_SUBSETS
        )  # 'viridis' is the colormap name, 10 is the number of colors

        # create a list of 10 colors from the colormap
        colors = [cmap(i) for i in range(cmap.N)]
        overall_sigma_dev, overall_sigma_dev_error = [], []
        overall_sigma_dev_with_bh_mass, overall_sigma_dev_error_with_bh_mass = [], []
        overall_h_mean, overall_h_error = [], []
        overall_h_mean_with_bh_mass, overall_h_error_with_bh_mass = [], []

        for count in range(NUMBER_OF_SUBSETS):
            _LOGGER.info(f"Creating subset {count}")
            posteriors_data_subset_indices = np.random.choice(
                list(self.posterior_data.keys()), NUMBER_OF_DETECTIONS, replace=False
            )
            posteriors_data_subset = [
                (index, self.posterior_data[index])
                for index in posteriors_data_subset_indices
            ]
            # find same subset in posteriors with bh mass
            posteriors_data_with_bh_mass_subset = [
                value
                for value in posterior_data_with_bh_mass_sorted
                if value[0] in posteriors_data_subset_indices
            ]
            sub_posteriors = np.ones(len(self.h_values))
            sub_posteriors_with_bh_mass = np.ones(len(self.h_values_with_bh_mass))

            for index, posterior in posteriors_data_subset:
                if check_overflow(sub_posteriors * np.array(posterior)):
                    # print("Overflow detected")
                    sub_posteriors = sub_posteriors / np.max(sub_posteriors)
                elif np.max(sub_posteriors * posterior) == 0.0:
                    print("All zeros detected")
                    sub_posteriors = sub_posteriors / np.max(sub_posteriors)
                sub_posteriors *= np.array(posterior)
            for index, posterior in posteriors_data_with_bh_mass_subset:
                if check_overflow(sub_posteriors_with_bh_mass * np.array(posterior)):
                    # print("Overflow detected")
                    sub_posteriors_with_bh_mass = sub_posteriors_with_bh_mass / np.max(
                        sub_posteriors_with_bh_mass
                    )
                elif np.max(sub_posteriors_with_bh_mass * posterior) == 0.0:
                    print("All zeros detected")
                    sub_posteriors_with_bh_mass = sub_posteriors_with_bh_mass / np.max(
                        sub_posteriors_with_bh_mass
                    )
                sub_posteriors_with_bh_mass *= np.array(posterior)

            sub_posteriors = sub_posteriors / np.max(sub_posteriors)
            sub_posteriors_with_bh_mass = sub_posteriors_with_bh_mass / np.max(
                sub_posteriors_with_bh_mass
            )
            sub_zipped = list(zip(self.h_values, sub_posteriors))
            sub_zipped.sort(key=lambda x: x[0])
            temp_h_values, sub_posteriors = zip(*sub_zipped)

            sub_zipped_with_bh_mass = list(
                zip(self.h_values_with_bh_mass, sub_posteriors_with_bh_mass)
            )
            sub_zipped_with_bh_mass.sort(key=lambda x: x[0])
            temp_h_values_with_bh_mass, sub_posteriors_with_bh_mass = zip(
                *sub_zipped_with_bh_mass
            )

            # fit normal distribution to posteriors with h values
            h_values_fine = np.linspace(0.6, 0.86, 1000)
            try:
                popt, pcov = curve_fit(
                    gaussian,
                    temp_h_values,
                    sub_posteriors,
                    p0=[H, 0.1, 1],
                    bounds=([0.6, 0, 0], [0.86, 10, 10]),
                )
                perr = np.sqrt(np.diag(pcov))

                popt_with_bh_mass, pcov_with_bh_mass = curve_fit(
                    gaussian,
                    temp_h_values_with_bh_mass,
                    sub_posteriors_with_bh_mass,
                    p0=[H, 0.1, 1],
                    bounds=([0.6, 0, 0], [0.86, 10, 10]),
                )
                perr_with_bh_mass = np.sqrt(np.diag(pcov_with_bh_mass))

                sigma_dev, sigma_dev_error = compute_sigma_deviation(
                    popt[1], perr[1], popt[0], perr[0]
                )

                sigma_dev_with_bh_mass, sigma_dev_error_with_bh_mass = (
                    compute_sigma_deviation(
                        popt_with_bh_mass[1],
                        perr_with_bh_mass[1],
                        popt_with_bh_mass[0],
                        perr_with_bh_mass[0],
                    )
                )

                overall_sigma_dev.append(sigma_dev)
                overall_sigma_dev_error.append(sigma_dev_error)
                overall_sigma_dev_with_bh_mass.append(sigma_dev_with_bh_mass)
                overall_sigma_dev_error_with_bh_mass.append(
                    sigma_dev_error_with_bh_mass
                )
                overall_h_mean.append(popt[0])
                overall_h_error.append(popt[1])
                overall_h_mean_with_bh_mass.append(popt_with_bh_mass[0])
                overall_h_error_with_bh_mass.append(popt_with_bh_mass[1])

                ax[0, 0].plot(
                    h_values_fine,
                    gaussian(h_values_fine, *popt),
                    label=f"std: {np.round(popt[1], 3)}, mean: {np.round(popt[0], 3)}",
                    color=colors[count],
                    linestyle="--",
                )
                ax[1, 0].errorbar(
                    popt[0],
                    subset_y_positions[count],
                    capsize=5,
                    xerr=popt[1],
                    color=colors[count],
                )
                ax[2, 0].errorbar(
                    sigma_dev,
                    subset_y_positions[count],
                    capsize=5,
                    xerr=sigma_dev_error,
                    color=colors[count],
                )

                ax[0, 1].plot(
                    h_values_fine,
                    gaussian(h_values_fine, *popt_with_bh_mass),
                    label=f"std: {np.round(popt_with_bh_mass[1], 3)}, mean: {np.round(popt_with_bh_mass[0], 3)}",
                    color=colors[count],
                    linestyle="--",
                )
                ax[1, 1].errorbar(
                    popt_with_bh_mass[0],
                    subset_y_positions[count],
                    capsize=5,
                    xerr=popt_with_bh_mass[1],
                    color=colors[count],
                )
                ax[2, 1].errorbar(
                    sigma_dev_with_bh_mass,
                    subset_y_positions[count],
                    capsize=5,
                    xerr=sigma_dev_error_with_bh_mass,
                    color=colors[count],
                )

            except RuntimeError:
                pass

            ax[0, 0].scatter(
                temp_h_values,
                sub_posteriors,
                label="without BH mass",
                color=colors[count],
                s=2,
            )
            ax[0, 1].scatter(
                temp_h_values_with_bh_mass,
                sub_posteriors_with_bh_mass,
                label="with BH mass",
                color=colors[count],
                s=2,
            )
        # plot overall sigma deviation
        mean_sigma_dev = np.mean(overall_sigma_dev)
        mean_sigma_dev_error = np.mean(overall_sigma_dev_error)
        mean_sigma_dev_with_bh_mass = np.mean(overall_sigma_dev_with_bh_mass)
        mean_sigma_dev_error_with_bh_mass = np.mean(
            overall_sigma_dev_error_with_bh_mass
        )

        mean_h = np.mean(overall_h_mean)
        mean_h_error = np.mean(overall_h_error)
        mean_h_with_bh_mass = np.mean(overall_h_mean_with_bh_mass)
        mean_h_error_with_bh_mass = np.mean(overall_h_error_with_bh_mass)

        ax[2, 0].errorbar(
            mean_sigma_dev,
            0,
            fmt="|",
            xerr=mean_sigma_dev_error,
            color="red",
            lw=2,
            capthick=2,
            capsize=10,
            label=f"mean sigma deviation: {np.round(mean_sigma_dev, 3)} +/- {np.round(mean_sigma_dev_error, 3)}",
        )
        ax[1, 0].errorbar(
            mean_h,
            0,
            xerr=mean_h_error,
            lw=2,
            capthick=2,
            fmt="|",
            color="red",
            capsize=10,
            label=f"mean h: {np.round(mean_h, 3)} +/- {np.round(mean_h_error, 3)}",
        )

        ax[2, 1].errorbar(
            mean_sigma_dev_with_bh_mass,
            0,
            fmt="|",
            xerr=mean_sigma_dev_error_with_bh_mass,
            color="red",
            capsize=10,
            lw=2,
            capthick=2,
            label=f"mean sigma deviation: {np.round(mean_sigma_dev_with_bh_mass, 3)} +/- {np.round(mean_sigma_dev_error_with_bh_mass, 3)}",
        )

        ax[1, 1].errorbar(
            mean_h_with_bh_mass,
            0,
            fmt="|",
            xerr=mean_h_error_with_bh_mass,
            color="red",
            capsize=10,
            lw=2,
            capthick=2,
            label=f"mean h: {np.round(mean_h_with_bh_mass, 3)} +/- {np.round(mean_h_error_with_bh_mass, 3)}",
        )

        ax[0, 0].axvline(H, color="g", linestyle="--")
        ax[0, 0].set_xlabel("Hubble constant h")
        ax[0, 0].set_ylabel("Posterior")
        ax[0, 0].set_title(f"without BH mass")
        ax[0, 0].set_xlim(0.6, 0.86)

        ax[1, 0].axvline(H, color="g", linestyle="--")
        ax[1, 0].set_yticks([0, 1], ["mean", "subsets"])
        ax[1, 0].set_ylim(-0.2, 1.5)
        ax[1, 0].set_xlabel("predicted h")
        ax[1, 0].set_xlim(0.6, 0.86)
        ax[1, 0].legend()

        ax[2, 0].set_yticks([0, 1], ["mean", "subsets"])
        ax[2, 0].set_ylim(-0.2, 1.5)
        ax[2, 0].set_xlabel("sigma deviation")
        ax[2, 0].legend()

        ax[0, 1].axvline(H, color="g", linestyle="--")
        ax[0, 1].set_xlabel("Hubble constant h")
        ax[0, 1].set_ylabel("Posterior")
        ax[0, 1].set_title(f"with BH mass")
        ax[0, 1].set_xlim(0.6, 0.86)

        ax[1, 1].axvline(H, color="g", linestyle="--")
        ax[1, 1].set_yticks([0, 1], ["mean", "subsets"])
        ax[1, 1].set_ylim(-0.2, 1.5)
        ax[1, 1].set_xlabel("predicted h")
        ax[1, 1].set_xlim(0.6, 0.86)
        ax[1, 1].legend()

        ax[2, 1].set_yticks([0, 1], ["mean", "subsets"])
        ax[2, 1].set_ylim(-0.2, 1.5)
        ax[2, 1].set_xlabel("sigma deviation")
        ax[2, 1].legend()
        plt.tight_layout()
        plt.savefig(
            f"saved_figures/bayesian_statistics_event_posteriors_subsets.png",
            dpi=300,
        )
        plt.close()

        posteriors = np.ones(len(self.h_values))
        posteriors_with_bh_mass = np.ones(len(self.h_values_with_bh_mass))

        for index, posterior in posterior_data_sorted:
            if check_overflow(posteriors * np.array(posterior)):
                # print("Overflow detected")
                posteriors = posteriors / np.max(posteriors)
            posteriors *= np.array(posterior)
        for index, posterior in posterior_data_with_bh_mass_sorted:
            if check_overflow(posteriors_with_bh_mass * np.array(posterior)):
                # print("Overflow detected")
                posteriors_with_bh_mass = posteriors_with_bh_mass / np.max(
                    posteriors_with_bh_mass
                )
            posteriors_with_bh_mass *= np.array(posterior)

        # print(posteriors, posteriors_with_bh_mass)
        posteriors = posteriors / np.max(posteriors)
        posteriors_with_bh_mass = posteriors_with_bh_mass / np.max(
            posteriors_with_bh_mass
        )

        zipped = list(zip(self.h_values, posteriors))
        zipped.sort(key=lambda x: x[0])
        self.h_values, posteriors = zip(*zipped)

        zipped_with_bh_mass = list(
            zip(self.h_values_with_bh_mass, posteriors_with_bh_mass)
        )
        zipped_with_bh_mass.sort(key=lambda x: x[0])
        self.h_values_with_bh_mass, posteriors_with_bh_mass = zip(*zipped_with_bh_mass)

        # fit normal distribution to posteriors with h values
        fig = plt.figure(figsize=(16, 9))
        plt.title(
            f"Posterior distribution of Hubble constant h using {len(detections)} detections"
        )
        h_fine = np.linspace(0.6, 0.86, 1000)
        try:
            popt, pcov = curve_fit(
                gaussian,
                self.h_values,
                posteriors,
                p0=[H, 0.1, 1],
                bounds=([0.6, 0, 0], [0.86, 10, 1]),
            )
            perr = np.sqrt(np.diag(pcov))

            popt_with_bh_mass, pcov_with_bh_mass = curve_fit(
                gaussian,
                self.h_values_with_bh_mass,
                posteriors_with_bh_mass,
                p0=[H, 0.1, 1],
                bounds=([0.6, 0, 0], [0.86, 10, 10]),
            )
            perr_with_bh_mass = np.sqrt(np.diag(pcov_with_bh_mass))

            # prepare confidence bands
            upper_limit = [
                gaussian(x, popt[0], popt[1] + perr[1], popt[2] + perr[2])
                for x in h_fine
            ]
            lower_limit = [
                gaussian(x, popt[0], popt[1] - perr[1], popt[2] - perr[2])
                for x in h_fine
            ]
            plt.fill_between(h_fine, lower_limit, upper_limit, alpha=0.5, color="b")

            plt.plot(
                h_fine,
                gaussian(h_fine, *popt),
                label=f"std: {np.round(popt[1], 3)} +/- {np.round(perr[1], 3)},\nmean: {np.round(popt[0], 3)} +/- {np.round(perr[0], 3)}",
                color="b",
                linestyle="--",
            )
            sigma_deviation = np.abs(popt[0] - H) / popt[1]
            plt.vlines(
                popt[0],
                0,
                1,
                color="black",
                linestyle=":",
            )
            plt.hlines(
                [0.5],
                min(H, popt[0]),
                max(H, popt[0]),
                color="b",
                linestyle=":",
                label=f"sigma deviation: {np.round(sigma_deviation, 3)}",
            )

            upper_limit_with_bh_mass = [
                gaussian(
                    x,
                    popt_with_bh_mass[0],
                    popt_with_bh_mass[1] + perr_with_bh_mass[1],
                    popt_with_bh_mass[2] + perr_with_bh_mass[2],
                )
                for x in h_fine
            ]
            lower_limit_with_bh_mass = [
                gaussian(
                    x,
                    popt_with_bh_mass[0],
                    popt_with_bh_mass[1] - perr_with_bh_mass[1],
                    popt_with_bh_mass[2] - perr_with_bh_mass[2],
                )
                for x in h_fine
            ]
            plt.fill_between(
                h_fine,
                lower_limit_with_bh_mass,
                upper_limit_with_bh_mass,
                alpha=0.5,
                color="r",
            )

            plt.plot(
                h_fine,
                gaussian(h_fine, *popt_with_bh_mass),
                label=f"std: {np.round(popt_with_bh_mass[1], 3)} +/- {np.round(perr_with_bh_mass[1], 3)},\nmean: {np.round(popt_with_bh_mass[0], 3)} +/- {np.round(perr_with_bh_mass[0], 3)}",
                color="r",
                linestyle="--",
            )
            sigma_deviation_with_bh_mass = (
                np.abs(popt_with_bh_mass[0] - H) / popt_with_bh_mass[1]
            )
            plt.vlines(
                popt_with_bh_mass[0],
                0,
                1,
                color="black",
                linestyle=":",
            )
            plt.hlines(
                [0.4],
                min(H, popt_with_bh_mass[0]),
                max(H, popt_with_bh_mass[0]),
                color="r",
                linestyle=":",
                label=f"sigma deviation: {np.round(sigma_deviation_with_bh_mass, 3)}",
            )
        except RuntimeError:
            logging.warning("Could not fit gaussian to data")

        # add true value as line
        plt.axvline(H, color="g", linestyle="--")
        plt.scatter(self.h_values, posteriors, label="without BH mass", color="b")
        plt.scatter(
            self.h_values_with_bh_mass,
            posteriors_with_bh_mass,
            label="with BH mass",
            color="r",
        )
        plt.xlabel("Hubble constant h")
        plt.ylabel("Posterior")
        plt.xlim(0.6, 0.86)
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig("saved_figures/bayesian_statistics.png", dpi=300)
        plt.close()

    def visualize_galaxy_weights(self, galaxy_catalog: GalaxyCatalogueHandler) -> None:
        _LOGGER.info("Visualizing galaxy weights...")
        # visualize galaxy weights
        # restructure galaxy weights data
        weight_data = {}
        for h, data in self.galaxy_weights.items():
            for detection_index, host_galaxy_weights in data.items():
                try:
                    weight_data[int(detection_index)][h] = host_galaxy_weights
                except KeyError:
                    if len(host_galaxy_weights) == 0:
                        continue
                    weight_data[int(detection_index)] = {}
                    weight_data[int(detection_index)][h] = host_galaxy_weights

        # remove weight_data with less samples than h_values
        weight_data = {
            detection_index: host_galaxy_weights
            for detection_index, host_galaxy_weights in weight_data.items()
            if len(host_galaxy_weights.keys()) == len(self.h_values_with_bh_mass)
        }

        max_likelihood_without_bh_mass_by_detection = {
            detection_index: np.max(
                [
                    np.sum([likelihood[0] for _, likelihood in value])
                    for value in host_galaxy_weights.values()
                ]
            )
            for detection_index, host_galaxy_weights in weight_data.items()
        }

        max_likelihood_with_bh_mass_by_detection = {
            detection_index: np.max(
                [
                    np.sum([likelihood[1] for _, likelihood in value])
                    for value in host_galaxy_weights.values()
                ]
            )
            for detection_index, host_galaxy_weights in weight_data.items()
        }
        for detection_index, host_galaxy_weights_by_h_value in weight_data.items():
            _LOGGER.info(f"Visualizing galaxy weights for detection {detection_index}")
            detection = Detection(self.cramer_rao_bounds.iloc[int(detection_index)])
            true_galaxy = Detection(
                self.true_cramer_rao_bounds.iloc[int(detection_index)]
            )
            max_likelihood_without_bh_mass = (
                max_likelihood_without_bh_mass_by_detection[detection_index]
            )
            max_likelihood_with_bh_mass = max_likelihood_with_bh_mass_by_detection[
                detection_index
            ]
            true_galaxy_index = int(
                self.true_cramer_rao_bounds.iloc[int(detection_index)][
                    "host_galaxy_index"
                ]
            )

            # plot h values on x axis and weights on y axis and the sum of weights
            fig, axs = plt.subplots(2, 3, figsize=(16, 9))
            # figure title
            fig.suptitle(
                f"Galaxy likelihood visualization for detection {detection_index} (Skyloc error: {np.round(detection.get_skylocalization_error(), 4)})"
            )
            for h_value, host_galaxy_weights in host_galaxy_weights_by_h_value.items():
                h_value = float(h_value)
                host_galaxies = [
                    galaxy_catalog.get_host_galaxy_by_index(int(index))
                    for index, _ in host_galaxy_weights
                ]
                host_galaxies_phi = np.array([galaxy.phiS for galaxy in host_galaxies])
                host_galaxies_theta = np.array([galaxy.qS for galaxy in host_galaxies])
                host_galaxies_redshift = np.array(
                    [galaxy.z for galaxy in host_galaxies]
                )
                host_galaxies_mean_redshift = np.mean(host_galaxies_redshift)

                host_galaxies_mass = np.array([galaxy.M for galaxy in host_galaxies])
                likelihoods_without_bh_mass = np.array(
                    [weights[0] for _, weights in host_galaxy_weights]
                )
                likelihoods_with_bh_mass = np.array(
                    [weights[1] for _, weights in host_galaxy_weights]
                )

                detection_likelihood_without_bh_mass = (
                    np.sum(likelihoods_without_bh_mass) / max_likelihood_without_bh_mass
                )
                detection_likelihood_with_bh_mass = (
                    np.sum(likelihoods_with_bh_mass) / max_likelihood_with_bh_mass
                )

                # plot likelihood contribution by redshift bins
                redshift_bins = np.linspace(
                    min(host_galaxies_redshift), max(host_galaxies_redshift), num=21
                )
                likelihood_bin_contribution = []
                likelihood_with_bh_mass_bin_contribution = []
                galaxies_per_bin = []
                for bin_number in range(20):
                    redshift_min = redshift_bins[bin_number]
                    redshift_max = redshift_bins[bin_number + 1]
                    bin_galaxies = np.where(
                        np.logical_and(
                            host_galaxies_redshift >= redshift_min,
                            host_galaxies_redshift < redshift_max,
                        )
                    )[0]
                    galaxies_per_bin.append(len(bin_galaxies))
                    contribution = np.sum(
                        [
                            likelihood
                            for likelihood in likelihoods_without_bh_mass[bin_galaxies]
                        ]
                    )
                    likelihood_bin_contribution.append(contribution)
                    contribution_with_bh_mass = np.sum(
                        [
                            likelihood
                            for likelihood in likelihoods_with_bh_mass[bin_galaxies]
                        ]
                    )
                    likelihood_with_bh_mass_bin_contribution.append(
                        contribution_with_bh_mass
                    )
                # plt galaxy number per bin in plot 0,0

                cmap = cm.get_cmap("viridis")

                axs[0, 0].scatter(
                    redshift_bins[:-1],
                    np.array(likelihood_bin_contribution) / np.array(galaxies_per_bin),
                    color=cmap(h_value),
                )

                axs[0, 1].scatter(
                    redshift_bins[:-1], likelihood_bin_contribution, c=cmap(h_value)
                )
                axs[0, 1].axvline(
                    dist_to_redshift(true_galaxy.d_L), color="g", linestyle="--"
                )
                axs[0, 1].set_xlabel("redshift")
                axs[0, 1].axvline(
                    host_galaxies_mean_redshift,
                    color="r",
                    linestyle="--",
                    label="mean redshift",
                )
                axs[0, 1].axvline(
                    dist_to_redshift(detection.d_L),
                    color="black",
                    linestyle="-.",
                    label="detection redshift",
                )
                axs[0, 1].set_title("redshift distribution")

                # plot likelihoods
                # try the weighting by the number of galaxies in the bin
                detection_likelihood_without_bh_mass = np.sum(
                    np.array(likelihood_bin_contribution) / np.array(galaxies_per_bin)
                )
                detection_likelihood_with_bh_mass = np.sum(
                    np.array(likelihood_with_bh_mass_bin_contribution)
                    / np.array(galaxies_per_bin)
                )
                axs[1, 0].scatter(
                    [h_value],
                    detection_likelihood_without_bh_mass,
                    c="b",
                    label="without BH mass",
                )

                axs[1, 1].scatter(
                    [h_value],
                    detection_likelihood_with_bh_mass,
                    c="r",
                    label="with BH mass",
                )
                axs[1, 0].axvline(H, color="g", linestyle="--")
                axs[1, 0].set_title(f"likelihood without BH mass")
                axs[1, 1].axvline(H, color="g", linestyle="--")
                axs[1, 1].set_title(f"likelihood with BH mass")

                """
                # plot weights of true galaxy in detection
                true_galaxy_weights = [
                    weights
                    for index, weights in host_galaxy_weights
                    if int(index) == true_galaxy_index
                ]
                try:
                    true_galaxy_weights = true_galaxy_weights[0]
                    true_galaxy_likelihood_without_bh_mass = true_galaxy_weights[0]
                    true_galaxy_likelihood_with_bh_mass = true_galaxy_weights[1]
                except IndexError:
                    true_galaxy_likelihood_without_bh_mass = 0.0
                    true_galaxy_likelihood_with_bh_mass = 0.0


                
                host_galaxy_indices = [int(index) for index, _ in host_galaxy_weights]
                zipped_likelihood_without_bh_mass = list(
                    zip(host_galaxy_indices, likelihoods_without_bh_mass)
                )
                zipped_likelihood_without_bh_mass.sort(key=lambda x: x[1], reverse=True)
                ranked_indices, ranked_likelihood = zip(*zipped_likelihood_without_bh_mass)
                # find index of true galaxy
                try:
                    true_galaxy_ranking_index = ranked_indices.index(
                        true_galaxy_index, -1
                    )
                except ValueError:
                    true_galaxy_ranking_index = -1

                axs[1, 1].scatter(
                    [h_value],
                    true_galaxy_likelihood_without_bh_mass,
                    c="b",
                    label=f"true galaxy rank {true_galaxy_ranking_index + 1} without bh mass.",
                )

                zipped_likelihood_with_bh_mass = list(
                    zip(
                        host_galaxy_indices,
                        likelihoods_with_bh_mass,
                    )
                )
                zipped_likelihood_with_bh_mass.sort(key=lambda x: x[1], reverse=True)
                ranked_indices_bh_mass, ranked_likelihood_bh_mass = zip(
                    *zipped_likelihood_with_bh_mass
                )
                try:
                    true_galaxy_ranking_index_bh_mass = ranked_indices_bh_mass.index(
                        true_galaxy_index,
                    )
                except ValueError:
                    true_galaxy_ranking_index_bh_mass = -1
                axs[1, 1].scatter(
                    [h_value],
                    true_galaxy_likelihood_with_bh_mass,
                    c="r",
                    label=f"true galaxy rank {true_galaxy_ranking_index_bh_mass + 1}.",
                )
                """
                PLOT_GALAXIES = False
                if (np.round(h_value, 2) == H) and PLOT_GALAXIES:
                    # plot redshift mass distribution
                    axs[1, 1].scatter(
                        host_galaxies_redshift,
                        np.log10(host_galaxies_mass),
                        s=likelihoods_with_bh_mass
                        / max(likelihoods_with_bh_mass)
                        * 100,
                        c=likelihoods_with_bh_mass,
                        cmap="viridis",
                    )
                    axs[1, 1].axvline(
                        dist_to_redshift(true_galaxy.d_L), color="g", linestyle="--"
                    )
                    axs[1, 1].axhline(
                        np.log10(true_galaxy.M), color="g", linestyle="--"
                    )
                    axs[1, 1].axvline(
                        dist_to_redshift(detection.d_L), color="black", linestyle="-."
                    )
                    axs[1, 1].axhline(
                        np.log10(detection.M), color="black", linestyle="-."
                    )
                    axs[1, 1].set_xlabel("redshift")
                    axs[1, 1].set_ylabel("log 10 mass in solar masses")
                    axs[1, 1].set_title("redshift mass distribution")

                    axs[0, 2].scatter(
                        host_galaxies_phi,
                        host_galaxies_theta,
                        c=likelihoods_without_bh_mass,
                        cmap="viridis",
                    )
                    axs[1, 2].scatter(
                        host_galaxies_phi,
                        host_galaxies_theta,
                        c=likelihoods_with_bh_mass,
                        cmap="viridis",
                    )
                    for index in [0, 1]:
                        axs[index, 2].axvline(
                            detection.phi,
                            color="black",
                            linestyle="-.",
                            label="detection",
                        )
                        axs[index, 2].axhline(
                            detection.theta, color="black", linestyle="-."
                        )
                        axs[index, 2].axvline(
                            true_galaxy.phi, color="g", linestyle="--", label="true"
                        )
                        axs[index, 2].axhline(
                            true_galaxy.theta, color="g", linestyle="--"
                        )
                        axs[index, 2].set_xlabel("phi in rad")
                        axs[index, 2].set_ylabel("theta in rad")
                        axs[index, 2].set_title(
                            f"Galaxy skylocalization weight for h = {h_value}"
                        )

            plt.savefig(
                f"saved_figures/galaxy_weights/detection_weight_relations_{detection_index}.png",
                dpi=300,
            )
            plt.close()

            """
            # setup subplots with 2 graphs
            fig, axs = plt.subplots(1, 2, figsize=(16, 9))
            _LOGGER.debug(f"Visualizing galaxy weights for detection {detection_index}")
            _LOGGER.debug(f"found {len(host_galaxy_weights)} host galaxies...")
            for h_index, host_galaxy_weights in enumerate(
                host_galaxy_weights_by_h_value
            ):
                h_value = h_values[h_index]

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
                weighted_likelihoods = unweighted_likelihoods * weights
                weighted_likelihoods_bh_mass = (
                    unweighted_likelihoods * weights_bh_mass * weights
                )
                weighted_scales = weighted_likelihoods / np.max(weighted_likelihoods)
                weighted_scales_bh_mass = weighted_likelihoods_bh_mass / np.max(
                    weighted_likelihoods_bh_mass
                )

                axs[0].scatter(
                    [galaxy.z for galaxy in host_galaxies],
                    [h_value] * len(host_galaxies),
                    s=weighted_scales * 100,
                    c=unweighted_likelihoods * weights,
                    cmap="viridis",
                )
                axs[1].scatter(
                    [galaxy.z for galaxy in host_galaxies],
                    [h_value] * len(host_galaxies),
                    s=weighted_scales_bh_mass * 100,
                    c=unweighted_likelihoods * weights_bh_mass * weights,
                    cmap="viridis",
                )
            axs[0].set_title(
                f"Galaxy weighted likelihood for detection {detection_index}"
            )
            axs[1].set_title(
                f"Galaxy mass weighted likelihood for detection {detection_index}"
            )
            # plot lines for detection (true values)
            detection_redshift = dist_to_redshift(detection.d_L)
            axs[0].axvline(detection_redshift, color="r", linestyle="--", label="detection")
            axs[1].axvline(detection_redshift, color="r", linestyle="--")
            # plot lines for true values
            true_redshift = dist_to_redshift(true_galaxy.d_L)
            axs[0].axvline(true_redshift, color="g", linestyle="--", label="true")
            axs[1].axvline(true_redshift, color="g", linestyle="--")
            axs[0].axhline(H, color="r", linestyle="--")
            axs[1].axhline(H, color="r", linestyle="--")
            axs[0].set_xlabel("z")
            axs[0].set_ylabel("h")
            axs[1].set_xlabel("z")
            axs[1].set_ylabel("h")
            # create colorbar
            fig.colorbar(
                plt.cm.ScalarMappable(
                    norm=plt.Normalize(
                        vmin=np.min(unweighted_likelihoods * weights),
                        vmax=np.max(unweighted_likelihoods * weights),
                    ),
                    cmap="viridis",
                ),
                ax=axs[0],
                label="weight",
            )
            fig.colorbar(
                plt.cm.ScalarMappable(
                    norm=plt.Normalize(
                        vmin=np.min(unweighted_likelihoods * weights_bh_mass * weights),
                        vmax=np.max(unweighted_likelihoods * weights_bh_mass * weights),
                    ),
                    cmap="viridis",
                ),
                ax=axs[1],
                label="mass weight",
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"saved_figures/galaxy_weights/galaxy_weighted_likelihood_detection_{detection_index}.png",
                dpi=300,
            )
            plt.close()
            """

            # BELOW IS THE OLD CODE FOR VISUALIZING GALAXY WEIGHTS

            """
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
            """

    def evaluate(self, galaxy_catalog: GalaxyCatalogueHandler, h_value: float) -> None:
        _LOGGER.info(f"Computing posteriors for h = {h_value}...")
        if (h_value < self.cosmological_model.h.lower_limit) or (
            h_value > self.cosmological_model.h.upper_limit
        ):
            raise ValueError("Hubble constant out of bounds.")

        _LOGGER.debug(f"Loaded {len(self.cramer_rao_bounds)} detections...")
        # filter detections with skylocalization error > 0.01
        for index, detection in self.cramer_rao_bounds.iterrows():
            detection = Detection(detection)
            if use_detection(detection) is False:
                self.cramer_rao_bounds.drop(index, inplace=True)
        _LOGGER.debug(
            f"After filtering {len(self.cramer_rao_bounds)} detections with skylocalization error < 0.0006"
        )

        # JUST FOR DEBUGGING THE BIAS
        self.cramer_rao_bounds["M"] = self.cramer_rao_bounds["M"] * (
            1 + np.array([dist_to_redshift(d) for d in self.cramer_rao_bounds["dist"]])
        )


        self.h = h_value
        _LOGGER.info("prepare global variable for multiprocessing")
        distances = self.cramer_rao_bounds["dist"]
        redshifts = [
            dist_to_redshift(distance=dist, h=h_value)
            for dist in self.cramer_rao_bounds["dist"]
        ]
        phis = self.cramer_rao_bounds["phiS"]
        thetas = self.cramer_rao_bounds["qS"]
        masses = self.cramer_rao_bounds["M"]
        log_10_masses = np.log10(masses)
        # get covariances
        distance_vars = self.cramer_rao_bounds["delta_dist_delta_dist"]
        phi_vars = self.cramer_rao_bounds["delta_phiS_delta_phiS"] 
        theta_vars = self.cramer_rao_bounds["delta_qS_delta_qS"]
        mass_vars = (
            self.cramer_rao_bounds["delta_M_delta_M"] 
        )  
        dist_phi_covs = self.cramer_rao_bounds["delta_phiS_delta_dist"]
        dist_theta_covs = self.cramer_rao_bounds["delta_qS_delta_dist"]
        dist_mass_covs = self.cramer_rao_bounds["delta_dist_delta_M"]
        phi_theta_covs = self.cramer_rao_bounds["delta_phiS_delta_qS"]
        phi_mass_covs = self.cramer_rao_bounds["delta_phiS_delta_M"]
        theta_mass_covs = self.cramer_rao_bounds["delta_qS_delta_M"]

        self._max_redshift = dist_to_redshift(
            max(self.cramer_rao_bounds["dist"]), self.cosmological_model.h.upper_limit
        )
        self._max_mass = max(self.cramer_rao_bounds["M"])

        # calculate delta redshift
        NUMBER_OF_REDSHIFT_STEPS = 2000
        self._delta_redshift = self._max_redshift / NUMBER_OF_REDSHIFT_STEPS

        _LOGGER.info(f"redshift resolution: {self._delta_redshift}")

        # rescale mass between 0 and 1 for detection distribution
        masses = masses / self._max_mass
        mass_vars = mass_vars / self._max_mass**2
        dist_mass_covs = dist_mass_covs / self._max_mass
        phi_mass_covs = phi_mass_covs / self._max_mass
        theta_mass_covs = theta_mass_covs / self._max_mass

        # compare resolution with variance and chose the larger one
        luminosity_distance_resolution_limit = C / H_MIN / KM_TO_M / GPC_TO_MPC * self._delta_redshift / 2
        mass_resolution_limit = masses * self._delta_redshift / 2

        _LOGGER.warning(
            f"detection distribution: d_L resolution limit violated for {np.sum(distance_vars < luminosity_distance_resolution_limit**2)} detections\n"
            f"max resolution limit variance: {luminosity_distance_resolution_limit**2}, min distance variance: {min(distance_vars)}"
        )

        distance_vars = np.maximum(
            luminosity_distance_resolution_limit**2, distance_vars
        )

        _LOGGER.warning(
            f"detection distribution: mass resolution limit violated for {np.sum(mass_vars < mass_resolution_limit**2)} detections.\n"
            f"max resolution limit variance: {max(mass_resolution_limit**2)}, min mass variance: {min(mass_vars)}"
        )

        mass_vars = np.maximum(
            mass_resolution_limit**2, mass_vars
        )

        detection_distribution_gaussians = [
            multivariate_normal(
                mean=[dist, phi, theta],
                cov=[
                    [dist_var, dist_phi_cov, dist_theta_cov],
                    [dist_phi_cov, phi_var, phi_theta_cov],
                    [dist_theta_cov, phi_theta_cov, theta_var],
                ],
            )
            for dist, phi, theta, dist_var, phi_var, theta_var, dist_phi_cov, dist_theta_cov, phi_theta_cov in zip(
                distances,
                phis,
                thetas,
                distance_vars,
                phi_vars,
                theta_vars,
                dist_phi_covs,
                dist_theta_covs,
                phi_theta_covs,
            )
        ]

        detection_distribution_with_mass_gaussians = [
            multivariate_normal(
                mean=[dist, phi, theta, mass],
                cov=[
                    [dist_var, dist_phi_cov, dist_theta_cov, dist_mass_cov],
                    [dist_phi_cov, phi_var, phi_theta_cov, phi_mass_cov],
                    [dist_theta_cov, phi_theta_cov, theta_var, theta_mass_cov],
                    [dist_mass_cov, phi_mass_cov, theta_mass_cov, mass_var],
                ],
            )
            for dist, phi, theta, mass, dist_var, phi_var, theta_var, mass_var, dist_phi_cov, dist_theta_cov, dist_mass_cov, phi_theta_cov, phi_mass_cov, theta_mass_cov in zip(
                distances,
                phis,
                thetas,
                masses,
                distance_vars,
                phi_vars,
                theta_vars,
                mass_vars,
                dist_phi_covs,
                dist_theta_covs,
                dist_mass_covs,
                phi_theta_covs,
                phi_mass_covs,
                theta_mass_covs,
            )
        ]
        
        PLOT_GAUSSIANS = False
        if PLOT_GAUSSIANS:
            luminosity_distance_range = np.linspace(
                0, max(distances), 50
            )
            redshift_range = [dist_to_redshift(d, h_value) for d in luminosity_distance_range]

            phi_range = np.linspace(0, 2 * np.pi, 10)
            theta_range = np.linspace(0.0001, np.pi - 0.0001, 10)
            log_10_mass_range = np.linspace(
                np.min(log_10_masses), np.max(log_10_masses), 100
            )
            mass_range = np.geomspace(
                np.min(masses), np.max(masses), 100
            )

            luminosity_distance_resolution = np.ones_like(len(self.cramer_rao_bounds)) * (luminosity_distance_range[1] - luminosity_distance_range[0] / 2 )
            phi_resolution = np.ones_like(len(self.cramer_rao_bounds)) * (phi_range[1] - phi_range[0] / 2)
            theta_resolution = np.ones_like(len(self.cramer_rao_bounds)) * (theta_range[1] - theta_range[0] / 2)
            mass_resolution = np.diff(mass_range) / 2
            mass_resolution = np.concatenate((mass_resolution, [mass_resolution[-1]]))

            # compare resolution with variance and chose the larger one
            distance_resolution = np.array(np.maximum(
                luminosity_distance_resolution**2, distance_vars
            ))
            phi_resolution = np.array(np.maximum(phi_resolution**2, phi_vars))
            theta_resolution = np.array(np.maximum(theta_resolution**2, theta_vars))
            mass_resolution = np.array([
                max((mass_resolution[index - 1 ])**2, var) 
                for index, var in zip(np.digitize(masses, mass_range), mass_vars)])
            
            print(f"distance min resolution: {min(distance_resolution)}")
            print(f"phi min resolution: {min(phi_resolution)}")
            print(f"theta min resolution: {min(theta_resolution)}")
            print(f"mass min resolution: {min(mass_resolution)}")


            # detection distribution as the sum of gaussians with parameter standard deviation
            detection_distribution_gaussians = [
                multivariate_normal(
                    mean=[dist, phi, theta],
                    cov=[
                        [dist_var, dist_phi_cov, dist_theta_cov],
                        [dist_phi_cov, phi_var, phi_theta_cov],
                        [dist_theta_cov, phi_theta_cov, theta_var],
                    ],
                )
                for dist, phi, theta, dist_var, phi_var, theta_var, dist_phi_cov, dist_theta_cov, phi_theta_cov in zip(
                    distances,
                    phis,
                    thetas,
                    distance_resolution,
                    phi_resolution,
                    theta_resolution,
                    dist_phi_covs,
                    dist_theta_covs,
                    phi_theta_covs,
                )
            ]

            detection_distribution_with_mass_gaussians = [
                multivariate_normal(
                    mean=[dist, phi, theta, mass],
                    cov=[
                        [dist_var, dist_phi_cov, dist_theta_cov, dist_mass_cov],
                        [dist_phi_cov, phi_var, phi_theta_cov, phi_mass_cov],
                        [dist_theta_cov, phi_theta_cov, theta_var, theta_mass_cov],
                        [dist_mass_cov, phi_mass_cov, theta_mass_cov, mass_var],
                    ],
                )
                for dist, phi, theta, mass, dist_var, phi_var, theta_var, mass_var, dist_phi_cov, dist_theta_cov, dist_mass_cov, phi_theta_cov, phi_mass_cov, theta_mass_cov in zip(
                    distances,
                    phis,
                    thetas,
                    masses,
                    distance_resolution,
                    phi_resolution,
                    theta_resolution,
                    mass_resolution,
                    dist_phi_covs,
                    dist_theta_covs,
                    dist_mass_covs,
                    phi_theta_covs,
                    phi_mass_covs,
                    theta_mass_covs,
                )
            ]
            # plot masses of detections
            fig = plt.figure(figsize=(16, 9))
            plt.scatter(redshifts, log_10_masses)
            plt.xlabel("redshift")
            plt.ylabel("log 10 mass")
            plt.title("masses of detections")
            plt.savefig("saved_figures/masses_of_detections.png", dpi=300)
            plt.close()



            distance_mesh, phi_mesh, theta_mesh = np.meshgrid(
                luminosity_distance_range, phi_range, theta_range, indexing="ij"
            )

            (
                distance_mesh_with_mass,
                phi_mesh_with_mass,
                theta_mesh_with_mass,
                mass_mesh,
            ) = np.meshgrid(
                luminosity_distance_range, phi_range, theta_range, mass_range, indexing="ij"
            )

            values = np.array(
                [
                    distance_mesh.ravel(),
                    phi_mesh.ravel(),
                    theta_mesh.ravel(),
                ]
            ).T

            densities = np.array(
                np.sum(
                    [
                        gaussian.pdf(values)
                        for gaussian in detection_distribution_gaussians
                    ],
                    axis=0,
                )
            )

            values_with_mass = np.array(
                [
                    distance_mesh_with_mass.ravel(),
                    phi_mesh_with_mass.ravel(),
                    theta_mesh_with_mass.ravel(),
                    mass_mesh.ravel(),
                ]
            ).T
            densities_with_mass = np.array(
                np.sum(
                    [
                        gaussian.pdf(values_with_mass)
                        for gaussian in detection_distribution_with_mass_gaussians
                    ],
                    axis=0,
                )
            )

            densities = densities.reshape(
                (len(redshift_range), len(phi_range), len(theta_range))
            )

            densities_with_mass = densities_with_mass.reshape(
                (
                    len(redshift_range),
                    len(phi_range),
                    len(theta_range),
                    len(mass_range),
                )
            )

            densities_phi_integrated = np.trapz(densities, phi_range, axis=1)

            # TODO: do I need to use sin(theta) or not?
            redshift_kde = np.trapz(
                densities_phi_integrated,
                theta_range,
                axis=1,
            )

            densities_phi_integrated_with_mass = np.trapz(
                densities_with_mass, phi_range, axis=1
            )

            densities_theta_integrated_with_mass = np.trapz(
                densities_phi_integrated_with_mass,
                theta_range,
                axis=1,
            )
            densities_mass_integrated_with_mass = np.trapz(
                densities_theta_integrated_with_mass,
                mass_range,
                axis=1,
            )

            # for galactic north and south plots
            galactic_densities = np.trapz(densities, luminosity_distance_range, axis=0)
            densities_redshift_integrated_with_mass = np.trapz(
                densities_with_mass, luminosity_distance_range, axis=0
            )
            galactic_densities_with_mass = np.trapz(
                densities_redshift_integrated_with_mass, mass_range, axis=2
            )

            # plot integrated densities
            fig, axs = plt.subplots(1, 2, figsize=(16, 9))
            # plot 0,0 redshift distribution with and without mass
            axs[0].plot(redshift_range, redshift_kde, label="without mass")
            axs[0].plot(
                redshift_range, densities_mass_integrated_with_mass, label="with mass"
            )
            axs[0].set_title("redshift distribution")
            axs[0].set_xlabel("redshift")
            axs[0].set_ylabel("density")
            axs[0].legend()

            # 2d redshift mass distribution with bh mass
            distance_mesh, mass_mesh = np.meshgrid(redshift_range, mass_range, indexing="ij")
            axs[1].contourf(
                distance_mesh,
                mass_mesh,
                np.log10(densities_theta_integrated_with_mass + 1e-10),
                cmap="viridis",
            )
            axs[1].set_title("redshift mass distribution with BH mass")
            axs[1].set_xlabel("redshift")
            axs[1].set_ylabel("log 10 mass")
            axs[1].set_yscale("log")
            plt.savefig(
                "saved_figures/redshift_mass_distribution_with_bh_mass.png", dpi=300
            )
            plt.close()

            # create phi theta mesh
            phi_mesh, theta_mesh = np.meshgrid(phi_range, theta_range, indexing="ij")

            # plot sky distribution of detections of 3d gaussians
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(111, projection="mollweide")
            plt.title("Sky distribution of detections without BH mass")
            img = ax.scatter(
                phi_mesh.ravel() - np.pi,
                theta_mesh.ravel() - np.pi / 2,
                c=np.log10(galactic_densities.ravel() + 1e-10),
                cmap="viridis",
            )
            plt.grid(True)
            plt.colorbar(img, label="log10 density", orientation="horizontal")
            plt.savefig(
                "saved_figures/sky_detection_distribution_without_bh_mass.png", dpi=300
            )
            plt.close()

            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(111, projection="mollweide")
            plt.title("Sky distribution of detections with BH mass")
            img = ax.scatter(
                phi_mesh.ravel() - np.pi,
                theta_mesh.ravel() - np.pi / 2,
                c=np.log10(galactic_densities_with_mass.ravel() + 1e-10),
                cmap="viridis",
            )
            plt.grid(True)
            plt.colorbar(img, label="log10 density", orientation="horizontal", ax=ax)
            plt.savefig(
                "saved_figures/sky_detection_distribution_with_bh_mass.png", dpi=300
            )
            plt.close()

        PLOT_KDE = False
        if PLOT_KDE:
            self._redshift_skylocalization_histogramm = np.histogramdd(
                np.array([redshifts, phis, thetas]).T,
                bins=(20, 30, 20),
                range=(
                    (0, self._max_redshift),
                    (0, 2 * np.pi),
                    (0, np.pi),
                ),
            )
            # renormalize histogramm
            self._redshift_skylocalization_histogram_detection_count = np.sum(
                self._redshift_skylocalization_histogramm[0]
            )

            self._redshift_skylocalization_histogramm = (
                self._redshift_skylocalization_histogramm[0]
                / self._redshift_skylocalization_histogram_detection_count,
                self._redshift_skylocalization_histogramm[1],
            )

            self._redshift_skylocalization_mass_histogramm = np.histogramdd(
                np.array([redshifts, phis, thetas, log_10_masses]).T,
                bins=(20, 30, 20, 40),
                range=(
                    (0, self._max_redshift),
                    (0, 2 * np.pi),
                    (0, np.pi),
                    (min(log_10_masses), max(log_10_masses)),
                ),
            )
            # renormalize histogramm
            self._redshift_skylocalization_mass_histogram_detection_count = np.sum(
                self._redshift_skylocalization_mass_histogramm[0]
            )
            self._redshift_skylocalization_mass_histogramm = (
                self._redshift_skylocalization_mass_histogramm[0]
                / self._redshift_skylocalization_mass_histogram_detection_count,
                self._redshift_skylocalization_mass_histogramm[1],
            )

            # get 3d gaussian kde for redshift and skylocalization
            redshift_distribution_from_4d_histogramm = np.sum(
                self._redshift_skylocalization_mass_histogramm[0], axis=(1, 2, 3)
            )

            redshift_distribution_from_3d_histogramm = np.sum(
                self._redshift_skylocalization_histogramm[0], axis=(1, 2)
            )

            redshift_mass_distribution_from_4d_histogramm = np.sum(
                self._redshift_skylocalization_mass_histogramm[0], axis=(1, 2)
            )

            """self._redshift_skylocalization_mass_kde = sm.nonparametric.KDEMultivariate(
            data=np.array([distances, phis, thetas, log_10_masses]).T,
            var_type="uuuu",
            bw="normal_reference",
            )"""
            redshift_range = np.linspace(0, self._max_redshift, 50)
            phi_range = np.linspace(0, 2 * np.pi, 30)
            theta_range = np.linspace(0, np.pi, 20)
            log_10_mass_range = np.linspace(min(log_10_masses), max(log_10_masses), 40)
            mass_range = 10**log_10_mass_range

            distance_mesh, phi_mesh, theta_mesh = np.meshgrid(
                redshift_range, phi_range, theta_range, indexing="ij"
            )

            (
                redshift_mesh_with_mass,
                phi_mesh_with_mass,
                theta_mesh_with_mass,
                mass_mesh,
            ) = np.meshgrid(
                redshift_range, phi_range, theta_range, mass_range, indexing="ij"
            )

            values = np.array(
                [
                    [dist(redshift, h_value) for redshift in distance_mesh.ravel()],
                    phi_mesh.ravel(),
                    theta_mesh.ravel(),
                ]
            ).T

            densities = np.array(
                np.sum(
                    [
                        gaussian.pdf(values)
                        for gaussian in detection_distribution_gaussians
                    ],
                    axis=0,
                )
            )

            values_with_mass = np.array(
                [
                    [
                        dist(redshift, h_value)
                        for redshift in redshift_mesh_with_mass.ravel()
                    ],
                    phi_mesh_with_mass.ravel(),
                    theta_mesh_with_mass.ravel(),
                    mass_mesh.ravel(),
                ]
            ).T
            densities_with_mass = np.array(
                np.sum(
                    [
                        gaussian.pdf(values_with_mass)
                        for gaussian in detection_distribution_with_mass_gaussians
                    ],
                    axis=0,
                )
            )

            densities = densities.reshape(
                (len(redshift_range), len(phi_range), len(theta_range))
            )

            densities_with_mass = densities_with_mass.reshape(
                (
                    len(redshift_range),
                    len(phi_range),
                    len(theta_range),
                    len(mass_range),
                )
            )

            densities_phi_integrated = np.trapz(densities, phi_range, axis=1)

            # TODO: do I need to use sin(theta) or not?
            redshift_kde = np.trapz(
                densities_phi_integrated,
                theta_range,
                axis=1,
            )

            densities_phi_integrated_with_mass = np.trapz(
                densities_with_mass, phi_range, axis=1
            )

            densities_theta_integrated_with_mass = np.trapz(
                densities_phi_integrated_with_mass,
                theta_range,
                axis=1,
            )
            densities_mass_integrated_with_mass = np.trapz(
                densities_theta_integrated_with_mass,
                mass_range,
                axis=1,
            )

            only_redshift_kde = gaussian_kde(redshifts)

            only_redshift_mass_kde = gaussian_kde(np.array([redshifts, log_10_masses]))

            redshift_redshift_mass_meshgrid, mass_redshift_mass_meshgrid = np.meshgrid(
                redshift_range, log_10_mass_range, indexing="ij"
            )

            # compare kde with histogram
            fig, ax = plt.subplots(2, 2, figsize=(16, 9), height_ratios=[3, 1])
            ax[0, 0].plot(
                redshift_range,
                redshift_kde,
                label="3d gaussian sum",
            )
            ax[0, 0].plot(
                redshift_range,
                only_redshift_kde(redshift_range),
                label="redshift kde",
            )
            ax[0, 0].plot(
                redshift_range,
                np.trapz(
                    only_redshift_mass_kde(
                        np.array(
                            [
                                redshift_redshift_mass_meshgrid.ravel(),
                                mass_redshift_mass_meshgrid.ravel(),
                            ]
                        )
                    ).reshape((len(redshift_range), len(log_10_mass_range))),
                    log_10_mass_range,
                    axis=1,
                ),
                label="redshift mass kde integrated",
            )
            ax[0, 0].plot(
                redshift_range,
                densities_mass_integrated_with_mass,
                label="4d gaussian sum integrated",
            )
            # plot reduced histogramm
            ax[0, 0].bar(
                self._redshift_skylocalization_mass_histogramm[1][0][:-1],
                redshift_distribution_from_4d_histogramm,
                width=np.diff(self._redshift_skylocalization_mass_histogramm[1][0]),
                label=f"4d histogram total #detection={np.round(np.sum(redshift_distribution_from_4d_histogramm), 2)}",
                alpha=0.5,
            )
            ax[0, 0].bar(
                self._redshift_skylocalization_histogramm[1][0][:-1],
                redshift_distribution_from_3d_histogramm,
                width=np.diff(self._redshift_skylocalization_histogramm[1][0]),
                label=f"3d histogram total #detection={np.round(np.sum(redshift_distribution_from_3d_histogramm), 2)}",
                alpha=0.5,
            )
            redshift_histogram = np.histogram(
                redshifts,
                bins=20,
                range=(0, self._max_redshift),
            )
            ax[0, 1].bar(
                redshift_histogram[1][:-1],
                redshift_histogram[0]
                / np.sum(self._redshift_skylocalization_mass_histogram_detection_count),
                width=np.diff(redshift_histogram[1]),
                label=f"histogram total #detection={np.sum(redshift_histogram[0])/self._redshift_skylocalization_mass_histogram_detection_count}",
            )
            # plot 1,0 show difference between histograms
            ax[1, 0].plot(
                self._redshift_skylocalization_histogramm[1][0][:-1],
                redshift_distribution_from_3d_histogramm
                - redshift_distribution_from_4d_histogramm,
                label="3d - 4d histogram",
            )
            fig.suptitle("Redshift kde vs histogram")
            ax[0, 0].set_xlabel("redshift")
            ax[0, 0].set_ylabel("density")
            ax[0, 0].legend()
            ax[0, 1].set_xlabel("redshift")
            ax[0, 1].set_ylabel("density")
            ax[0, 1].legend()
            ax[1, 0].set_xlabel("redshift")
            ax[1, 0].set_ylabel("difference")
            ax[1, 0].legend()
            plt.savefig("saved_figures/redshift_kde_vs_histogram.png", dpi=300)
            plt.close()

            # compare 4d histogramm with 4d kde and 2d histogramm
            fig, ax = plt.subplots(1, 3, figsize=(16, 9))
            ax[0].contourf(
                redshift_range,
                mass_range,
                densities_theta_integrated_with_mass.T,
                levels=10,
            )
            ax[0].set_xlabel("redshift")
            ax[0].set_ylabel("log 10 mass")
            ax[0].set_yscale("log")
            ax[0].set_title("4d kde")

            extent = [
                self._redshift_skylocalization_mass_histogramm[1][0][0],  # min x
                self._redshift_skylocalization_mass_histogramm[1][0][-2],  # max x
                self._redshift_skylocalization_mass_histogramm[1][3][0],  # min y
                self._redshift_skylocalization_mass_histogramm[1][3][-2],  # max y
            ]
            ax[1].imshow(
                redshift_mass_distribution_from_4d_histogramm.T,
                aspect="auto",
                origin="lower",
                extent=extent,
                interpolation="nearest",
            )

            ax[1].set_xlabel("redshift")
            ax[1].set_ylabel("log 10 mass")
            ax[1].set_title(
                f"4d hist #detections={np.sum(redshift_mass_distribution_from_4d_histogramm)}"
            )

            # choose galaxy position from normal distribution
            distances_errors = [
                dist_to_redshift(dist)
                for dist in self.cramer_rao_bounds["delta_dist_delta_dist"] ** (1 / 2)
            ]

            distances_sampled = [
                np.random.normal(loc=distance, scale=distance_error)
                for distance, distance_error in zip(redshifts, distances_errors)
            ]

            redshift_mass_histogram = np.histogram2d(
                distances_sampled,
                log_10_masses,
                bins=100,
                range=[
                    [0, self._max_redshift],
                    [min(log_10_masses), max(log_10_masses)],
                ],
            )

            extent = [
                redshift_mass_histogram[1][0],
                redshift_mass_histogram[1][-1],
                redshift_mass_histogram[2][0],
                redshift_mass_histogram[2][-1],
            ]

            ax[2].imshow(
                redshift_mass_histogram[0].T,
                aspect="auto",
                origin="lower",
                extent=extent,
                interpolation="nearest",
            )
            ax[2].set_xlabel("redshift")
            ax[2].set_ylabel("log 10 mass")
            ax[2].set_title(
                f"2d histogram #detections={np.sum(redshift_mass_histogram[0])}"
            )
            plt.savefig("saved_figures/redshift_mass_kde_vs_histogram.png", dpi=300)
            plt.close()

            """
            # 3d plot of kde
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_title("3D KDE of redshift, phi and theta")
            distance_mesh, phi_mesh, theta_mesh = np.meshgrid(
                distance_range, phi_range, theta_range
            )
            positions = np.vstack(
                [distance_mesh.ravel(), phi_mesh.ravel(), theta_mesh.ravel()]
            )
            density = self._redshift_skylocalization_kde(positions)
            density_normalized = (density - np.min(density)) / (
                np.max(density) - np.min(density)
            )
            alpha_values = (np.cos(density_normalized * np.pi + np.pi) + 1) / 2
            ax.scatter(
                distance_mesh,
                phi_mesh,
                theta_mesh,
                c=density,
                cmap="viridis",
                alpha=alpha_values,
                s=alpha_values * 50,
            )
            ax.set_xlabel("redshift")
            ax.set_ylabel("phi")
            ax.set_zlabel("theta")
            # plot colormap
            norm = plt.Normalize(vmin=np.min(density), vmax=np.max(density))
            fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="density"
            )
            plt.savefig("saved_figures/3d_kde.png", dpi=300)
            plt.close()
            """

            # 2d plots for fixed thetas
            theta_range = np.linspace(0, np.pi, 6)
            min_density = np.inf
            max_density = -np.inf
            densities_dict = {}
            for index, theta in enumerate(theta_range):
                theta_densities_dict = {}
                distance_mesh, phi_mesh, theta_mesh = np.meshgrid(
                    redshift_range,
                    phi_range,
                    np.ones_like(phi_range) * theta,
                    indexing="ij",
                )
                positions = np.array(
                    [
                        distance_mesh.ravel(),
                        phi_mesh.ravel(),
                        theta_mesh.ravel(),
                    ]
                ).T

                density = np.array(
                    np.sum(
                        [
                            gaussian.pdf(positions)
                            for gaussian in detection_distribution_gaussians
                        ],
                        axis=0,
                    )
                )

                density_normalized = (density - np.min(density)) / (
                    np.max(density) - np.min(density)
                )
                alpha_values = (np.cos(density_normalized * np.pi + np.pi) + 1) / 2
                # store in dictionary
                theta_densities_dict["density"] = density
                theta_densities_dict["alpha"] = alpha_values
                theta_densities_dict["distance_mesh"] = distance_mesh
                theta_densities_dict["phi_mesh"] = phi_mesh
                theta_densities_dict["theta_mesh"] = theta_mesh

                densities_dict[theta] = theta_densities_dict
                min_density = min(min_density, np.min(density))
                max_density = max(max_density, np.max(density))

            # create density colormap
            color_map = cm.ScalarMappable(
                norm=plt.Normalize(vmin=min_density, vmax=max_density), cmap="viridis"
            )

            fig, axs = plt.subplots(1, 6, figsize=(16, 9), sharey=True)
            for index, theta in enumerate(theta_range):
                density = densities_dict[theta]["density"]
                alpha_values = densities_dict[theta]["alpha"]
                distance_mesh = densities_dict[theta]["distance_mesh"]
                phi_mesh = densities_dict[theta]["phi_mesh"]

                colors = color_map.to_rgba(density)

                ax = axs[index]
                ax.set_title(f"qS = {np.round(theta/np.pi, 2)}pi")
                ax.scatter(
                    distance_mesh,
                    phi_mesh,
                    c=colors,
                    alpha=alpha_values,
                    s=alpha_values * 50,
                )
                ax.set_xlabel("redshift")
            axs[0].set_ylabel("phi")
            fig.colorbar(color_map, ax=axs, orientation="vertical", label="density")
            plt.savefig("saved_figures/2d_kde_theta.png", dpi=300)
            plt.close()

        _LOGGER.debug(
            f"Found {len(os.sched_getaffinity(0))} / {os.cpu_count()} (available / system) cpus."
        )
        cpu_count = os.cpu_count()

        """
        if len(os.sched_getaffinity(0)) < cpu_count:
            try:
                os.sched_setaffinity(0, range(cpu_count))
            except OSError:
                _LOGGER.info("Could not set affinity")
        _LOGGER.debug(
            f"After trying to set affinity available cpus: {len(os.sched_getaffinity(0))}"
        )
        """

        with mp.get_context("spawn").Pool(
            len(os.sched_getaffinity(0)) - 4,
            initializer=child_process_init,
            initargs=(
                self._max_redshift,
                self._max_mass,
                self._delta_redshift,
                detection_distribution_gaussians,
                detection_distribution_with_mass_gaussians,
            ),
        ) as pool:
            self.p_D(
                galaxy_catalog=galaxy_catalog,
                pool=pool,
            )
        _LOGGER.info(f"posteriors comupted for h = {self.h}")

        if not os.path.isdir("simulations/posteriors"):
            os.makedirs("simulations/posteriors")
        if not os.path.isdir("simulations/posteriors_with_bh_mass"):
            os.makedirs("simulations/posteriors_with_bh_mass")
        try:
            with open(
                f"simulations/posteriors/h_{str(np.round(self.h,3)).replace('.', '_')}.json",
                "r",
            ) as file:
                try:
                    posteriors_existing_data: dict = dict(json.load(file))
                except json.decoder.JSONDecodeError:
                    posteriors_existing_data: dict = {}
        except FileNotFoundError:
            posteriors_existing_data: dict = {}

        with open(
            f"simulations/posteriors/h_{str(np.round(self.h,3)).replace('.', '_')}.json",
            "w",
        ) as file:
            # update existing data
            data = posteriors_existing_data | {
                str(key): value for key, value in self.posterior_data.items()
            }
            json.dump(data | {"h": self.h}, file)
        try:
            with open(
                f"simulations/posteriors_with_bh_mass/h_{str(np.round(self.h,3)).replace('.', '_')}.json",
                "r",
            ) as file:
                try:
                    posteriors_with_bh_mass_existing_data: dict = dict(json.load(file))
                except json.decoder.JSONDecodeError:
                    posteriors_with_bh_mass_existing_data: dict = {}
        except FileNotFoundError:
            posteriors_with_bh_mass_existing_data: dict = {}

        with open(
            f"simulations/posteriors_with_bh_mass/h_{str(np.round(self.h,3)).replace('.', '_')}.json",
            "w",
        ) as file:
            # update existing data

            data = posteriors_with_bh_mass_existing_data | {
                str(key): value
                for key, value in self.posterior_data_with_bh_mass.items()
            }
            json.dump(data | {"h": self.h}, file)

    def p_D(
        self,
        galaxy_catalog: GalaxyCatalogueHandler,
        pool: mp.Pool,
    ) -> None:
        count = 0
        self.posterior_data_with_bh_mass[GALAXY_LIKELIHOODS] = {}
        for index, detection in self.cramer_rao_bounds.iterrows():
            _LOGGER.info(
                f"Progess: detections: {count}/{len(self.cramer_rao_bounds)}..."
            )
            count += 1
            try:
                self.posterior_data[index]
            except KeyError:
                self.posterior_data[index] = []
                self.posterior_data_with_bh_mass[index] = []

            self.detection = Detection(detection)

            z_min, z_max = get_redshift_outer_bounds(
                distance=self.detection.d_L,
                distance_error=self.detection.d_L_uncertainty,
                h_min=self.cosmological_model.h.lower_limit,
                h_max=self.cosmological_model.h.upper_limit,
                Omega_m_min=self.cosmological_model.Omega_m.lower_limit,
                Omega_m_max=self.cosmological_model.Omega_m.upper_limit,
            )

            if z_max > self._max_redshift:
                z_max = self._max_redshift

            possible_hosts = galaxy_catalog.get_possible_hosts(
                z_min=z_min,
                z_max=z_max,
                phi=self.detection.phi,
                phi_error=self.detection.phi_error,
                theta=self.detection.theta,
                theta_error=self.detection.theta_error,
                M_z=self.detection.M,
                M_z_error=self.detection.M_uncertainty,
                cutoff_multiplier=2.0,
            )

            if possible_hosts is None:
                _LOGGER.debug("no possible hosts found...")
                continue
            possible_hosts, possible_hosts_with_bh_mass = possible_hosts
            _LOGGER.info(
                f"possible hosts found {len(possible_hosts)}/{len(possible_hosts_with_bh_mass)}..."
            )

            """
            if len(possible_hosts_with_bh_mass) == 0:
                detection_galaxy = _get_closest_possible_host(
                    self.detection, possible_hosts
                )
            else:
                detection_galaxy = _get_closest_possible_host(
                    self.detection, possible_hosts_with_bh_mass
                )

            self.detection.phi = detection_galaxy.phiS
            self.detection.theta = detection_galaxy.qS
            """

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
        # remove duplicates from possible_host_galaxies already covered in possible_host_galaxies_with_bh_mass

        possible_host_galaxies_reduced = [
            host
            for host in possible_host_galaxies
            if host not in possible_host_galaxies_with_bh_mass
        ]

        _LOGGER.debug(
            f"reduced possible hosts galaxies to unique, removed {len(possible_host_galaxies) - len(possible_host_galaxies_reduced)} galaxies."
        )

        chunksize = math.ceil(len(possible_host_galaxies_reduced) / pool._processes)
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
                    False,
                )
                for possible_host in possible_host_galaxies_reduced
            ],
            chunksize=chunksize,
        )
        end = time.time()
        _LOGGER.info(f"parallel computing took: {end - start}s")

        galaxy_likelihoods = list(
            zip(
                [
                    galaxy.catalog_index
                    for galaxy in possible_host_galaxies_with_bh_mass
                ],
                results_with_bh_mass,
            )
        )

        self.posterior_data_with_bh_mass[GALAXY_LIKELIHOODS][
            detection_index
        ] = galaxy_likelihoods

        if len(results) == 0:
            print("no results found")
            return 0.0, 0.0

        results.extend([result[0] for result in results_with_bh_mass])

        likelihood_without_bh_mass = np.sum(results)

        if len(results_with_bh_mass) == 0:
            return likelihood_without_bh_mass, 0.0

        likelihood_with_bh_mass = np.sum([result[1] for result in results_with_bh_mass])

        return likelihood_without_bh_mass, likelihood_with_bh_mass


def use_detection(detection: Detection) -> bool:
    sky_localization_uncertainty = _sky_localization_uncertainty(
        phi_error=detection.phi_error,
        theta=detection.theta,
        theta_error=detection.theta_error,
        cov_theta_phi=detection.theta_phi_covariance,
    )
    distance_relative_error = detection.d_L_uncertainty / detection.d_L

    if distance_relative_error < 0.05:
        return True
    _LOGGER.info(
        f"Detection skipped: distance_relative_error {distance_relative_error}, sky_localization_uncertainty {sky_localization_uncertainty}"
    )
    return False


def gaussian(x: float, mu: float, sigma: float, a: float) -> float:
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _sky_localization_uncertainty(
    phi_error: float, theta: float, theta_error: float, cov_theta_phi: float
) -> float:
    return (
        2
        * np.pi
        * np.abs(np.sin(theta))
        * np.sqrt(phi_error**2 * theta_error**2 - cov_theta_phi**2)
    )


def single_host_likelihood(
    possible_host: HostGalaxy,
    detection: Detection,
    h: float,
    evaluate_with_bh_mass: bool,
) -> list[float]:
    global max_redshift
    global max_mass
    global delta_redshift
    global redshift_skylocalization_histogram
    global redshift_skylocalization_mass_histogram
    """WL_uncertainty = (
        d_L * 0.066 * (1 - (1 + possible_host.z) ** (-0.25) / 0.25) ** (1.8)
    )"""  # TODO check if correct
    start = time.time()
    # redshift samples around peak
    z_lower_bound = possible_host.z - 4 * possible_host.z_error
    if z_lower_bound < 0:
        # print(f"lower bound is less than 0: {z_lower_bound}", flush=True)
        z_lower_bound = 0.0
    z_upper_bound = possible_host.z + 4 * possible_host.z_error
    if z_upper_bound > max_redshift:
        # print(f"upper bound is greater than max redshift: {z_upper_bound}", flush=True)
        z_upper_bound = max_redshift
    z_gws = np.arange(z_lower_bound, z_upper_bound, delta_redshift)
    distances = [dist(redshift, h=h) for redshift in z_gws]
    phis = np.ones(z_gws.shape) * possible_host.phiS
    thetas = np.ones(z_gws.shape) * possible_host.qS
    masses = possible_host.M * (1 + z_gws)

    # TODO: use delta_redshift to regard limits on the errors of redshift and distance in gaussians
    # TODO: also adjust delta redshift usage in evaluation

    # get distribution values for parameters
    parameters = np.array([distances, phis, thetas]).T
    parameters_with_bh_mass = np.array([distances, phis, thetas, masses/max_mass]).T

    # get distribution values
    redshift_detection_distribution_weights = np.array(
        np.sum(
            [
                gaussian.pdf(parameters)
                for gaussian in redshift_skylocalization_distribution
            ],
            axis=0,
        )
    ) / len(redshift_skylocalization_distribution)

    redshift_mass_detection_distribution_weights = np.array(
        np.sum(
            [
                gaussian.pdf(parameters_with_bh_mass)
                for gaussian in redshift_skylocalization_mass_distribution
            ],
            axis=0,
        )
    ) / len(redshift_skylocalization_mass_distribution)

    # checking numerical limits due to delta_redshift for gaussian variances
    luminosity_distance_resolution_limit = C / H_MIN / KM_TO_M / GPC_TO_MPC * delta_redshift / 2
    redshift_resoultion_limit = delta_redshift / 2
    mass_resolution_limit = detection.M * delta_redshift / ((1 + delta_redshift)) / 2

    if detection.d_L_uncertainty < luminosity_distance_resolution_limit:
        print(f"numeric luminosity distance resolution limit reached: {detection.d_L_uncertainty} < {luminosity_distance_resolution_limit}", flush=True)
        detection.d_L_uncertainty = luminosity_distance_resolution_limit

    if possible_host.z_error < redshift_resoultion_limit:
        print(f"numeric redshift resolution limit reached: {possible_host.z_error} < {redshift_resoultion_limit}", flush=True)
        possible_host.z_error = redshift_resoultion_limit
    
    if possible_host.M_error < mass_resolution_limit:
        print(f"numeric mass resolution limit reached: {possible_host.M_error} < {mass_resolution_limit}", flush=True)
        possible_host.M_error = mass_resolution_limit


    # multivariate normal distribution for all parameters including the mass
    if np.isnan(possible_host.M):
        # print(f"possible host has no mass information: {possible_host}", flush=True)
        possible_host.M = 0.0
        possible_host.M_error = 1.0
    covariance = [
        [
            detection.phi_error**2,
            detection.theta_phi_covariance,
            detection.d_L_phi_covariance,
        ],
        [
            detection.theta_phi_covariance,
            detection.theta_error**2,
            detection.d_L_theta_covariance,
        ],
        [
            detection.d_L_phi_covariance,
            detection.d_L_theta_covariance,
            detection.d_L_uncertainty**2,
        ],
    ]

    redshift_normal_distribution = NormalDist(
        mu=possible_host.z, sigma=possible_host.z_error
    )
    redshift_normal_distribution = np.array(
        [redshift_normal_distribution.pdf(redshift) for redshift in z_gws]
    )

    normal_distribution = multivariate_normal(
        mean=[
            detection.phi,
            detection.theta,
            detection.d_L,
        ],
        cov=covariance,
    )

    # prepare positions for multivariate normal distribution
    positions = np.vstack(
        [
            np.ones(z_gws.shape) * possible_host.phiS,
            np.ones(z_gws.shape) * possible_host.qS,
            distances,
        ]
    ).T

    # evaluate multivariate normal distribution
    likelihood_without_bh_mass = (
        normal_distribution.pdf(positions) * redshift_normal_distribution
    )

    # weight with redshift detection distribution
    likelihood_without_bh_mass_weighted = (
        likelihood_without_bh_mass * redshift_detection_distribution_weights
    )

    # integrate over redshift
    likelihood_without_bh_mass_weighted = np.trapz(
        likelihood_without_bh_mass_weighted, z_gws
    )

    if evaluate_with_bh_mass:
        covariance = [
            [
                detection.phi_error**2,
                detection.theta_phi_covariance,
                detection.d_L_phi_covariance,
                detection.M_phi_covariance,
            ],
            [
                detection.theta_phi_covariance,
                detection.theta_error**2,
                detection.d_L_theta_covariance,
                detection.M_theta_covariance,
            ],
            [
                detection.d_L_phi_covariance,
                detection.d_L_theta_covariance,
                detection.d_L_uncertainty**2,
                detection.d_L_M_covariance,
            ],
            [
                detection.M_phi_covariance,
                detection.M_theta_covariance,
                detection.d_L_M_covariance,
                detection.M_uncertainty**2,
            ],
        ]
        normal_distribution_with_mass = multivariate_normal(
            mean=[
                detection.phi,
                detection.theta,
                detection.d_L,
                detection.M,
            ],
            cov=covariance,
        )

        # treat redshifted mass peak as delta function # TODO: check if correct
        M_g = detection.M / (1 + z_gws)

        mass_normal_distribution = NormalDist(
            mu=possible_host.M, sigma=possible_host.M_error
        )

        mass_normal_distribution = np.array(
            [mass_normal_distribution.pdf(mass) for mass in M_g]
        )

        # prepare positions for multivariate normal distribution for all parameters including the mass
        positions = np.vstack(
            [
                np.ones(z_gws.shape) * possible_host.phiS,
                np.ones(z_gws.shape) * possible_host.qS,
                distances,
                np.ones(z_gws.shape) * detection.M,
            ]
        ).T

        likelihood_with_bh_mass = (
            normal_distribution_with_mass.pdf(positions)
            * mass_normal_distribution
            * redshift_normal_distribution
        )

        # weight with redshift detection distribution

        likelihood_with_bh_mass_weighted = (
            likelihood_with_bh_mass * redshift_mass_detection_distribution_weights
        )

        # integrate over mass and redshift
        likelihood_with_bh_mass_weighted = np.trapz(
            likelihood_with_bh_mass_weighted, z_gws
        )
        return [likelihood_without_bh_mass_weighted, likelihood_with_bh_mass_weighted]
    return likelihood_without_bh_mass_weighted


def child_process_init(
    current_max_redshift: float,
    current_max_mass: float,
    current_delta_redshift: float,
    current_redshift_skylocalization_distribution: list,
    current_redshift_skylocalization_mass_distribution: list,
) -> None:
    global max_redshift
    global max_mass
    global delta_redshift
    global redshift_skylocalization_distribution
    global redshift_skylocalization_mass_distribution
    max_redshift = current_max_redshift
    max_mass = current_max_mass
    delta_redshift = current_delta_redshift
    redshift_skylocalization_distribution = (
        current_redshift_skylocalization_distribution
    )
    redshift_skylocalization_mass_distribution = (
        current_redshift_skylocalization_mass_distribution
    )


def check_overflow(arr: np.array) -> bool:
    return np.any(np.isinf(arr))


def _get_closest_possible_host(
    detection: Detection, possible_hosts: List[HostGalaxy]
) -> HostGalaxy:
    distances = [
        _distance_spherical_coordinates(
            phi1=detection.phi,
            theta1=detection.theta,
            phi2=host.phiS,
            theta2=host.qS,
        )
        for host in possible_hosts
    ]
    return possible_hosts[np.argmin(distances)]


def _distance_spherical_coordinates(
    phi1: float, theta1: float, phi2: float, theta2: float
) -> float:
    return np.arccos(
        np.sin(theta1) * np.sin(theta2)
        + np.cos(theta1) * np.cos(theta2) * np.cos(phi1 - phi2)
    )


def compute_sigma_deviation(
    sigma: float, sigma_error: float, h_mean: float, h_mean_error: float
) -> float:
    sigma_dev = (h_mean - H) / sigma
    sigma_dev_error = (
        np.sqrt((sigma_error * sigma_dev) ** 2 + (h_mean_error) ** 2) / sigma
    )
    return sigma_dev, sigma_dev_error
