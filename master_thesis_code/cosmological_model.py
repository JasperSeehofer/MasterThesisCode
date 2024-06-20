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
from scipy.optimize import curve_fit
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
    CRAMER_RAO_BOUNDS_OUTPUT_PATH,
    PREPARED_CRAMER_RAO_BOUNDS_PATH,
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
            self.phi = (np.random.normal(self.phi, self.phi_error) + (2 * np.pi)) % (
                2 * np.pi
            )
            if 0 <= self.phi <= 2 * np.pi:
                break
        while True:
            self.theta = np.random.normal(self.theta, self.theta_error)
            if 0 <= self.theta <= np.pi:
                break
        while True:
            self.d_L = np.random.normal(self.d_L, self.d_L_uncertainty)
            if 0 <= self.d_L <= 6.8:
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

        self.parameter_space.M.lower_limit = 10 ** (4.0)
        self.parameter_space.M.upper_limit = 10 ** (6.5)

        self.parameter_space.a.value = 0.98
        self.parameter_space.a.is_fixed = True

        self.parameter_space.mu.value = 10
        self.parameter_space.mu.is_fixed = True

        self.parameter_space.e0.upper_limit = 0.2

        self.parameter_space.dist.upper_limit = dist(redshift=3.0)

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
        if not 0 < redshift < dist_to_redshift(self.parameter_space.dist.upper_limit):
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
        p0_redshift = np.random.rand(nwalkers, 1) * dist_to_redshift(
            self.parameter_space.dist.upper_limit
        )
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
            distances_subset = np.random.choice(distances, 50, replace=False)
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
        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        fig.suptitle(
            "Posterior distribution of Hubble constant h using several subsets of 50 detections"
        )
        # create color list with 10 different colors
        NUMBER_OF_SUBSETS = 10
        # create a colormap
        cmap = plt.cm.get_cmap(
            "viridis", NUMBER_OF_SUBSETS
        )  # 'viridis' is the colormap name, 10 is the number of colors

        # create a list of 10 colors from the colormap
        colors = [cmap(i) for i in range(cmap.N)]

        for count in range(NUMBER_OF_SUBSETS):
            _LOGGER.info(f"Creating subset {count}")
            posteriors_data_subset_indices = np.random.choice(
                list(self.posterior_data.keys()), 50, replace=False
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
                sub_posteriors *= np.array(posterior)
            for index, posterior in posteriors_data_with_bh_mass_subset:
                if check_overflow(sub_posteriors_with_bh_mass * np.array(posterior)):
                    # print("Overflow detected")
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
                popt, perr = curve_fit(
                    gaussian,
                    temp_h_values,
                    sub_posteriors,
                    p0=[H, 0.1, 1],
                )

                popt_with_bh_mass, perr_with_bh_mass = curve_fit(
                    gaussian,
                    temp_h_values_with_bh_mass,
                    sub_posteriors_with_bh_mass,
                    p0=[H, 0.1, 1],
                )
                ax[0].plot(
                    h_values_fine,
                    gaussian(h_values_fine, *popt),
                    label=f"std: {np.round(popt[1], 3)}, mean: {np.round(popt[0], 3)}",
                    color=colors[count],
                    linestyle="--",
                )
                ax[1].plot(
                    h_values_fine,
                    gaussian(h_values_fine, *popt_with_bh_mass),
                    label=f"std: {np.round(popt_with_bh_mass[1], 3)}, mean: {np.round(popt_with_bh_mass[0], 3)}",
                    color=colors[count],
                    linestyle="--",
                )
            except RuntimeError:
                pass

            ax[0].scatter(
                temp_h_values,
                sub_posteriors,
                label="without BH mass",
                color=colors[count],
                s=2,
            )
            ax[1].scatter(
                temp_h_values_with_bh_mass,
                sub_posteriors_with_bh_mass,
                label="with BH mass",
                color=colors[count],
                s=2,
            )
        # print true value
        ax[0].axvline(H, color="g", linestyle="--")
        ax[0].set_xlabel("Hubble constant h")
        ax[0].set_ylabel("Posterior")
        ax[0].set_title(f"without BH mass")
        ax[1].axvline(H, color="g", linestyle="--")
        ax[1].set_xlabel("Hubble constant h")
        ax[1].set_ylabel("Posterior")
        ax[1].set_title(f"with BH mass")
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
            popt, perr = curve_fit(
                gaussian,
                self.h_values,
                posteriors,
                p0=[H, 0.1, 1],
            )

            popt_with_bh_mass, perr_with_bh_mass = curve_fit(
                gaussian,
                self.h_values_with_bh_mass,
                posteriors_with_bh_mass,
                p0=[H, 0.1, 1],
            )
            plt.plot(
                h_fine,
                gaussian(h_fine, *popt),
                label=f"std: {np.round(popt[1], 3)}, mean: {np.round(popt[0], 3)}",
                color="b",
                linestyle="--",
            )
            plt.plot(
                h_fine,
                gaussian(h_fine, *popt_with_bh_mass),
                label=f"std: {np.round(popt_with_bh_mass[1], 3)}, mean: {np.round(popt_with_bh_mass[0], 3)}",
                color="r",
                linestyle="--",
            )
        except RuntimeError:
            pass

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
        plt.legend()
        plt.savefig("saved_figures/bayesian_statistics.png")
        plt.close()

        # self.visualize_galaxy_weights(galaxy_catalog)

    def visualize_galaxy_weights(self, galaxy_catalog: GalaxyCatalogueHandler) -> None:
        _LOGGER.info("Visualizing galaxy weights...")
        # visualize galaxy weights
        # restructure galaxy weights data
        weight_data = {}
        h_values = self.h_values_with_bh_mass
        for h, data in self.galaxy_weights.items():
            for detection_index, host_galaxy_weights in data.items():
                try:
                    weight_data[int(detection_index)].append(host_galaxy_weights)
                except KeyError:
                    if len(host_galaxy_weights) == 0:
                        continue
                    weight_data[int(detection_index)] = [host_galaxy_weights]

        # remove weight_data with less samples than h_values
        weight_data = {
            detection_index: host_galaxy_weights
            for detection_index, host_galaxy_weights in weight_data.items()
            if len(host_galaxy_weights) == len(h_values)
        }

        for detection_index, host_galaxy_weights_by_h_value in weight_data.items():
            _LOGGER.info(f"Visualizing galaxy weights for detection {detection_index}")
            detection = Detection(self.cramer_rao_bounds.iloc[int(detection_index)])
            true_galaxy = Detection(
                self.true_cramer_rao_bounds.iloc[int(detection_index)]
            )
            true_galaxy_index = int(
                self.true_cramer_rao_bounds.iloc[int(detection_index)][
                    "host_galaxy_index"
                ]
            )

            # plot h values on x axis and weights on y axis and the sum of weights
            fig, axs = plt.subplots(2, 3, figsize=(16, 9))
            # figure title
            fig.suptitle(f"Galaxy weight visualization for detection {detection_index}")

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

                detection_likelihood = np.sum(
                    unweighted_likelihoods * weights
                ) / np.sum(weights)
                detection_likelihood_bh_mass = np.sum(
                    unweighted_likelihoods * weights_bh_mass * weights
                ) / np.sum(weights * weights_bh_mass)

                if np.round(h_value, 2) == H:
                    # plot resulting sum of weights
                    axs[0, 0].scatter(
                        host_galaxies_phi,
                        host_galaxies_theta,
                        s=weights / max(weights) * 100,
                        c=weights,
                        cmap="viridis",
                    )
                    axs[0, 0].set_title(f"Weights for h = {H}")
                    axs[0, 1].scatter(
                        host_galaxies_phi,
                        host_galaxies_theta,
                        s=weights_bh_mass / max(weights_bh_mass) * 100,
                        c=weights_bh_mass,
                        cmap="viridis",
                    )
                    axs[0, 1].set_title(f"BH mass weight for h = {H}")
                    axs[0, 2].scatter(
                        host_galaxies_phi,
                        host_galaxies_theta,
                        s=weights
                        * weights_bh_mass
                        / max(weights * weights_bh_mass)
                        * 100,
                        c=weights * weights_bh_mass,
                        cmap="viridis",
                    )
                    axs[0, 2].set_title(
                        f"Combined weights (product of weights) for h = {H}"
                    )

                axs[1, 0].scatter(
                    [h_value],
                    detection_likelihood,
                    c="b",
                    label="without BH mass",
                )
                axs[1, 0].scatter(
                    [h_value],
                    detection_likelihood_bh_mass,
                    c="r",
                    label="with BH mass",
                )

                # plot weights of true galaxy in detection
                true_galaxy_weights = [
                    weights
                    for index, weights in host_galaxy_weights
                    if int(index) == true_galaxy_index
                ]
                try:
                    true_galaxy_weights = true_galaxy_weights[0]
                    true_galaxy_likelihood = true_galaxy_weights[0]
                    true_galaxy_weight = true_galaxy_weights[1]
                    true_galaxy_weight_bh_mass = true_galaxy_weights[2]
                except IndexError:
                    true_galaxy_likelihood = 0
                    true_galaxy_weight = 0
                    true_galaxy_weight_bh_mass = 0

                host_galaxy_indices = [int(index) for index, _ in host_galaxy_weights]
                zipped_likelihood = list(
                    zip(host_galaxy_indices, unweighted_likelihoods * weights)
                )
                zipped_likelihood.sort(key=lambda x: x[1], reverse=True)
                ranked_indices, ranked_likelihood = zip(*zipped_likelihood)
                # find index of true galaxy
                try:
                    true_galaxy_ranking_index = ranked_indices.index(
                        true_galaxy_index, -1
                    )
                except ValueError:
                    true_galaxy_ranking_index = -1

                axs[1, 1].scatter(
                    [h_value],
                    true_galaxy_likelihood * true_galaxy_weight,
                    c="b",
                    label=f"true galaxy rank {true_galaxy_ranking_index + 1}.",
                )

                zipped_likelihood_bh_mass = list(
                    zip(
                        host_galaxy_indices,
                        unweighted_likelihoods * weights_bh_mass * weights,
                    )
                )
                zipped_likelihood_bh_mass.sort(key=lambda x: x[1], reverse=True)
                ranked_indices_bh_mass, ranked_likelihood_bh_mass = zip(
                    *zipped_likelihood_bh_mass
                )
                try:
                    true_galaxy_ranking_index_bh_mass = ranked_indices_bh_mass.index(
                        true_galaxy_index,
                    )
                except ValueError:
                    true_galaxy_ranking_index_bh_mass = -1
                axs[1, 2].scatter(
                    [h_value],
                    true_galaxy_likelihood * true_galaxy_weight_bh_mass,
                    c="r",
                    label=f"true galaxy rank {true_galaxy_ranking_index_bh_mass + 1}.",
                )

                # plot detection lines and true lines
                for column in range(3):
                    axs[0, column].axvline(
                        detection.phi, color="black", linestyle="-.", label="detection"
                    )
                    axs[0, column].axhline(detection.theta, color="r", linestyle="--")
                    axs[0, column].axvline(
                        true_galaxy.phi, color="g", linestyle="--", label="true"
                    )
                    axs[0, column].axhline(true_galaxy.theta, color="g", linestyle="--")
                    axs[0, column].set_xlabel("phi in rad")
                    axs[0, column].set_ylabel("theta in rad")
            axs[1, 0].set_title(f"detection likelihood")
            # plot detection lines and true lines for plot 1, 0
            axs[1, 0].axvline(H, color="g", linestyle="--")
            axs[1, 1].set_title(f"true galaxy weighted likelihood without BH mass")
            axs[1, 1].axvline(H, color="g", linestyle="--")
            axs[1, 2].set_title(f"true galaxy weighted likelihood with BH mass")
            axs[1, 2].axvline(H, color="g", linestyle="--")
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

        self.h = h_value
        _LOGGER.info("prepare global variable for multiprocessing")
        distances = [dist_to_redshift(dist) for dist in self.cramer_rao_bounds["dist"]]
        _LOGGER.debug(f"distances: {distances}.")

        self._max_redshift = np.max(distances)

        self._redshift_distribution = np.histogram(
            np.array(
                distances,
            ),
            bins=np.linspace(0, max(distances), int(max(distances) * 100)),
        )[0]

        # scale list such that it has the same length as the number of z_gws values
        self._redshift_distribution = np.array(
            [
                value / np.sum(self._redshift_distribution)
                for value in self._redshift_distribution
            ]
        )
        SCALING_FACTOR = 100
        self._redshift_distribution = np.array(
            [np.full(SCALING_FACTOR, value) for value in self._redshift_distribution]
        ).flatten()

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
            initargs=(self._max_redshift,),
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
            # convert true mass to redshifted mass because that is the observed mass
            detection["M"] = convert_true_mass_to_redshifted_mass_with_distance(
                detection["M"], detection["dist"]
            )
            self.detection = Detection(detection)
            # self.detection.convert_to_best_guess_parameters()

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
            return 0.0, 0.0

        results.extend([result[0] for result in results_with_bh_mass])

        likelihood_without_bh_mass = np.sum(results) / float(
            len(results) + len(results_with_bh_mass)
        )

        if len(results_with_bh_mass) == 0:
            return likelihood_without_bh_mass, 0.0

        likelihood_with_bh_mass = np.sum(
            [result[1] for result in results_with_bh_mass]
        ) / float(len(results_with_bh_mass))

        return likelihood_without_bh_mass, likelihood_with_bh_mass


def use_detection(detection: Detection) -> bool:
    sky_localization_uncertainty = _sky_localization_uncertainty(
        phi_error=detection.phi_error,
        theta=detection.theta,
        theta_error=detection.theta_error,
        cov_theta_phi=detection.theta_phi_covariance,
    )
    distance_relative_error = detection.d_L_uncertainty / detection.d_L

    if (distance_relative_error < 0.1) and (sky_localization_uncertainty < 0.01):
        return True
    _LOGGER.info(
        f"Detection skipped: distance_relative_error {distance_relative_error}, sky_localization_uncertainty {sky_localization_uncertainty}"
    )
    return False


def gaussian(x: float, mu: float, sigma: float, a: float) -> float:
    return 1 / a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


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
    """WL_uncertainty = (
        d_L * 0.066 * (1 - (1 + possible_host.z) ** (-0.25) / 0.25) ** (1.8)
    )"""  # TODO check if correct

    # redshift samples around peak
    z_lower_bound = possible_host.z - 5 * possible_host.z_error
    if z_lower_bound < 0:
        # print(f"lower bound is less than 0: {z_lower_bound}", flush=True)
        z_lower_bound = 0.0
    z_upper_bound = possible_host.z + 5 * possible_host.z_error
    if z_upper_bound > max_redshift:
        # print(f"upper bound is greater than max redshift: {z_upper_bound}", flush=True)
        z_upper_bound = max_redshift
    z_gws = np.linspace(
        z_lower_bound,
        z_upper_bound,
        1000,
    )
    distances = [dist(redshift, h=h) for redshift in z_gws]

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

    # weight with redshift distribution
    # likelihood_without_bh_mass = likelihood_without_bh_mass * redshift_distribution

    # integrate over redshift
    likelihood_without_bh_mass = np.trapz(likelihood_without_bh_mass, z_gws)

    if evaluate_with_bh_mass:
        """
        SAMPLING_POINTS = 20
        SIGMA_RANGE = 6
        M_gs_z = np.ones(shape=(2 * SAMPLING_POINTS + 1, len(z_gws))) * detection.M
        for i, factor in enumerate(range(-SAMPLING_POINTS, SAMPLING_POINTS, 1)):
            M_gs_z[i] = (
                M_gs_z[i]
                + factor * detection.M_uncertainty * SIGMA_RANGE / SAMPLING_POINTS
            )

        z_gw_grid = np.array([z_gws for _ in range(2 * SAMPLING_POINTS + 1)])
        distances_grid = np.array([distances for _ in range(2 * SAMPLING_POINTS + 1)])
        redshift_distribution_grid = np.array(
            [redshift_distribution for _ in range(2 * SAMPLING_POINTS + 1)]
        )

        M_g = M_gs_z / (1 + z_gw_grid)
        """
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
        # treat redshifted mass peak as delta function
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
                M_g * (1 + z_gws),
            ]
        ).T

        likelihood_with_bh_mass = (
            normal_distribution_with_mass.pdf(positions)
            * mass_normal_distribution
            * redshift_normal_distribution
        )

        # weight with redshift distribution
        """
        likelihood_with_bh_mass_grid = (
            likelihood_with_bh_mass_grid * redshift_distribution_grid
        )
        """

        # integrate over mass and redshift
        likelihood_with_bh_mass = np.trapz(likelihood_with_bh_mass, z_gws)

        return [likelihood_without_bh_mass, likelihood_with_bh_mass]
    return likelihood_without_bh_mass


def child_process_init(
    current_max_redshift: float,
) -> None:
    global max_redshift
    max_redshift = current_max_redshift


def check_overflow(arr: np.array) -> bool:
    return np.any(np.isinf(arr))
