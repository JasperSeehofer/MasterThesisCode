from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import json
import pandas as pd
import os
import math
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from scipy.stats import multivariate_normal, truncnorm, gaussian_kde, _multivariate, norm
from scipy.optimize import curve_fit
import emcee
from scipy.special import erf
from scipy.integrate import dblquad, quad, fixed_quad
from scipy.interpolate import RegularGridInterpolator



# import statsmodels.api as sm
from statistics import NormalDist
import multiprocessing as mp
from master_thesis_code.datamodels.parameter_space import (
    ParameterSpace,
    Parameter,
    uniform,
)
from master_thesis_code.M1_model_extracted_data.detection_fraction import (
    DetectionFraction,
)


from master_thesis_code.constants import (
    C,
    H0,
    H,
    H_MIN,
    CRAMER_RAO_BOUNDS_OUTPUT_PATH,
    PREPARED_CRAMER_RAO_BOUNDS_PATH,
    UNDETECTED_EVENTS_OUTPUT_PATH,
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
    dist_vectorized,
    cached_dist,
    dist_derivative,
    convert_true_mass_to_redshifted_mass_with_distance,
    get_redshift_outer_bounds,
    dist_to_redshift,
)

_LOGGER = logging.getLogger()

DEFAULT_GALAXY_Z_ERROR = 0.0015
GALAXY_LIKELIHOODS = "galaxy_likelihoods"
ADDITIONAL_GALAXIES_WITHOUT_BH_MASS = "additional_galaxies_without_bh_mass"

FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD = 0.05

# detection fraction LISA M1 model


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
        self.snr = parameters["SNR"]
        self.host_galaxy_index = parameters["host_galaxy_index"]

    def get_skylocalization_error(self) -> float:
        return _sky_localization_uncertainty(
            self.phi_error, self.theta, self.theta_error, self.theta_phi_covariance
        )

    def get_relative_distance_error(self) -> float:
        return self.d_L_uncertainty / self.d_L

    def convert_to_best_guess_parameters(self) -> None:

        self.phi = truncnorm(
            (0 - self.phi) / self.phi_error,
            (2 * np.pi - self.phi) / self.phi_error,
            loc=self.phi,
            scale=self.phi_error,
        ).rvs(1)[0]

        self.theta = truncnorm(
            (0 - self.theta) / self.theta_error,
            (np.pi - self.theta) / self.theta_error,
            loc=self.theta,
            scale=self.theta_error,
        ).rvs(1)[0]

        self.d_L = truncnorm(
            (0 - self.d_L) / self.d_L_uncertainty,
            (dist(1.5) - self.d_L) / self.d_L_uncertainty,
            loc=self.d_L,
            scale=self.d_L_uncertainty,
        ).rvs(1)[0]

        self.M = truncnorm(
            (1e4 - self.M) / self.M_uncertainty,
            (1e6 - self.M) / self.M_uncertainty,
            loc=self.M,
            scale=self.M_uncertainty,
        ).rvs(1)[0]


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
    detection_fraction = DetectionFraction()

    def __init__(self) -> None:
        self.parameter_space = ParameterSpace()
        self._apply_model_assumptions()
        self.setup_emri_events_sampler()

    def plot_expected_detection_distribution(self) -> None:
        redshift_range = np.linspace(0, self.max_redshift, 80)
        mass_range = np.logspace(4.5, 7, 100)

        emri_distribution = np.array(
            [
                [self.emri_distribution(mass, redshift) for mass in mass_range]
                for redshift in redshift_range
            ]
        )
        detection_fraction = np.array(
            [
                [
                    self.detection_fraction.get_detection_fraction(redshift, mass)
                    for mass in mass_range
                ]
                for redshift in redshift_range
            ]
        )
        print(detection_fraction.shape)
        product = np.multiply(emri_distribution, detection_fraction)
        print(product.shape)

        normalization = np.trapz(
            np.trapz(product, redshift_range, axis=0),
            mass_range,
            axis=0,
        )

        # integrate over galaxy covered volume
        max_redshift = 0.1
        max_redshift_index = np.argmin(np.abs(redshift_range - max_redshift))
        reduced_detection_distribution = (
            emri_distribution[:max_redshift_index]
            * detection_fraction[:max_redshift_index]
        )
        print(reduced_detection_distribution.shape)

        integral = (
            np.trapz(
                np.trapz(
                    reduced_detection_distribution, redshift_range[:max_redshift_index]
                ),
                mass_range,
            )
            * 100
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.contourf(
            redshift_range,
            mass_range,
            emri_distribution * detection_fraction / normalization,
            cmap="viridis",
            levels=50,
        )
        # show integrated area and volume
        ax.text(
            0.5,
            1e5,
            f"integrated area: {integral:.2f}%",
            horizontalalignment="center",
            verticalalignment="center",
        )

        plt.colorbar()
        plt.yscale("log")
        plt.xlabel("redshift")
        plt.ylabel("mass")
        plt.savefig(
            "saved_figures/cosmological_model/expected_detection_distribution.png"
        )

    def _apply_model_assumptions(self) -> None:

        self.parameter_space.M.lower_limit = 10 ** (4.5)
        self.parameter_space.M.upper_limit = 10 ** (6.0)

        self.parameter_space.a.value = 0.98
        self.parameter_space.a.is_fixed = True

        self.parameter_space.mu.value = 10
        self.parameter_space.mu.is_fixed = True

        self.parameter_space.e0.upper_limit = 0.2

        self.max_redshift = 1.5
        self.parameter_space.dist.upper_limit = dist(redshift=self.max_redshift)
        self.luminostity_detection_threshold = 1.55  # as in Hitchikers Guide

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
            fraction = (mass_bin - 6.0) / 0.5
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
        nwalkers = 20
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

        self.plot_expected_detection_distribution()

        masses = np.logspace(4, 7, 100)
        redshifts = np.linspace(0, self.max_redshift, 1000)
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

        plt.contourf(redshifts, masses, dN_dz_distribution, cmap="viridis", levels=30)
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

    def simplified_event_mass_dependency(self, mass: float) -> float:
        pass

    def setup_simplified_event_sampler(self) -> None:
        pass


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


class DetectionProbability:
    """Detection probability of a given event."""
    def __init__(
            self,
            luminosity_distance_lower_limit: float,
            luminosity_distance_upper_limit: float,
            mass_lower_limit: float,
            mass_upper_limit: float,
            detected_events: pd.DataFrame,
            undetected_events: pd.DataFrame,
            bandwidth: float | None = None,
    ) -> None:
        self.luminosity_distance_lower_limit = luminosity_distance_lower_limit
        self.luminosity_distance_upper_limit = luminosity_distance_upper_limit
        self.mass_lower_limit = mass_lower_limit
        self.mass_upper_limit = mass_upper_limit

        undetected_events_points = np.array(
            [
                (undetected_events["dist"] - luminosity_distance_lower_limit) / (
                    luminosity_distance_upper_limit - luminosity_distance_lower_limit
                ),
                (np.log10(undetected_events["M"]) - np.log10(mass_lower_limit)) / (
                    np.log10(mass_upper_limit) - np.log10(mass_lower_limit)
                ),
                undetected_events["phiS"] / (2 * np.pi),
                undetected_events["qS"] / np.pi,
            ]
        )

        detected_events_points = np.array(
            [
                (detected_events["dist"] - luminosity_distance_lower_limit) / (
                    luminosity_distance_upper_limit - luminosity_distance_lower_limit
                ),
                (np.log10(detected_events["M"]) - np.log10(mass_lower_limit)) / (
                    np.log10(mass_upper_limit) - np.log10(mass_lower_limit)
                ),
                detected_events["phiS"] / (2 * np.pi),
                detected_events["qS"] / np.pi,
            ]
        )

        self.kde_undetected_with_bh_mass = gaussian_kde(undetected_events_points, bw_method=bandwidth)
        self.kde_detected_with_bh_mass = gaussian_kde(detected_events_points, bw_method=bandwidth)

        # create kde and detection probability function for the case without BH mass
        undetected_events_points_without_bh_mass = np.delete(
            undetected_events_points, 1, axis=0
        )
        detected_events_points_without_bh_mass = np.delete(
            detected_events_points, 1, axis=0
        )

        self.kde_detected_without_bh_mass = gaussian_kde(detected_events_points_without_bh_mass, bw_method=bandwidth)
        self.kde_undetected_without_bh_mass = gaussian_kde(undetected_events_points_without_bh_mass, bw_method=bandwidth)

        self._setup_interpolator(d_L_steps=20, M_z_steps=30, phi_steps=20, theta_steps=20)

    def evaluate_with_bh_mass(
            self,
            d_L: float,
            M_z: float,
            phi: float,
            theta: float,
        ) -> float:
            # check if the input values are within the limits
            if any(
                [
                    d_L < self.luminosity_distance_lower_limit,
                    d_L > self.luminosity_distance_upper_limit,
                    M_z < self.mass_lower_limit,
                    M_z > self.mass_upper_limit,
                    phi < 0,
                    phi >= 2 * np.pi,
                    theta < 0,
                    theta > np.pi,
                ]):
                
                return 0.0
            # normalize the input values to the range [0, 1]
            d_L, M_z, phi, theta = self._normalize_parameters(
                d_L, phi, theta, M_z
            )

            detected_evaluated = self.kde_detected_with_bh_mass.evaluate([d_L, M_z, phi, theta])[0]
            undetected_evaluated = self.kde_undetected_with_bh_mass.evaluate([d_L, M_z, phi, theta])[0]
            if undetected_evaluated + detected_evaluated == 0.0:
                return 0.0
            return detected_evaluated / (undetected_evaluated + detected_evaluated)
    
    def evaluate_with_bh_mass_vectorized(
            self,
            d_L: np.ndarray[float],
            M_z: np.ndarray[float],
            phi: np.ndarray[float],
            theta: np.ndarray[float],
        ) -> np.ndarray[float]:
            # check if the input values are within the limits
        valid_mask = (
            (d_L >= self.luminosity_distance_lower_limit)
            & (d_L <= self.luminosity_distance_upper_limit)
            & (M_z >= self.mass_lower_limit)
            & (M_z <= self.mass_upper_limit)
            & (phi >= 0)
            & (phi < 2 * np.pi)
            & (theta >= 0)
            & (theta <= np.pi)
        )
            
        # Initialize result array with zeros (or another default value)
        probabilities = np.zeros_like(d_L, dtype=float)
        
        # Apply the mask to filter valid values
        d_L_valid = d_L[valid_mask]
        M_z_valid = M_z[valid_mask]
        phi_valid = phi[valid_mask]
        theta_valid = theta[valid_mask]
        
        # Normalize the valid parameters
        d_L_norm, M_z_norm, phi_norm, theta_norm = self._normalize_parameters(
        d_L_valid, phi_valid, theta_valid, M_z_valid
        )

        # Evaluate KDEs for valid values
        detected_evaluated = self.kde_detected_with_bh_mass.evaluate(
            np.vstack((d_L_norm, M_z_norm, phi_norm, theta_norm))
        )
        undetected_evaluated = self.kde_undetected_with_bh_mass.evaluate(
            np.vstack((d_L_norm, M_z_norm, phi_norm, theta_norm))
        )

        # Compute probabilities for valid values
        probabilities_valid = np.divide(
            detected_evaluated,
            detected_evaluated + undetected_evaluated,
            out=np.zeros_like(detected_evaluated),
            where=(detected_evaluated + undetected_evaluated) != 0,
        )

        # Assign probabilities back to the result array
        probabilities[valid_mask] = probabilities_valid

        return probabilities
                    
    def evaluate_without_bh_mass(
            self,
            d_L: float,
            phi: float,
            theta: float,
        ) -> float:
        # check if the input values are within the limits
        if any(
            [
                d_L < self.luminosity_distance_lower_limit,
                d_L > self.luminosity_distance_upper_limit,
                phi < 0,
                phi > 2 * np.pi,
                theta < 0,
                theta > np.pi,
            ]):
            return 0.0
        # normalize the input values to the range [0, 1]
        d_L, phi, theta = self._normalize_parameters(
            d_L, phi, theta
        )
        detected_evaluated = self.kde_detected_without_bh_mass.evaluate([d_L, phi, theta])[0]
        undetected_evaluated = self.kde_undetected_without_bh_mass.evaluate([d_L, phi, theta])[0]
        if undetected_evaluated + detected_evaluated == 0:
            return 0.0
        return detected_evaluated / (undetected_evaluated + detected_evaluated)

    def _normalize_parameters(
            self,
            d_L: float,
            phi: float,
            theta: float,
            M_z: Optional[float] = None,
        ) -> Union[Tuple[float, float, float, float], Tuple[float, float, float]]:
        # normalize the input values to the range [0, 1]
        d_L = (d_L - self.luminosity_distance_lower_limit) / (
            self.luminosity_distance_upper_limit - self.luminosity_distance_lower_limit
        )
        phi = phi / (2 * np.pi)
        theta = theta / np.pi
        if M_z is not None:
            M_z = (np.log10(M_z) - np.log10(self.mass_lower_limit)) / (
                np.log10(self.mass_upper_limit) - np.log10(self.mass_lower_limit)
            )
            return d_L, M_z, phi, theta
        return d_L, phi, theta

    def plot_detection_probability(self) -> None:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
        fig.suptitle("Detection probability")
        # plot detection probability for d_L and M
        d_L_range = np.linspace(
            self.luminosity_distance_lower_limit,
            self.luminosity_distance_upper_limit,
            100,
        )
        M_range = np.geomspace(
            self.mass_lower_limit, self.mass_upper_limit, 150
        )
        fixed_phi = 3 * np.pi / 2
        fixed_theta = np.pi / 2

        detected_events_with_bh_mass = np.array(
            [
                [self.kde_detected_with_bh_mass.evaluate(list(self._normalize_parameters(d_L, fixed_phi, fixed_theta, M)))[0] for M in M_range]
                for d_L in d_L_range
            ]
        )

        ax[0,0].contourf(
            d_L_range,
            M_range,
            detected_events_with_bh_mass.T,
            cmap="viridis",
            levels=50,
        )
        ax[0,0].set_xlabel("Luminosity distance [Gpc]")
        ax[0,0].set_ylabel("Mass  [M_solar]")
        ax[0,0].set_title("Detected events with BH mass")
        ax[0,0].set_yscale("log")
        ax[0,0].set_xlim(
            self.luminosity_distance_lower_limit,
            self.luminosity_distance_upper_limit,
        )
        ax[0,0].set_ylim(self.mass_lower_limit, self.mass_upper_limit)

        # plot undetected events with BH mass
        undetected_events_with_bh_mass = np.array(
            [
                [self.kde_undetected_with_bh_mass.evaluate(list(self._normalize_parameters(d_L, fixed_phi, fixed_theta, M)))[0] for M in M_range]
                for d_L in d_L_range
            ]
        )
        ax[0,1].contourf(
            d_L_range,
            M_range,
            undetected_events_with_bh_mass.T,
            cmap="viridis",
            levels=50,
        )
        ax[0,1].set_xlabel("Luminosity distance [Gpc]")
        ax[0,1].set_ylabel("Mass  [M_solar]")
        ax[0,1].set_title("Undetected events with BH mass")
        ax[0,1].set_yscale("log")
        ax[0,1].set_xlim(
            self.luminosity_distance_lower_limit,
            self.luminosity_distance_upper_limit,
        )
        ax[0,1].set_ylim(self.mass_lower_limit, self.mass_upper_limit)

        detection_probability_with_bh_mass = np.array(
            [
                [self.evaluate_with_bh_mass(d_L, M, fixed_phi, fixed_theta) for M in M_range]
                for d_L in d_L_range
            ]
        )
        
        # plot detection probability for d_L and M
        ax[1,0].contourf(
            d_L_range,
            M_range,
            detection_probability_with_bh_mass.T,
            cmap="viridis",
            levels=50,
        )
        ax[1,0].set_xlabel("Luminosity distance [Gpc]")
        ax[1,0].set_ylabel("Mass  []")
        ax[1,0].set_title("Detection probability with BH mass")
        ax[1,0].set_yscale("log")
        ax[1,0].set_xlim(
            self.luminosity_distance_lower_limit,
            self.luminosity_distance_upper_limit,
        )
        ax[1,0].set_ylim(self.mass_lower_limit, self.mass_upper_limit)
        
        # plot detection probability for d_L and M
        # evaluate for 10 different random phi and theta values
        phi_values = np.linspace(0, 2 * np.pi, 10)
        theta_values = np.linspace(0, np.pi, 10)
        for phi, theta in zip(phi_values,theta_values):
            detection_probability_without_bh_mass = np.array(
                [
                    self.evaluate_without_bh_mass(d_L, phi, theta) for d_L in d_L_range
                ]
            )
            
            ax[1,1].plot(
            d_L_range,
            detection_probability_without_bh_mass,
        )
        ax[1,1].set_xlabel("Luminosity distance")
        ax[1,1].set_ylabel("Detection probability")
        ax[1,1].set_title("Detection probability without BH mass")
        ax[1,1].set_xlim(
            self.luminosity_distance_lower_limit,
            self.luminosity_distance_upper_limit,
        )
        ax[1,1].legend()
        plt.savefig("saved_figures/cosmological_model/detection_probability.png")
        plt.show()
        plt.close()
        _LOGGER.info("Detection probability plot saved.")

    def _setup_interpolator(self, d_L_steps: int, M_z_steps: int, phi_steps: int, theta_steps: int) -> None:
        # setup grid
        d_L_range = np.linspace(
            self.luminosity_distance_lower_limit,
            self.luminosity_distance_upper_limit,
            d_L_steps,
        )
        M_z_range = np.geomspace(
            self.mass_lower_limit, self.mass_upper_limit, M_z_steps
        )
        phi_range = np.linspace(0, 2 * np.pi, phi_steps)
        theta_range = np.linspace(0, np.pi, theta_steps)

        # normalize the ranges to [0, 1]
        d_L_range_norm, M_z_range_norm, phi_range_norm, theta_range_norm = self._normalize_parameters(
            d_L_range, phi_range, theta_range, M_z_range
        )

        # create meshgrid
        d_L_grid, phi_grid, theta_grid = np.meshgrid(
            d_L_range_norm, phi_range_norm, theta_range_norm, indexing="ij"
        )

        # flatten the grid
        d_L_flat = d_L_grid.flatten()
        phi_flat = phi_grid.flatten()
        theta_flat = theta_grid.flatten()

        # evaluate the kde for the grid points without bh mass
        kde_values_without_bh_mass = self.kde_detected_without_bh_mass.evaluate(
            np.vstack((d_L_flat, phi_flat, theta_flat))
        )
        kde_values_undetected_without_bh_mass = self.kde_undetected_without_bh_mass.evaluate(
            np.vstack((d_L_flat, phi_flat, theta_flat))
        )

        detection_probabilities_without_bh_mass = np.divide(
            kde_values_without_bh_mass,
            kde_values_without_bh_mass + kde_values_undetected_without_bh_mass,
            out=np.zeros_like(kde_values_without_bh_mass),
            where=(kde_values_without_bh_mass + kde_values_undetected_without_bh_mass) != 0,
        )

        # reshape to grid
        detection_probabilities_without_bh_mass = detection_probabilities_without_bh_mass.reshape(
            d_L_steps, phi_steps, theta_steps
        )

        self.detection_probability_without_bh_mass_interpolator = RegularGridInterpolator(
            (d_L_range, phi_range, theta_range),
            detection_probabilities_without_bh_mass,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        # evaluate the kde for the grid points with bh mass
        d_L_grid, M_z_grid, phi_grid, theta_grid = np.meshgrid(
            d_L_range_norm, M_z_range_norm, phi_range_norm, theta_range_norm, indexing="ij"
        )
        d_L_flat = d_L_grid.flatten()
        M_z_flat = M_z_grid.flatten()
        phi_flat = phi_grid.flatten()
        theta_flat = theta_grid.flatten()

        kde_values_with_bh_mass = self.kde_detected_with_bh_mass.evaluate(
            np.vstack((d_L_flat, M_z_flat, phi_flat, theta_flat))
        )
        kde_values_undetected_with_bh_mass = self.kde_undetected_with_bh_mass.evaluate(
            np.vstack((d_L_flat, M_z_flat, phi_flat, theta_flat))
        )
        detection_probabilities_with_bh_mass = np.divide(
            kde_values_with_bh_mass,
            kde_values_with_bh_mass + kde_values_undetected_with_bh_mass,
            out=np.zeros_like(kde_values_with_bh_mass),
            where=(kde_values_with_bh_mass + kde_values_undetected_with_bh_mass) != 0,
        )
        # reshape to grid
        detection_probabilities_with_bh_mass = detection_probabilities_with_bh_mass.reshape(
            d_L_steps, M_z_steps, phi_steps, theta_steps
        )

        self.detection_probability_with_bh_mass_interpolator = RegularGridInterpolator(
            (d_L_range, M_z_range, phi_range, theta_range),
            detection_probabilities_with_bh_mass,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    def detection_probability_with_bh_mass_interpolated(
            self,
            d_L: float,
            M_z: float,
            phi: float,
            theta: float,
        ) -> float:       
        # check if the input values are within the limits
        return self.detection_probability_with_bh_mass_interpolator(
            (d_L, M_z, phi, theta)
        )[0]
    
    def detection_probability_without_bh_mass_interpolated(
            self,
            d_L: float,
            phi: float,
            theta: float,
        ) -> float:
        return self.detection_probability_without_bh_mass_interpolator(
            (d_L, phi, theta)
        )[0]


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
    additional_galaxies_without_bh_mass = {}
    posterior_data: Dict[int, List[float]] = {}
    posterior_data_with_bh_mass: Dict[int, List[float]] = {}

    def __init__(self) -> None:
        self.cramer_rao_bounds = pd.read_csv(PREPARED_CRAMER_RAO_BOUNDS_PATH)
        self.true_cramer_rao_bounds = pd.read_csv(CRAMER_RAO_BOUNDS_OUTPUT_PATH)
        self.undetected_events = pd.read_csv(UNDETECTED_EVENTS_OUTPUT_PATH)
        _LOGGER.info(f"Loaded {len(self.cramer_rao_bounds)} detections...")
        self.cosmological_model = LamCDMScenario()
        self.h = self.cosmological_model.h.fiducial_value
        self.Omega_m = self.cosmological_model.Omega_m.fiducial_value
        self.Omega_DE = 1 - self.Omega_m
        self.w_0 = self.cosmological_model.w_0
        self.w_a = self.cosmological_model.w_a


    def plot_detection_fraction(self) -> None:
       # TODO: implement using the detection_probability function
        pass

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
                self.additional_galaxies_without_bh_mass[h] = h_data.pop(
                    ADDITIONAL_GALAXIES_WITHOUT_BH_MASS
                )

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
                and (np.max(posterior) != np.inf)
            )
        }
        self.posterior_data = {
            detection_index: posterior
            for detection_index, posterior in self.posterior_data.items()
            if (
                (len(posterior) == len(self.h_values))
                and (np.max(posterior) > 0)
                and (np.max(posterior) != np.inf)
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

        # plot number of possible hosts histogram with and without bh mass
        fig, ax = plt.subplots(figsize=(16, 9))
        galaxy_numbers_with_bh_mass = np.array(
            [
                len(galaxy_weights)
                for galaxy_weights in self.galaxy_weights[
                    str(self.h_values[0])
                ].values()
            ]
        )
        galaxy_numbers_without_bh_mass = (
            np.array(
                [
                    len(galaxy_weights)
                    for galaxy_weights in self.additional_galaxies_without_bh_mass[
                        str(self.h_values[0])
                    ].values()
                ]
            )
            + galaxy_numbers_with_bh_mass
        )
        print(len(galaxy_numbers_without_bh_mass))
        bins = np.logspace(
            np.log10(min(galaxy_numbers_without_bh_mass)),
            np.log10(max(galaxy_numbers_without_bh_mass)),
            30,
        )

        plt.hist(
            galaxy_numbers_without_bh_mass,
            bins=bins,
            histtype="step",
            color="b",
            label="without bh mass",
        )
        plt.hist(
            galaxy_numbers_with_bh_mass,
            bins=bins,
            histtype="step",
            color="r",
            label="with bh mass",
            linestyle="dotted",
        )
        plt.xlabel("number of possible hosts")
        plt.ylabel("number of detections")
        plt.xscale("log")
        plt.legend()
        plt.savefig("saved_figures/number_of_possible_hosts.png", dpi=300)
        plt.close()

        # define colormap for skylocalization coloring
        sky_localization_error_min = min(
            [detection.get_skylocalization_error() for detection in detections]
        )
        sky_localization_error_max = max(
            [detection.get_skylocalization_error() for detection in detections]
        )
        relativ_distance_error_min = min(
            [detection.d_L_uncertainty / detection.d_L for detection in detections]
        )
        relativ_distance_error_max = max(
            [
                detection.d_L_uncertainty / detection.d_L
                for detection in detections
                if detection.d_L_uncertainty != 0
            ]
        )

        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(
            vmin=relativ_distance_error_min, vmax=relativ_distance_error_max
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
            color = cmap(norm(detection.get_relative_distance_error()))

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
            color = cmap(norm(detection.get_relative_distance_error()))

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
            label="relative d_L error",
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
        NUMBER_OF_DETECTIONS = 75
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

                """ax[0, 0].plot(
                    h_values_fine,
                    gaussian(h_values_fine, *popt),
                    label=f"std: {np.round(popt[1], 3)}, mean: {np.round(popt[0], 3)}",
                    color=colors[count],
                    linestyle="--",
                    linewidth=0.5,
                )"""
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

                """ax[0, 1].plot(
                    h_values_fine,
                    gaussian(h_values_fine, *popt_with_bh_mass),
                    label=f"std: {np.round(popt_with_bh_mass[1], 3)}, mean: {np.round(popt_with_bh_mass[0], 3)}",
                    color=colors[count],
                    linestyle="--",
                    linewidth=0.5,
                )"""
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

            ax[0, 0].plot(
                temp_h_values,
                sub_posteriors,
                label="without BH mass",
                color=colors[count],
            )
            ax[0, 1].plot(
                temp_h_values_with_bh_mass,
                sub_posteriors_with_bh_mass,
                label="with BH mass",
                color=colors[count],
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
            elif np.max(posteriors * posterior) == 0.0:
                print("All zeros detected")
                posteriors = posteriors / np.max(posteriors)
            posteriors *= np.array(posterior)
        for index, posterior in posterior_data_with_bh_mass_sorted:
            if check_overflow(posteriors_with_bh_mass * np.array(posterior)):
                # print("Overflow detected")
                posteriors_with_bh_mass = posteriors_with_bh_mass / np.max(
                    posteriors_with_bh_mass
                )
            elif np.max(posteriors_with_bh_mass * posterior) == 0.0:
                print("All zeros detected")
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
                label=f"std: {np.round(popt[1], 4)} +/- {np.round(perr[1], 4)},\nmean: {np.round(popt[0], 4)} +/- {np.round(perr[0], 4)}",
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
                label=f"std: {np.round(popt_with_bh_mass[1], 4)} +/- {np.round(perr_with_bh_mass[1], 4)},\nmean: {np.round(popt_with_bh_mass[0], 4)} +/- {np.round(perr_with_bh_mass[0], 4)}",
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

        additional_galaxy_likelihoods_by_detection = {}
        for h, data in self.additional_galaxies_without_bh_mass.items():
            for detection_index, host_galaxy_weights in data.items():
                try:
                    additional_galaxy_likelihoods_by_detection[int(detection_index)][
                        h
                    ] = host_galaxy_weights
                except KeyError:
                    if len(host_galaxy_weights) == 0:
                        continue
                    additional_galaxy_likelihoods_by_detection[int(detection_index)] = (
                        {}
                    )
                    additional_galaxy_likelihoods_by_detection[int(detection_index)][
                        h
                    ] = host_galaxy_weights

        # remove weight_data with less samples than h_values
        weight_data = {
            detection_index: host_galaxy_weights
            for detection_index, host_galaxy_weights in weight_data.items()
            if len(host_galaxy_weights.keys()) == len(self.h_values_with_bh_mass)
        }

        additional_galaxy_likelihoods_by_detection = {
            detection_index: host_galaxy_weights
            for detection_index, host_galaxy_weights in additional_galaxy_likelihoods_by_detection.items()
            if len(host_galaxy_weights.keys()) == len(self.h_values)
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

        max_likelihood_additional_galaxies_by_detection = {
            detection_index: np.max(
                [
                    np.sum([likelihood for _, likelihood in value])
                    for value in host_galaxy_weights.values()
                ]
            )
            for detection_index, host_galaxy_weights in additional_galaxy_likelihoods_by_detection.items()
        }

        for detection_index, host_galaxy_weights_by_h_value in weight_data.items():
            _LOGGER.info(f"Visualizing galaxy weights for detection {detection_index}")
            detection = Detection(self.cramer_rao_bounds.iloc[int(detection_index)])
            true_galaxy = Detection(
                self.true_cramer_rao_bounds.iloc[int(detection_index)]
            )
            max_likelihood_without_bh_mass = (
                max_likelihood_without_bh_mass_by_detection[detection_index]
                + max_likelihood_additional_galaxies_by_detection[detection_index]
            )

            max_likelihood_with_bh_mass = max_likelihood_with_bh_mass_by_detection[
                detection_index
            ]

            additional_galaxy_likelihoods_by_h = (
                additional_galaxy_likelihoods_by_detection[detection_index]
            )

            additional_galaxies = [
                galaxy_catalog.get_host_galaxy_by_index(int(galaxy_index))
                for galaxy_index, _ in additional_galaxy_likelihoods_by_h.values()
                .__iter__()
                .__next__()
            ]

            # plot h values on x axis and weights on y axis and the sum of weights
            fig, axs = plt.subplots(2, 3, figsize=(16, 9))
            # figure title
            fig.suptitle(
                f"Galaxy likelihood visualization for detection {detection_index} (Skyloc error: {np.round(detection.get_skylocalization_error(), 4)})"
            )

            host_galaxies = [
                galaxy_catalog.get_host_galaxy_by_index(int(index))
                for index, _ in host_galaxy_weights_by_h_value.values()
                .__iter__()
                .__next__()
            ]
            host_galaxies_without_bh_mass = host_galaxies.copy()
            host_galaxies_without_bh_mass.extend(additional_galaxies)

            host_galaxies_phi = np.array([galaxy.phiS for galaxy in host_galaxies])
            host_galaxies_theta = np.array([galaxy.qS for galaxy in host_galaxies])
            host_galaxies_redshift = np.array([galaxy.z for galaxy in host_galaxies])
            host_galaxies_mean_redshift = np.mean(host_galaxies_redshift)
            host_galaxies_mass = np.array([galaxy.M for galaxy in host_galaxies])

            host_galaxies_phi_without_bh_mass = np.array(
                [galaxy.phiS for galaxy in host_galaxies_without_bh_mass]
            )
            host_galaxies_theta_without_bh_mass = np.array(
                [galaxy.qS for galaxy in host_galaxies_without_bh_mass]
            )
            host_galaxies_redshift_without_bh_mass = np.array(
                [galaxy.z for galaxy in host_galaxies_without_bh_mass]
            )
            host_galaxies_mass_without_bh_mass = np.array(
                [galaxy.M for galaxy in host_galaxies_without_bh_mass]
            )
            host_galaxies_mean_redshift_without_bh_mass = np.mean(
                host_galaxies_redshift_without_bh_mass
            )

            # redshift distribution of galaxies
            gaussians_with_bh_mass = [
                truncnorm((0 - galaxy.z) / galaxy.z_error, 10, galaxy.z, galaxy.z_error)
                for galaxy in host_galaxies
            ]

            gaussians_without_bh_mass = [
                truncnorm(
                    (0 - galaxy.z) / galaxy.z_error,
                    10,
                    galaxy.z,
                    galaxy.z_error,
                )
                for galaxy in additional_galaxies
            ]

            # detection accuracy

            detection_accuracy_gaussian = truncnorm(
                (0 - detection.d_L) / detection.d_L_uncertainty,
                10,
                detection.d_L,
                detection.d_L_uncertainty,
            )

            d_L_range = np.linspace(
                max(0.0, detection.d_L - 2 * detection.d_L_uncertainty),
                detection.d_L + 2 * detection.d_L_uncertainty,
                100,
            )

            detection_redshift_accuracy_gaussian = truncnorm(
                (0 - dist_to_redshift(detection.d_L))
                / dist_to_redshift(detection.d_L_uncertainty),
                10,
                dist_to_redshift(detection.d_L),
                dist_to_redshift(detection.d_L_uncertainty),
            )

            mean_gaussian_std_with_bh_mass = np.mean(
                [distribution.std() for distribution in gaussians_with_bh_mass]
            )

            stds = [distribution.std() for distribution in gaussians_without_bh_mass]
            stds.extend([distribution.std() for distribution in gaussians_with_bh_mass])
            mean_gaussian_std = np.mean(stds)

            redshift_range = np.linspace(
                max(
                    -0.002,
                    min(host_galaxies_redshift_without_bh_mass) - 2 * mean_gaussian_std,
                ),
                max(host_galaxies_redshift_without_bh_mass) + 2 * mean_gaussian_std,
                100,
            )

            redshift_distribution_with_bh_mass_by_gaussian = np.array(
                [normal.pdf(redshift_range) for normal in gaussians_with_bh_mass]
            )

            redshift_distribution_with_bh_mass = np.sum(
                redshift_distribution_with_bh_mass_by_gaussian, axis=0
            )

            redshift_distribution_by_gaussian = np.array(
                [normal.pdf(redshift_range) for normal in gaussians_without_bh_mass]
            )

            redshift_distribution = (
                np.sum(redshift_distribution_by_gaussian, axis=0)
                + redshift_distribution_with_bh_mass
            ) / len(host_galaxies_without_bh_mass)

            redshift_distribution_with_bh_mass = (
                redshift_distribution_with_bh_mass / len(host_galaxies)
            )

            # plotting independent of h
            cmap = cm.get_cmap("viridis")

            axs[0, 0].plot(
                redshift_range,
                redshift_distribution,
                c="black",
            )

            axs[0, 0].axvline(
                redshift_range[0] + 3 * mean_gaussian_std,
                color="grey",
                linestyle="--",
            )

            axs[0, 0].axvline(
                redshift_range[-1] - 3 * mean_gaussian_std,
                color="grey",
                linestyle="--",
            )

            normal_dist_samples = min(50, len(gaussians_without_bh_mass))
            """gaussian_indices = np.random.choice(
                len(gaussians_without_bh_mass),
                normal_dist_samples,
                replace=False,
            )"""
            gaussian_indices = np.arange(len(gaussians_without_bh_mass))
            for index in gaussian_indices:
                normal_dist = redshift_distribution_by_gaussian[index]
                axs[0, 0].plot(
                    redshift_range,
                    normal_dist / max(normal_dist) * max(redshift_distribution) / 2,
                    c="grey",
                    linestyle="--",
                    linewidth=0.1,
                )

            detection_gaussian_values = detection_redshift_accuracy_gaussian.pdf(
                redshift_range
            )

            detection_gaussian_values = (
                detection_gaussian_values
                / max(detection_gaussian_values)
                * max(redshift_distribution)
                / 2
            )
            axs[0, 0].plot(
                redshift_range,
                detection_gaussian_values,
                c="orange",
                linestyle="dotted",
                linewidth=0.5,
            )

            axs[0, 0].axvline(
                dist_to_redshift(true_galaxy.d_L), color="g", linestyle="--"
            )
            axs[0, 0].set_xlabel("redshift")
            axs[0, 0].set_ylabel("density")
            axs[0, 0].axvline(
                host_galaxies_mean_redshift_without_bh_mass,
                color="r",
                linestyle="--",
                label="mean redshift",
            )
            axs[0, 0].axvline(
                dist_to_redshift(detection.d_L),
                color="black",
                linestyle="-.",
                label="detection redshift",
            )
            axs[0, 0].set_title("redshift distribution")

            # plot 0, 1
            axs[0, 1].plot(
                redshift_range,
                redshift_distribution_with_bh_mass,
                c="black",
            )

            axs[0, 1].axvline(
                redshift_range[0] + 3 * mean_gaussian_std_with_bh_mass,
                color="grey",
                linestyle="--",
            )
            axs[0, 1].axvline(
                redshift_range[-1] - 3 * mean_gaussian_std_with_bh_mass,
                color="grey",
                linestyle="--",
            )

            normal_dist_samples = min(50, len(gaussians_with_bh_mass))
            """gaussian_indices = np.random.choice(
                len(gaussians_with_bh_mass),
                normal_dist_samples,
                replace=False,
            )"""
            gaussian_indices = np.arange(len(gaussians_with_bh_mass))
            for index in gaussian_indices:
                normal_dist = redshift_distribution_with_bh_mass_by_gaussian[index]
                axs[0, 1].plot(
                    redshift_range,
                    normal_dist
                    / max(normal_dist)
                    * max(redshift_distribution_with_bh_mass)
                    / 2,
                    c="grey",
                    linestyle="--",
                    linewidth=0.1,
                )

            detection_gaussian_values = detection_redshift_accuracy_gaussian.pdf(
                redshift_range
            )

            detection_gaussian_values = (
                detection_gaussian_values
                / max(detection_gaussian_values)
                * max(redshift_distribution_with_bh_mass)
                / 2
            )
            axs[0, 1].plot(
                redshift_range,
                detection_gaussian_values,
                c="orange",
                linestyle="dotted",
                linewidth=0.5,
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
            axs[0, 1].set_title("redshift_distribution with BH mass")

            # plot 1,0
            axs[1, 0].axvline(H, color="g", linestyle="--")
            axs[1, 0].set_title(f"likelihood")

            axs[1, 1].axvline(H, color="g", linestyle="--")
            axs[1, 1].set_title(f"likelihood")

            axs[1, 2].set_title("bias correction factor")
            axs[1, 2].set_xlabel("h value")

            for h_value, host_galaxy_weights in host_galaxy_weights_by_h_value.items():
                additional_galaxy_likelihoods = np.array(
                    [
                        likelihood
                        for _, likelihood in additional_galaxy_likelihoods_by_h[h_value]
                    ]
                )
                h_value = float(h_value)

                likelihoods_without_bh_mass = np.concatenate(
                    [
                        np.array([weights[0] for _, weights in host_galaxy_weights]),
                        additional_galaxy_likelihoods,
                    ],
                    axis=None,
                )

                likelihoods_with_bh_mass = np.array(
                    [weights[1] for _, weights in host_galaxy_weights]
                )

                # compute bias correction factor

                detection_accuracy_gaussian_values = detection_accuracy_gaussian.pdf(
                    d_L_range
                )

                infered_z_range = np.array(
                    [dist_to_redshift(d_L, h=h_value) for d_L in d_L_range]
                )
                distance_relation_derivative_at_detection_redshift = np.array(
                    [dist_derivative(z, h=h_value) for z in infered_z_range]
                )

                p_gal_range_with_bh_mass = np.sum(
                    [normal.pdf(infered_z_range) for normal in gaussians_with_bh_mass],
                    axis=0,
                )

                p_gal_range = (
                    np.sum(
                        [
                            normal.pdf(infered_z_range)
                            for normal in gaussians_without_bh_mass
                        ],
                        axis=0,
                    )
                    + p_gal_range_with_bh_mass
                ) / len(host_galaxies_without_bh_mass)

                p_gal_range_with_bh_mass = p_gal_range_with_bh_mass / len(host_galaxies)

                alpha_without_bh_mass = np.trapz(
                    p_gal_range
                    * detection_accuracy_gaussian_values
                    / distance_relation_derivative_at_detection_redshift,
                    d_L_range,
                )

                alpha_with_bh_mass = np.trapz(
                    p_gal_range_with_bh_mass
                    * detection_accuracy_gaussian_values
                    / distance_relation_derivative_at_detection_redshift,
                    d_L_range,
                )

                detection_redshift = dist_to_redshift(detection.d_L, h=h_value)

                distance_relation_derivative_at_detection_redshift = dist_derivative(
                    detection_redshift, h=h_value
                )

                p_gal_at_detection_redshift_with_bh_mass = np.sum(
                    [
                        normal.pdf(detection_redshift)
                        for normal in gaussians_with_bh_mass
                    ]
                )

                p_gal_at_detection_redshift = (
                    np.sum(
                        [
                            normal.pdf(detection_redshift)
                            for normal in gaussians_without_bh_mass
                        ]
                    )
                    + p_gal_at_detection_redshift_with_bh_mass
                ) / len(host_galaxies_without_bh_mass)

                p_gal_at_detection_redshift_with_bh_mass = (
                    p_gal_at_detection_redshift_with_bh_mass / len(host_galaxies)
                )

                detection_likelihood_without_bh_mass = (
                    (np.sum(likelihoods_without_bh_mass))
                    / max_likelihood_without_bh_mass
                    / alpha_without_bh_mass
                )
                detection_likelihood_with_bh_mass = (
                    np.sum(likelihoods_with_bh_mass)
                    / max_likelihood_with_bh_mass
                    / alpha_with_bh_mass
                )

                # plot likelihood contribution by redshift bins

                # plt galaxy number per bin in plot 0,0

                h_normalized = (h_value - min(self.h_values)) / (
                    max(self.h_values) - min(self.h_values)
                )

                axs[0, 0].scatter(
                    detection_redshift,
                    p_gal_at_detection_redshift,
                    c=cmap(h_normalized),
                )

                # plot 0, 1

                axs[0, 1].scatter(
                    detection_redshift,
                    p_gal_at_detection_redshift_with_bh_mass,
                    c=cmap(h_normalized),
                )

                # plot likelihoods
                # try the weighting by the number of galaxies in the bin
                """detection_likelihood_without_bh_mass = np.sum(
                    np.array(likelihood_bin_contribution) / np.array(galaxies_per_bin)
                )
                detection_likelihood_with_bh_mass = np.sum(
                    np.array(likelihood_with_bh_mass_bin_contribution)
                    / np.array(galaxies_per_bin)
                )"""
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

                # plt 1,1 plot bias correction factor
                axs[1, 2].scatter(
                    [h_value],
                    alpha_without_bh_mass,
                    c="b",
                    label="without BH mass",
                )
                axs[1, 2].scatter(
                    [h_value],
                    alpha_with_bh_mass,
                    c="r",
                    label="with BH mass",
                )

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
                    """axs[1, 1].scatter(
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
                    axs[1, 1].set_title("redshift mass distribution")"""

                    axs[0, 2].scatter(
                        host_galaxies_phi_without_bh_mass,
                        host_galaxies_theta_without_bh_mass,
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
                            f"Galaxy skylocalization weight for h = {np.round(h_value,2)}"
                        )
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=0.6, vmax=0.86)
            )
            plt.colorbar(sm, ax=axs[0, 1], ticks=[0.6, 0.73, 0.86])
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

    def evaluate(self, galaxy_catalog: GalaxyCatalogueHandler, cosmological_model: Model1CrossCheck, h_value: float) -> None:
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
            f"After filtering {len(self.cramer_rao_bounds)} detections with relative luminosity distance error < {FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD}"
        )
        # parameter limitations
        REDSHIFT_LOWER_LIMIT = 0.0
        REDSHIFT_UPPER_LIMIT = cosmological_model.max_redshift
        BH_MASS_LOWER_LIMIT = cosmological_model.parameter_space.M.lower_limit
        BH_MASS_UPPER_LIMIT = cosmological_model.parameter_space.M.upper_limit

        # find parameter space limits for detections
        PARAMETERSPACE_MARGIN = 0.2
        luminosity_distance_lower_limit = 0.0
        luminosity_distance_upper_limit = max(
            self.cramer_rao_bounds["dist"]
        ) * (1 + PARAMETERSPACE_MARGIN)
        mass_lower_limit = min(self.cramer_rao_bounds["M"]) * (1 - PARAMETERSPACE_MARGIN)
        mass_upper_limit = max(self.cramer_rao_bounds["M"]) * (1 + PARAMETERSPACE_MARGIN)
        
        _LOGGER.debug("Creating detection probability functions...")
        detection_probability = DetectionProbability(
            luminosity_distance_lower_limit=luminosity_distance_lower_limit,
            luminosity_distance_upper_limit=luminosity_distance_upper_limit,
            mass_lower_limit=mass_lower_limit,
            mass_upper_limit=mass_upper_limit,
            detected_events=self.cramer_rao_bounds,
            undetected_events=self.undetected_events,
            bandwidth=None
        )
        _LOGGER.debug("Detection probability functions created.")

        _LOGGER.debug("Creating detection likelihood gaussian functions...")
        detection_likelihood_multivariate_gaussian_by_detection_index: Dict[int, Tuple[_multivariate.multivariate_normal_frozen, _multivariate.multivariate_normal_frozen]] = {}
        for index, detection in self.cramer_rao_bounds.iterrows():
            detection = Detection(detection)
            covariance_without_bh_mass = [
                [
                    detection.phi_error**2,
                    detection.theta_phi_covariance,
                    detection.d_L_phi_covariance / detection.d_L,
                ],
                [
                    detection.theta_phi_covariance,
                    detection.theta_error**2,
                    detection.d_L_theta_covariance / detection.d_L,
                ],
                [
                    detection.d_L_phi_covariance / detection.d_L,
                    detection.d_L_theta_covariance / detection.d_L,
                    detection.d_L_uncertainty**2 / detection.d_L**2,
                ],
            ]
            covariance_with_bh_mass = [
                [
                    detection.phi_error**2,
                    detection.theta_phi_covariance,
                    detection.d_L_phi_covariance / detection.d_L,
                    detection.M_phi_covariance / detection.M,
                ],
                [
                    detection.theta_phi_covariance,
                    detection.theta_error**2,
                    detection.d_L_theta_covariance / detection.d_L,
                    detection.M_theta_covariance / detection.M,
                ],
                [
                    detection.d_L_phi_covariance / detection.d_L,
                    detection.d_L_theta_covariance / detection.d_L,
                    detection.d_L_uncertainty**2 / detection.d_L**2,
                    detection.d_L_M_covariance / detection.d_L / detection.M,
                ],
                [
                    detection.M_phi_covariance / detection.M,
                    detection.M_theta_covariance / detection.M,
                    detection.d_L_M_covariance / detection.d_L / detection.M,
                    detection.M_uncertainty**2 / detection.M**2,
                ],
            ]

            gaussian_without_bh_mass = multivariate_normal(
                mean=[detection.phi, detection.theta, 1],
                cov=covariance_without_bh_mass,
                allow_singular=True, # TODO: this should not be needed in the end
            )
            gaussian_with_bh_mass = multivariate_normal(
                mean=[detection.phi, detection.theta, 1, 1],
                cov=covariance_with_bh_mass,
                allow_singular=True, # TODO: this should not be needed in the end
            )
            detection_likelihood_multivariate_gaussian_by_detection_index[
                index
            ] = (
                gaussian_without_bh_mass,
                gaussian_with_bh_mass
            )
        _LOGGER.debug("Detection likelihood gaussians created.")

        self.h = h_value
        
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count()

        _LOGGER.debug(
            f"Found {available_cpus} / {os.cpu_count()} (available / system) cpus."
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
            available_cpus - 2,
            initializer=child_process_init,
            initargs=(
                REDSHIFT_LOWER_LIMIT,
                REDSHIFT_UPPER_LIMIT,
                BH_MASS_LOWER_LIMIT,
                BH_MASS_UPPER_LIMIT,
                detection_probability,
                detection_likelihood_multivariate_gaussian_by_detection_index
            ),
        ) as pool:
            self.p_D(
                galaxy_catalog=galaxy_catalog,
                redshift_upper_limit=REDSHIFT_UPPER_LIMIT,
                pool=pool,
            )
        _LOGGER.info(f"posteriors comupted for h = {self.h}")

        if not os.path.isdir("simulations/posteriors"):
            os.makedirs("simulations/posteriors")
        if not os.path.isdir("simulations/posteriors_with_bh_mass"):
            os.makedirs("simulations/posteriors_with_bh_mass")
        

        with open(
            f"simulations/posteriors/h_{str(np.round(self.h,3)).replace('.', '_')}.json",
            "w",
        ) as file:
            data = {
                str(key): value for key, value in self.posterior_data.items()
            }
            json.dump(data | {"h": self.h}, file)

        with open(
            f"simulations/posteriors_with_bh_mass/h_{str(np.round(self.h,3)).replace('.', '_')}.json",
            "w",
        ) as file:
            # update existing data

            data = {
                str(key): value
                for key, value in self.posterior_data_with_bh_mass.items()
            }
            json.dump(data | {"h": self.h}, file)

    def p_D(
        self,
        galaxy_catalog: GalaxyCatalogueHandler,
        redshift_upper_limit: float,
        pool: mp.Pool,
    ) -> None:
        count = 0
        self.posterior_data_with_bh_mass[GALAXY_LIKELIHOODS] = {}
        self.posterior_data_with_bh_mass[ADDITIONAL_GALAXIES_WITHOUT_BH_MASS] = {}
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
                sigma_multiplier=2.0,
            )

            z_max = min(z_max, redshift_upper_limit)

            possible_hosts = galaxy_catalog.get_possible_hosts_from_ball_tree(
                phi=self.detection.phi,
                theta=self.detection.theta,
                phi_sigma=self.detection.phi_error,
                theta_sigma=self.detection.theta_error,
                z_min=z_min,
                z_max=z_max,
                M_z=self.detection.M,
                M_z_sigma=self.detection.M_uncertainty,
                sigma_multiplier=1.5,
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
        
        hosts_with_bh_mass_set = set(possible_host_galaxies_with_bh_mass)

        possible_host_galaxies_reduced = [
            host
            for host in possible_host_galaxies
            if host not in hosts_with_bh_mass_set
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
                    detection_index,
                    self.h,
                    True,
                )
                for possible_host in possible_host_galaxies_with_bh_mass
            ],
            chunksize=chunksize_with_bh_mass,
        )

        results_without_blackhole_mass = pool.starmap(
            single_host_likelihood,
            [
                (
                    possible_host,
                    self.detection,
                    detection_index,
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

        additional_likelihoods = list(
            zip(
                [galaxy.catalog_index for galaxy in possible_host_galaxies_reduced],
                results_without_blackhole_mass,
            )
        )

        self.posterior_data_with_bh_mass[ADDITIONAL_GALAXIES_WITHOUT_BH_MASS][
            detection_index
        ] = additional_likelihoods

        if len(results_without_blackhole_mass) == 0:
            print("no results found")
            return 0.0, 0.0

        selection_effect_correction_without_bh_mass = np.sum([result[1] for result in results_without_blackhole_mass])
        numerator_without_bh_mass = [result[0] for result in results_without_blackhole_mass]
        numerator_without_bh_mass.extend([result[0] for result in results_with_bh_mass])

        likelihood_without_bh_mass = np.sum(numerator_without_bh_mass)

        selection_effect_correction_without_bh_mass += np.sum(
            [result[1] for result in results_with_bh_mass]
        )

        if len(results_with_bh_mass) == 0:
            return likelihood_without_bh_mass / selection_effect_correction_without_bh_mass, 0.0

        likelihood_with_bh_mass = np.sum([result[2] for result in results_with_bh_mass])

        selection_effect_correction_with_bh_mass = np.sum(
            [result[3] for result in results_with_bh_mass]
        )
      
        return (
            likelihood_without_bh_mass / selection_effect_correction_without_bh_mass,
            likelihood_with_bh_mass / selection_effect_correction_with_bh_mass,
        )


def use_detection(detection: Detection) -> bool:
    sky_localization_uncertainty = _sky_localization_uncertainty(
        phi_error=detection.phi_error,
        theta=detection.theta,
        theta_error=detection.theta_error,
        cov_theta_phi=detection.theta_phi_covariance,
    )
    distance_relative_error = detection.d_L_uncertainty / detection.d_L

    if distance_relative_error < FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD:
        return True
    _LOGGER.debug(
        f"Detection skipped: distance_relative_error {distance_relative_error} > {FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD}, sky_localization_uncertainty {sky_localization_uncertainty}"
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


def single_host_likelihood_grid(
    possible_host: HostGalaxy,
    detection: Detection,
    detection_index: int,
    h: float,
    evaluate_with_bh_mass: bool,
) -> list[float]:
    global redshift_upper_integration_limit 
    global redshift_lower_integration_limit
    global bh_mass_upper_integration_limit
    global bh_mass_lower_integration_limit
    global detection_probability
    global detection_likelihood_gaussians_by_detection_index

    # find sharpest peak 
    print(possible_host.z, possible_host.z_error)
    print(detection.d_L, detection.d_L_uncertainty)

def single_host_likelihood(
    possible_host: HostGalaxy,
    detection: Detection,
    detection_index: int,
    h: float,
    evaluate_with_bh_mass: bool,
) -> list[float]:
    global redshift_upper_integration_limit
    global redshift_lower_integration_limit
    global bh_mass_upper_integration_limit
    global bh_mass_lower_integration_limit
    global detection_probability
    global detection_likelihood_gaussians_by_detection_index

    ABS_ERROR = 1e-10
    FIXED_QUAD_N = 50

    integration_limit_sigma_multiplier = 4.0

    numerator_integration_upper_redshift_limit = dist_to_redshift(
        detection.d_L + integration_limit_sigma_multiplier * detection.d_L_uncertainty, h=h
    )
    numerator_integration_lower_redshift_limit = dist_to_redshift(
        detection.d_L - integration_limit_sigma_multiplier * detection.d_L_uncertainty, h=h
    )
    denominator_integration_upper_redshift_limit = possible_host.z + integration_limit_sigma_multiplier * possible_host.z_error
    denominator_integration_lower_redshift_limit = possible_host.z - integration_limit_sigma_multiplier * possible_host.z_error

    # construct normal distribution for redshift and mass for host galaxy
    galaxy_redshift_normal_distribution = norm(
        loc=possible_host.z, scale=possible_host.z_error
    )
    # TODO: KEEP IN MIND SKYLOCALIZATION WEIGHT IS IN THE GW LIKELIHOOD ATM. possible source of error
    def numerator_integrant_without_bh_mass(z: np.ndarray[float]) -> np.ndarray[float]:
        d_L = dist(z, h=h)
        luminosity_distance_fraction = d_L / detection.d_L
        phi = np.full_like(z, possible_host.phiS)
        theta = np.full_like(z, possible_host.qS)

        return (
            detection_probability.detection_probability_without_bh_mass_interpolated(d_L, phi, theta)
            * detection_likelihood_gaussians_by_detection_index[
                detection_index
            ][0].pdf(np.vstack([phi, theta, luminosity_distance_fraction]).T)
            * galaxy_redshift_normal_distribution.pdf(z)
            / detection.d_L
        )

    def denominator_integrant_without_bh_mass(z: np.ndarray[float]) -> np.ndarray[float]:
        d_L = dist(z, h=h)
        phi = np.full_like(z, possible_host.phiS)
        theta = np.full_like(z, possible_host.qS)
        return (
            detection_probability.detection_probability_without_bh_mass_interpolated(d_L, phi, theta)
            * galaxy_redshift_normal_distribution.pdf(z)
        )
    
    single_host_likelihood_numerator_without_bh_mass, single_host_likelihood_numerator_without_bh_mass_error = fixed_quad(
        numerator_integrant_without_bh_mass, 
        numerator_integration_lower_redshift_limit, 
        numerator_integration_upper_redshift_limit, 
        n=FIXED_QUAD_N,
    )
    single_host_likelihood_denominator_without_bh_mass, single_host_likelihood_denominator_without_bh_mass_error = fixed_quad(
        denominator_integrant_without_bh_mass, 
        denominator_integration_lower_redshift_limit, 
        denominator_integration_upper_redshift_limit, 
        n=FIXED_QUAD_N,
    )

    if evaluate_with_bh_mass:
        galaxy_mass_normal_distribution = norm(
        loc=possible_host.M, scale=possible_host.M_error
        )
        def numerator_integrant_with_bh_mass(z: np.ndarray[float]) -> np.ndarray[float]:
            d_L = dist(z, h=h)
            luminosity_distance_fraction = d_L / detection.d_L
            M_z = np.full_like(z, detection.M)
            phi = np.full_like(z, possible_host.phiS)
            theta = np.full_like(z, possible_host.qS)
            return (
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L,
                    M_z, 
                    phi, 
                    theta
                )
                * detection_likelihood_gaussians_by_detection_index[
                    detection_index
                ][0].pdf(
                    np.vstack([
                        phi, 
                        theta, 
                        luminosity_distance_fraction
                    ]).T
                )
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(detection.M / (1+z))
                / (detection.d_L * (1 + z)) # TODO: check if this is correct
            )

        
        single_host_likelihood_numerator_with_bh_mass = fixed_quad(
            numerator_integrant_with_bh_mass, 
            numerator_integration_lower_redshift_limit, 
            numerator_integration_upper_redshift_limit, 
            n=FIXED_QUAD_N,
        )[0]

        def denominator_integrant_with_bh_mass_vectorized(M: np.ndarray[float], z: np.ndarray[float]) -> np.ndarray[float]:
            d_L = dist_vectorized(z, h=h)
            M_z = M * (1 + z)
            phi = np.full_like(M, possible_host.phiS)
            theta = np.full_like(M, possible_host.qS)
            return (
                detection_probability.detection_probability_with_bh_mass_interpolated(d_L, M_z, phi, theta)
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )
        
        N_SAMPLES = 10_000
        z_samples = galaxy_redshift_normal_distribution.rvs(size=N_SAMPLES)
        M_samples = galaxy_mass_normal_distribution.rvs(size=N_SAMPLES)

        numerator_integrant_from_samples = denominator_integrant_with_bh_mass_vectorized(M_samples, z_samples)

        sampling_pdf = galaxy_redshift_normal_distribution.pdf(z_samples) * galaxy_mass_normal_distribution.pdf(M_samples) 
        weights = numerator_integrant_from_samples / sampling_pdf

        single_host_likelihood_denominator_with_bh_mass = np.mean(weights)

        return [single_host_likelihood_numerator_without_bh_mass, single_host_likelihood_denominator_without_bh_mass, single_host_likelihood_numerator_with_bh_mass, single_host_likelihood_denominator_with_bh_mass]
    return [single_host_likelihood_numerator_without_bh_mass, single_host_likelihood_denominator_without_bh_mass]
            
    
def single_host_likelihood_integration_testing(
    possible_host: HostGalaxy,
    detection: Detection,
    detection_index: int,
    h: float,
    evaluate_with_bh_mass: bool,
) -> list[float]:
    global redshift_upper_integration_limit
    global redshift_lower_integration_limit
    global bh_mass_upper_integration_limit
    global bh_mass_lower_integration_limit
    global detection_probability
    global detection_likelihood_gaussians_by_detection_index

    ABS_ERROR = 1e-20

    # construct normal distribution for redshift and mass for host galaxy
    galaxy_redshift_normal_distribution = norm(
        loc=possible_host.z, scale=possible_host.z_error
    )
    # TODO: KEEP IN MIND SKYLOCALIZATION WEIGHT IS IN THE GW LIKELIHOOD ATM. possible source of error
    def numerator_integrant_without_bh_mass(z: float) -> float:
        d_L = dist(z, h=h)
        luminosity_distance_fraction = d_L / detection.d_L
        return (
            detection_probability.evaluate_without_bh_mass(d_L, possible_host.phiS, possible_host.qS)
            * detection_likelihood_gaussians_by_detection_index[
                detection_index
            ][0].pdf([possible_host.phiS, possible_host.qS, luminosity_distance_fraction])
            * galaxy_redshift_normal_distribution.pdf(z)
        )

    def denominator_integrant_without_bh_mass(z: float) -> float:
        d_L = dist(z, h=h)
        return (
            detection_probability.evaluate_without_bh_mass(d_L, possible_host.phiS, possible_host.qS)
            * galaxy_redshift_normal_distribution.pdf(z)
        )
    
    single_host_likelihood_numerator_without_bh_mass, single_host_likelihood_numerator_without_bh_mass_error = quad(numerator_integrant_without_bh_mass, redshift_lower_integration_limit, redshift_upper_integration_limit, epsabs=ABS_ERROR)
    single_host_likelihood_denominator_without_bh_mass, single_host_likelihood_denominator_without_bh_mass_error = quad(denominator_integrant_without_bh_mass, redshift_lower_integration_limit, redshift_upper_integration_limit, epsabs=ABS_ERROR)

    print(f"Numerator without bh m:{single_host_likelihood_numerator_without_bh_mass}, error estimation: {single_host_likelihood_numerator_without_bh_mass_error}", flush=True)
    print(f"Denominator without bh m:{single_host_likelihood_denominator_without_bh_mass}, error estimation {single_host_likelihood_denominator_without_bh_mass_error}", flush=True
    )

    if evaluate_with_bh_mass:
        galaxy_mass_normal_distribution = norm(
        loc=possible_host.M, scale=possible_host.M_error
        )
        """
        # double integral version
        def numerator_integrant_with_bh_mass(M: float, z: float) -> float:
            d_L = dist(z, h=h)
            M_z = M * (1 + z)
            luminosity_distance_fraction = d_L / detection.d_L
            redshifted_mass_fraction = M_z / detection.M
            return (
                detection_probability.evaluate_with_bh_mass(d_L, M_z, possible_host.phiS, possible_host.qS)
                * detection_likelihood_gaussians_by_detection_index[
                    detection_index
                ][1].pdf(
                    [possible_host.phiS, possible_host.qS, luminosity_distance_fraction, redshifted_mass_fraction]
                )
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )
        
        def denominator_integrant_with_bh_mass(M: float, z: float) -> float:
            d_L = dist(z, h=h)
            M_z = M * (1 + z)
            return (
                detection_probability.evaluate_with_bh_mass(d_L, M_z, possible_host.phiS, possible_host.qS)
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )
        start = time.time()
        single_host_likelihood_numerator_with_bh_mass, single_host_likelihood_numerator_without_bh_mass_error = dblquad(
            numerator_integrant_with_bh_mass,
            redshift_lower_integration_limit,
            redshift_upper_integration_limit,
            lambda z: bh_mass_lower_integration_limit,
            lambda z: bh_mass_upper_integration_limit,
            epsabs=ABS_ERROR
        )
        single_host_likelihood_denominator_with_bh_mass, single_host_likelihood_denominator_with_bh_mass_error = dblquad(
            denominator_integrant_with_bh_mass,
            redshift_lower_integration_limit,
            redshift_upper_integration_limit,
            lambda m: bh_mass_lower_integration_limit,
            lambda m: bh_mass_upper_integration_limit,
            epsabs=ABS_ERROR
        )
        end = time.time()
        print(f"Time taken for double integral: {end - start}", flush=True)
        
        print(f"Numerator with bh m:{single_host_likelihood_numerator_with_bh_mass}, error estimation: {single_host_likelihood_numerator_without_bh_mass_error}", flush=True)
        print(f"Denominator with bh m:{single_host_likelihood_denominator_with_bh_mass}, error estimation {single_host_likelihood_denominator_with_bh_mass_error}", flush=True)
        """

        # treat M_z gaussian as delta function
        def numerator_integrant_with_bh_mass(z: float) -> float:
            d_L = dist(z, h=h)
            M_z = detection.M
            M = M_z / (1 + z)
            luminosity_distance_fraction = d_L / detection.d_L

            return (
                detection_probability.evaluate_with_bh_mass(d_L, M_z, possible_host.phiS, possible_host.qS)
                * detection_likelihood_gaussians_by_detection_index[
                    detection_index
                ][0].pdf(
                    [possible_host.phiS, possible_host.qS, luminosity_distance_fraction]
                )
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
                / (1 + z) # delta function derivative
            )
        
        def denominator_integrant_with_bh_mass(M: float, z: float) -> float:
            d_L = dist(z, h=h)
            M_z = M * (1 + z)
            return (
                detection_probability.evaluate_with_bh_mass(d_L, M_z, possible_host.phiS, possible_host.qS)
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )
        start = time.time()
        single_host_likelihood_numerator_with_bh_mass, single_host_likelihood_numerator_with_bh_mass_error = quad(numerator_integrant_with_bh_mass, redshift_lower_integration_limit, redshift_upper_integration_limit, epsabs=ABS_ERROR)

        single_host_likelihood_denominator_with_bh_mass, single_host_likelihood_denominator_with_bh_mass_error = dblquad(
            denominator_integrant_with_bh_mass,
            galaxy_redshift_normal_distribution.mean() - 5 * galaxy_redshift_normal_distribution.std(),
            galaxy_redshift_normal_distribution.mean() + 5 * galaxy_redshift_normal_distribution.std(),
            lambda m: galaxy_mass_normal_distribution.mean() - 5 * galaxy_mass_normal_distribution.std(),
            lambda m: galaxy_mass_normal_distribution.mean() + 5 * galaxy_mass_normal_distribution.std(),
            epsabs=ABS_ERROR
        )
        end = time.time()
        print(f"Time taken for delta function approximation: {end - start}s", flush=True)

        print(f"Numerator with bh m:{single_host_likelihood_numerator_with_bh_mass}, error estimation: {single_host_likelihood_numerator_with_bh_mass_error}", flush=True)
        print(f"Denominator with bh m:{single_host_likelihood_denominator_with_bh_mass}, error estimation {single_host_likelihood_denominator_with_bh_mass_error}", flush=True)

        # monte carlo integration denominator 2D
        start = time.time()
        def denominator_integrant_with_bh_mass_vectorized(M: np.ndarray[float], z: np.ndarray[float]) -> np.ndarray[float]:
            d_L = dist_vectorized(z, h=h)
            M_z = M * (1 + z)
            phi = np.ones_like(M) * possible_host.phiS
            theta = np.ones_like(M) * possible_host.qS
            return (
                detection_probability.evaluate_with_bh_mass_vectorized(d_L, M_z, phi, theta)
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )
        
        N_SAMPLES = 100_00
        z_samples = galaxy_redshift_normal_distribution.rvs(size=N_SAMPLES)
        M_samples = galaxy_mass_normal_distribution.rvs(size=N_SAMPLES)

        numerator_integrant_from_samples = denominator_integrant_with_bh_mass_vectorized(M_samples, z_samples)

        sampling_pdf = galaxy_redshift_normal_distribution.pdf(z_samples) * galaxy_mass_normal_distribution.pdf(M_samples) 
        weights = numerator_integrant_from_samples / sampling_pdf

        integral = np.mean(weights)
        integral_error = np.std(weights) / np.sqrt(N_SAMPLES)
        end = time.time()
        print(f"Time taken for monte carlo integration: {end - start}s", flush=True)
        print(f"Monte Carlo denominator integral with bh mass: {integral}, error estimation: {integral_error}", flush=True)
        print(f"Integration difference: {abs(single_host_likelihood_denominator_with_bh_mass - integral)}", flush=True)




        return [single_host_likelihood_numerator_without_bh_mass, single_host_likelihood_denominator_without_bh_mass, single_host_likelihood_numerator_with_bh_mass, single_host_likelihood_denominator_with_bh_mass]
    return [single_host_likelihood_numerator_without_bh_mass, single_host_likelihood_denominator_without_bh_mass]


def child_process_init(
    redshift_lower_limit: float,
    redshift_upper_limit: float,
    bh_mass_lower_limit: float,
    bh_mass_upper_limit: float,
    current_detection_probability: DetectionProbability,
    current_detection_likelihood_gaussians_by_detection_index: Dict[int, Tuple[_multivariate.multivariate_normal_frozen, _multivariate.multivariate_normal_frozen]],
) -> None:
    global redshift_upper_integration_limit
    global redshift_lower_integration_limit
    global bh_mass_upper_integration_limit
    global bh_mass_lower_integration_limit
    global detection_probability
    global detection_likelihood_gaussians_by_detection_index

    redshift_upper_integration_limit = redshift_upper_limit
    redshift_lower_integration_limit = redshift_lower_limit
    bh_mass_upper_integration_limit = bh_mass_upper_limit
    bh_mass_lower_integration_limit = bh_mass_lower_limit
    detection_probability = current_detection_probability
    detection_likelihood_gaussians_by_detection_index = (
        current_detection_likelihood_gaussians_by_detection_index
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
