"""EMRI event-rate cosmological model and H₀ evaluation orchestration.

:class:`Model1CrossCheck` samples EMRI events from a cosmological rate model.
:class:`BayesianStatistics` loads saved Cramér-Rao bounds and orchestrates the
full Hubble-constant posterior evaluation via :class:`~master_thesis_code.bayesian_inference.bayesian_inference.BayesianInference`.
"""

import json
import logging
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any

import emcee
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import dblquad, fixed_quad, quad
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import _multivariate, gaussian_kde, multivariate_normal, norm, truncnorm

from master_thesis_code.constants import (
    CRAMER_RAO_BOUNDS_OUTPUT_PATH,
    PREPARED_CRAMER_RAO_BOUNDS_PATH,
    UNDETECTED_EVENTS_OUTPUT_PATH,
    H,
)
from master_thesis_code.datamodels.detection import (
    Detection as Detection,
)
from master_thesis_code.datamodels.detection import (
    _sky_localization_uncertainty,
)
from master_thesis_code.datamodels.parameter_space import (
    Parameter,
    ParameterSpace,
    uniform,
)
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
    ParameterSample,
)
from master_thesis_code.M1_model_extracted_data.detection_fraction import (
    DetectionFraction,
)
from master_thesis_code.physical_relations import (
    dist,
    dist_to_redshift,
    dist_vectorized,
    get_redshift_outer_bounds,
)

_LOGGER = logging.getLogger()

DEFAULT_GALAXY_Z_ERROR = 0.0015
GALAXY_LIKELIHOODS = "galaxy_likelihoods"
ADDITIONAL_GALAXIES_WITHOUT_BH_MASS = "additional_galaxies_without_bh_mass"

FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD = 0.05

# Module-level globals used by child_process_init for multiprocessing worker state
redshift_upper_integration_limit: float = 0.0
redshift_lower_integration_limit: float = 0.0
bh_mass_upper_integration_limit: float = 0.0
bh_mass_lower_integration_limit: float = 0.0
detection_probability: Any = None
detection_likelihood_gaussians_by_detection_index: Any = None

# detection fraction LISA M1 model


@dataclass
class CosmologicalParameter(Parameter):
    fiducial_value: float = 1.0


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


def polynomial(
    x: float | npt.NDArray[np.float64],
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    g: float,
    h: float,
    i: float,
) -> float:
    if isinstance(x, int | float):
        if x > 3:
            x = 3.0
    else:
        x = np.array([value if value <= 3 else 3.0 for value in x])  # end of fit range

    result = (
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
    return float(result) if isinstance(result, int | float) else result  # type: ignore[return-value]


def MBH_spin_distribution(lower_limit: float, upper_limit: float) -> float:
    """https://iopscience.iop.org/article/10.1088/0004-637X/762/2/68/pdf"""
    return float(a_distribution.rvs(1)[0])


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

    def _apply_model_assumptions(self) -> None:
        self.parameter_space.M.lower_limit = 10 ** (4.5)
        self.parameter_space.M.upper_limit = 10 ** (6.0)

        self.parameter_space.a.value = 0.98
        self.parameter_space.a.is_fixed = True

        self.parameter_space.mu.value = 10
        self.parameter_space.mu.is_fixed = True

        self.parameter_space.e0.upper_limit = 0.2

        self.max_redshift = 1.5
        self.parameter_space.luminosity_distance.upper_limit = dist(redshift=self.max_redshift)
        self.luminostity_detection_threshold = 1.55  # as in Hitchikers Guide

    def emri_distribution(self, M: float, redshift: float) -> float:
        return self.dN_dz_of_mass(M, redshift) * self.R_emri(M)

    @staticmethod
    def dN_dz_of_mass(mass: float, redshift: float) -> float:
        mass_bin = float(np.log10(mass))
        if mass_bin < 4.5:
            return float(polynomial(redshift, *merger_distribution_coefficients[0]))
        elif mass_bin < 5.0:
            fraction = (mass_bin - 4.5) / 0.5
            return float(
                (1 - fraction) * polynomial(redshift, *merger_distribution_coefficients[0])
                + fraction * polynomial(redshift, *merger_distribution_coefficients[1])
            )
        elif mass_bin < 5.5:
            fraction = (mass_bin - 5.0) / 0.5
            return float(
                (1 - fraction) * polynomial(redshift, *merger_distribution_coefficients[1])
                + fraction * polynomial(redshift, *merger_distribution_coefficients[2])
            )
        elif mass_bin < 6.0:
            fraction = (mass_bin - 5.5) / 0.5
            return float(
                (1 - fraction) * polynomial(redshift, *merger_distribution_coefficients[2])
                + fraction * polynomial(redshift, *merger_distribution_coefficients[3])
            )
        else:  # mass_bin >= 6.25
            fraction = (mass_bin - 6.0) / 0.5
            fraction = min(fraction, 1.0)
            return float(
                (1 - fraction) * polynomial(redshift, *merger_distribution_coefficients[3])
                + fraction * polynomial(redshift, *merger_distribution_coefficients[4])
            )

    @staticmethod
    def R_emri(M: float) -> float:
        if M < 1.2e5:
            return float(10 ** ((1.02445) * np.log10(M / 1.2e5) + np.log10(33.1)))
        elif M < 2.5e5:
            return float(10 ** ((0.4689) * np.log10(M / 2.5e5) + np.log10(46.7)))
        else:
            return float(10 ** ((-0.2475) * np.log10(M / 2.9e7) + np.log10(14.4)))

    def _log_probability(self, M: float, redshift: float) -> float:
        if not self.parameter_space.M.lower_limit < M < self.parameter_space.M.upper_limit:
            return -np.inf
        if not 0 < redshift < self.max_redshift:
            return -np.inf
        return float(np.log(self.emri_distribution(M, redshift)))

    def setup_emri_events_sampler(self) -> None:
        # use emcee to sample the distribution

        def log_probability(x: list[float]) -> float:
            return self._log_probability(10 ** x[0], x[1])

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

    def sample_emri_events(self, number_of_samples: int) -> list[ParameterSample]:
        _LOGGER.info("Sampling EMRI events...")
        pos, prob, state = self._emri_event_sampler.run_mcmc(
            initial_state=self._sample_positions, nsteps=number_of_samples
        )
        samples = self._emri_event_sampler.get_chain(flat=True)
        self._sample_positions = pos
        self._emri_event_sampler.reset()
        return_samples = [
            ParameterSample(M=10 ** sample[0], redshift=sample[1], a=MBH_spin_distribution(0, 1))
            for sample in samples
        ]
        _LOGGER.info(f"Sampling complete (number of samples ({len(return_samples)})).")

        return return_samples

    def simplified_event_mass_dependency(self, mass: float) -> float:
        raise NotImplementedError

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

    def de_equation(self, z: float) -> float:
        return float(self.w_0 + z / (1 + z) / self.w_a)  # type: ignore[operator]


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
                (undetected_events["luminosity_distance"] - luminosity_distance_lower_limit)
                / (luminosity_distance_upper_limit - luminosity_distance_lower_limit),
                (np.log10(undetected_events["M"]) - np.log10(mass_lower_limit))
                / (np.log10(mass_upper_limit) - np.log10(mass_lower_limit)),
                undetected_events["phiS"] / (2 * np.pi),
                undetected_events["qS"] / np.pi,
            ]
        )

        detected_events_points = np.array(
            [
                (detected_events["luminosity_distance"] - luminosity_distance_lower_limit)
                / (luminosity_distance_upper_limit - luminosity_distance_lower_limit),
                (np.log10(detected_events["M"]) - np.log10(mass_lower_limit))
                / (np.log10(mass_upper_limit) - np.log10(mass_lower_limit)),
                detected_events["phiS"] / (2 * np.pi),
                detected_events["qS"] / np.pi,
            ]
        )

        self.kde_undetected_with_bh_mass = gaussian_kde(
            undetected_events_points, bw_method=bandwidth
        )
        self.kde_detected_with_bh_mass = gaussian_kde(detected_events_points, bw_method=bandwidth)

        # create kde and detection probability function for the case without BH mass
        undetected_events_points_without_bh_mass = np.delete(undetected_events_points, 1, axis=0)
        detected_events_points_without_bh_mass = np.delete(detected_events_points, 1, axis=0)

        self.kde_detected_without_bh_mass = gaussian_kde(
            detected_events_points_without_bh_mass, bw_method=bandwidth
        )
        self.kde_undetected_without_bh_mass = gaussian_kde(
            undetected_events_points_without_bh_mass, bw_method=bandwidth
        )

        self._setup_interpolator(d_L_steps=40, M_z_steps=50, phi_steps=20, theta_steps=20)

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
            ]
        ):
            return 0.0
        # normalize the input values to the range [0, 1]
        d_L, M_z, phi, theta = self._normalize_parameters(  # type: ignore[assignment,misc]
            d_L, phi, theta, M_z
        )

        detected_evaluated = self.kde_detected_with_bh_mass.evaluate([d_L, M_z, phi, theta])[0]
        undetected_evaluated = self.kde_undetected_with_bh_mass.evaluate([d_L, M_z, phi, theta])[0]
        if undetected_evaluated + detected_evaluated == 0.0:
            return 0.0
        return float(detected_evaluated / (undetected_evaluated + detected_evaluated))

    def evaluate_with_bh_mass_vectorized(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
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
        d_L_norm, M_z_norm, phi_norm, theta_norm = self._normalize_parameters(  # type: ignore[misc]
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
            ]
        ):
            return 0.0
        # normalize the input values to the range [0, 1]
        d_L, phi, theta = self._normalize_parameters(  # type: ignore[assignment,misc]
            d_L, phi, theta
        )
        detected_evaluated = self.kde_detected_without_bh_mass.evaluate([d_L, phi, theta])[0]
        undetected_evaluated = self.kde_undetected_without_bh_mass.evaluate([d_L, phi, theta])[0]
        if undetected_evaluated + detected_evaluated == 0:
            return 0.0
        return float(detected_evaluated / (undetected_evaluated + detected_evaluated))

    def _normalize_parameters(
        self,
        d_L: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
        M_z: float | npt.NDArray[np.float64] | None = None,
    ) -> (
        tuple[
            float | npt.NDArray[np.float64],
            float | npt.NDArray[np.float64],
            float | npt.NDArray[np.float64],
            float | npt.NDArray[np.float64],
        ]
        | tuple[
            float | npt.NDArray[np.float64],
            float | npt.NDArray[np.float64],
            float | npt.NDArray[np.float64],
        ]
    ):
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

    def _setup_interpolator(
        self, d_L_steps: int, M_z_steps: int, phi_steps: int, theta_steps: int
    ) -> None:
        # setup grid
        d_L_range = np.linspace(
            self.luminosity_distance_lower_limit,
            self.luminosity_distance_upper_limit,
            d_L_steps,
        )
        M_z_range = np.geomspace(self.mass_lower_limit, self.mass_upper_limit, M_z_steps)
        phi_range = np.linspace(0, 2 * np.pi, phi_steps)
        theta_range = np.linspace(0, np.pi, theta_steps)

        # normalize the ranges to [0, 1]
        d_L_range_norm, M_z_range_norm, phi_range_norm, theta_range_norm = (
            self._normalize_parameters(  # type: ignore[misc]
                d_L_range, phi_range, theta_range, M_z_range
            )
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
        d_L: float | npt.NDArray[np.float64],
        M_z: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
    ) -> float | npt.NDArray[np.float64]:
        # check if the input values are floats
        if all([isinstance(attribute, float) for attribute in [d_L, M_z, phi, theta]]):
            # if all attributes are float, convert them to 1D arrays
            d_L = np.array([d_L])
            M_z = np.array([M_z])
            phi = np.array([phi])
            theta = np.array([theta])
        return self.detection_probability_with_bh_mass_interpolator(  # type: ignore[no-any-return]
            np.array([d_L, M_z, phi, theta]).T
        )

    def detection_probability_without_bh_mass_interpolated(
        self,
        d_L: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
    ) -> float | npt.NDArray[np.float64]:
        # check if the input values are floats
        if all([isinstance(attribute, float) for attribute in [d_L, phi, theta]]):
            # if all attributes are float, convert them to 1D arrays
            d_L = np.array([d_L])
            phi = np.array([phi])
            theta = np.array([theta])
        return self.detection_probability_without_bh_mass_interpolator(  # type: ignore[no-any-return]
            np.array([d_L, phi, theta]).T
        )


class BayesianStatistics:
    cramer_rao_bounds: pd.DataFrame
    detection: Detection
    cosmological_model: LamCDMScenario
    h: float
    Omega_m: float
    Omega_DE: float
    w_0: float
    w_a: float
    h_values: list[float] = []
    h_values_with_bh_mass: list[float] = []
    galaxy_weights: dict[str, dict[str, list[float]]] = {}
    additional_galaxies_without_bh_mass: dict[str, dict[str, list[float]]] = {}
    posterior_data: dict[int, list[float]] = {}
    posterior_data_with_bh_mass: dict[int | str, Any] = {}

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

    def evaluate(
        self,
        galaxy_catalog: GalaxyCatalogueHandler,
        cosmological_model: Model1CrossCheck,
        h_value: float,
    ) -> None:
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
        luminosity_distance_upper_limit = max(self.cramer_rao_bounds["luminosity_distance"]) * (
            1 + PARAMETERSPACE_MARGIN
        )
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
            bandwidth=None,
        )
        _LOGGER.debug("Detection probability functions created.")

        _LOGGER.debug("Creating detection likelihood gaussian functions...")
        detection_likelihood_multivariate_gaussian_by_detection_index: dict[
            int,
            tuple[
                _multivariate.multivariate_normal_frozen, _multivariate.multivariate_normal_frozen
            ],
        ] = {}
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
                allow_singular=True,  # TODO: this should not be needed in the end
            )
            gaussian_with_bh_mass = multivariate_normal(
                mean=[detection.phi, detection.theta, 1, 1],
                cov=covariance_with_bh_mass,
                allow_singular=True,  # TODO: this should not be needed in the end
            )
            detection_likelihood_multivariate_gaussian_by_detection_index[index] = (
                gaussian_without_bh_mass,
                gaussian_with_bh_mass,
            )
        _LOGGER.debug("Detection likelihood gaussians created.")

        self.h = h_value

        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 1

        _LOGGER.debug(f"Found {available_cpus} / {os.cpu_count()} (available / system) cpus.")
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
                detection_likelihood_multivariate_gaussian_by_detection_index,
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
            f"simulations/posteriors/h_{str(np.round(self.h, 3)).replace('.', '_')}.json",
            "w",
        ) as file:
            data = {str(key): value for key, value in self.posterior_data.items()}
            json.dump(data | {"h": self.h}, file)

        with open(
            f"simulations/posteriors_with_bh_mass/h_{str(np.round(self.h, 3)).replace('.', '_')}.json",
            "w",
        ) as file:
            # update existing data

            data = {str(key): value for key, value in self.posterior_data_with_bh_mass.items()}
            json.dump(data | {"h": self.h}, file)

    def p_D(
        self,
        galaxy_catalog: GalaxyCatalogueHandler,
        redshift_upper_limit: float,
        pool: mp.pool.Pool,
    ) -> None:
        count = 0
        self.posterior_data_with_bh_mass[GALAXY_LIKELIHOODS] = {}
        self.posterior_data_with_bh_mass[ADDITIONAL_GALAXIES_WITHOUT_BH_MASS] = {}
        for index, detection in self.cramer_rao_bounds.iterrows():
            _LOGGER.info(f"Progess: detections: {count}/{len(self.cramer_rao_bounds)}...")
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
                sigma_multiplier=1.5,  # type: ignore[arg-type]
            )

            if possible_hosts is None:
                _LOGGER.debug("no possible hosts found...")
                continue
            possible_hosts, possible_hosts_with_bh_mass = possible_hosts  # type: ignore[assignment]
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
                possible_host_galaxies=possible_hosts,  # type: ignore[arg-type]
                possible_host_galaxies_with_bh_mass=possible_hosts_with_bh_mass,
                detection_index=index,
                pool=pool,
            )

            self.posterior_data[index].append(event_likelihood)
            self.posterior_data_with_bh_mass[index].append(event_likelihood_with_bh_mass)
            _LOGGER.debug(
                f"event likelihood: {event_likelihood}\nevent likelihood with bh mass: {event_likelihood_with_bh_mass}"
            )

    def p_Di(
        self,
        possible_host_galaxies: list[HostGalaxy],
        possible_host_galaxies_with_bh_mass: list[HostGalaxy],
        detection_index: int,
        pool: mp.pool.Pool,
    ) -> tuple[float, float]:
        # start parallel computation
        _LOGGER.info(f"start parallel computation with: {pool}")
        start = time.time()
        # remove duplicates from possible_host_galaxies already covered in possible_host_galaxies_with_bh_mass

        hosts_with_bh_mass_set = set(possible_host_galaxies_with_bh_mass)

        possible_host_galaxies_reduced = [
            host for host in possible_host_galaxies if host not in hosts_with_bh_mass_set
        ]

        _LOGGER.debug(
            f"reduced possible hosts galaxies to unique, removed {len(possible_host_galaxies) - len(possible_host_galaxies_reduced)} galaxies."
        )

        chunksize = math.ceil(len(possible_host_galaxies_reduced) / pool._processes)  # type: ignore[attr-defined]
        chunksize_with_bh_mass = math.ceil(
            len(possible_host_galaxies_with_bh_mass) / pool._processes  # type: ignore[attr-defined]
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
                [galaxy.catalog_index for galaxy in possible_host_galaxies_with_bh_mass],
                results_with_bh_mass,
            )
        )

        self.posterior_data_with_bh_mass[GALAXY_LIKELIHOODS][detection_index] = galaxy_likelihoods

        additional_likelihoods = list(
            zip(
                [galaxy.catalog_index for galaxy in possible_host_galaxies_reduced],
                results_without_blackhole_mass,
            )
        )

        self.posterior_data_with_bh_mass[ADDITIONAL_GALAXIES_WITHOUT_BH_MASS][detection_index] = (
            additional_likelihoods
        )

        if len(results_without_blackhole_mass) == 0:
            print("no results found")
            return 0.0, 0.0

        selection_effect_correction_without_bh_mass = np.sum(
            [result[1] for result in results_without_blackhole_mass]
        )
        numerator_without_bh_mass = [result[0] for result in results_without_blackhole_mass]
        numerator_without_bh_mass.extend([result[0] for result in results_with_bh_mass])

        likelihood_without_bh_mass = np.sum(numerator_without_bh_mass)

        selection_effect_correction_without_bh_mass += np.sum(
            [result[1] for result in results_with_bh_mass]
        )

        if len(results_with_bh_mass) == 0:
            return float(
                likelihood_without_bh_mass / selection_effect_correction_without_bh_mass
            ), 0.0

        likelihood_with_bh_mass = np.sum([result[2] for result in results_with_bh_mass])

        selection_effect_correction_with_bh_mass = np.sum(
            [result[3] for result in results_with_bh_mass]
        )

        return (
            float(likelihood_without_bh_mass / selection_effect_correction_without_bh_mass),
            float(likelihood_with_bh_mass / selection_effect_correction_with_bh_mass),
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


def gaussian(
    x: float | npt.NDArray[np.float64], mu: float, sigma: float, a: float
) -> float | npt.NDArray[np.float64]:
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


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
    return []


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
    denominator_integration_upper_redshift_limit = (
        possible_host.z + integration_limit_sigma_multiplier * possible_host.z_error
    )
    denominator_integration_lower_redshift_limit = (
        possible_host.z - integration_limit_sigma_multiplier * possible_host.z_error
    )

    # construct normal distribution for redshift and mass for host galaxy
    galaxy_redshift_normal_distribution = norm(loc=possible_host.z, scale=possible_host.z_error)

    # TODO: KEEP IN MIND SKYLOCALIZATION WEIGHT IS IN THE GW LIKELIHOOD ATM. possible source of error
    def numerator_integrant_without_bh_mass(z: npt.NDArray[np.float64]) -> Any:
        d_L = dist_vectorized(z, h=h)
        luminosity_distance_fraction = detection.d_L / d_L
        phi = np.full_like(z, possible_host.phiS)
        theta = np.full_like(z, possible_host.qS)

        return (
            detection_probability.detection_probability_without_bh_mass_interpolated(
                d_L, phi, theta
            )
            * detection_likelihood_gaussians_by_detection_index[detection_index][0].pdf(
                np.vstack([phi, theta, luminosity_distance_fraction]).T
            )
            * galaxy_redshift_normal_distribution.pdf(z)
            / d_L
        )

    def denominator_integrant_without_bh_mass(z: npt.NDArray[np.float64]) -> Any:
        d_L = dist_vectorized(z, h=h)
        phi = np.full_like(z, possible_host.phiS)
        theta = np.full_like(z, possible_host.qS)
        return detection_probability.detection_probability_without_bh_mass_interpolated(
            d_L, phi, theta
        ) * galaxy_redshift_normal_distribution.pdf(z)

    (
        single_host_likelihood_numerator_without_bh_mass,
        single_host_likelihood_numerator_without_bh_mass_error,
    ) = fixed_quad(
        numerator_integrant_without_bh_mass,
        numerator_integration_lower_redshift_limit,
        numerator_integration_upper_redshift_limit,
        n=FIXED_QUAD_N,
    )
    (
        single_host_likelihood_denominator_without_bh_mass,
        single_host_likelihood_denominator_without_bh_mass_error,
    ) = fixed_quad(
        denominator_integrant_without_bh_mass,
        denominator_integration_lower_redshift_limit,
        denominator_integration_upper_redshift_limit,
        n=FIXED_QUAD_N,
    )

    if evaluate_with_bh_mass:
        galaxy_mass_normal_distribution = norm(loc=possible_host.M, scale=possible_host.M_error)

        def numerator_integrant_with_bh_mass(z: npt.NDArray[np.float64]) -> Any:
            d_L = dist_vectorized(z, h=h)
            luminosity_distance_fraction = detection.d_L / d_L
            M_z = np.full_like(z, detection.M)
            phi = np.full_like(z, possible_host.phiS)
            theta = np.full_like(z, possible_host.qS)

            return (
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L, M_z, phi, theta
                )
                * detection_likelihood_gaussians_by_detection_index[detection_index][0].pdf(
                    np.vstack([phi, theta, luminosity_distance_fraction]).T
                )
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(detection.M / (1 + z))
                / (d_L * (1 + z))  # TODO: check if this is correct
            )

        single_host_likelihood_numerator_with_bh_mass = fixed_quad(
            numerator_integrant_with_bh_mass,
            numerator_integration_lower_redshift_limit,
            numerator_integration_upper_redshift_limit,
            n=FIXED_QUAD_N,
        )[0]

        def denominator_integrant_with_bh_mass_vectorized(
            M: npt.NDArray[np.float64], z: npt.NDArray[np.float64]
        ) -> Any:
            d_L = dist_vectorized(z, h=h)
            M_z = M * (1 + z)
            phi = np.full_like(M, possible_host.phiS)
            theta = np.full_like(M, possible_host.qS)
            return (
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L, M_z, phi, theta
                )
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )

        N_SAMPLES = 10_000
        z_samples = galaxy_redshift_normal_distribution.rvs(size=N_SAMPLES)
        M_samples = galaxy_mass_normal_distribution.rvs(size=N_SAMPLES)

        numerator_integrant_from_samples = denominator_integrant_with_bh_mass_vectorized(
            M_samples, z_samples
        )

        sampling_pdf = galaxy_redshift_normal_distribution.pdf(
            z_samples
        ) * galaxy_mass_normal_distribution.pdf(M_samples)
        weights = numerator_integrant_from_samples / sampling_pdf

        single_host_likelihood_denominator_with_bh_mass = np.mean(weights)

        return [
            single_host_likelihood_numerator_without_bh_mass,
            single_host_likelihood_denominator_without_bh_mass,
            single_host_likelihood_numerator_with_bh_mass,
            single_host_likelihood_denominator_with_bh_mass,
        ]
    return [
        single_host_likelihood_numerator_without_bh_mass,
        single_host_likelihood_denominator_without_bh_mass,
    ]


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
    galaxy_redshift_normal_distribution = norm(loc=possible_host.z, scale=possible_host.z_error)

    # TODO: KEEP IN MIND SKYLOCALIZATION WEIGHT IS IN THE GW LIKELIHOOD ATM. possible source of error
    def numerator_integrant_without_bh_mass(z: float) -> float:
        d_L = dist(z, h=h)
        luminosity_distance_fraction = d_L / detection.d_L
        return float(
            detection_probability.evaluate_without_bh_mass(
                d_L, possible_host.phiS, possible_host.qS
            )
            * detection_likelihood_gaussians_by_detection_index[detection_index][0].pdf(
                [possible_host.phiS, possible_host.qS, luminosity_distance_fraction]
            )
            * galaxy_redshift_normal_distribution.pdf(z)
        )

    def denominator_integrant_without_bh_mass(z: float) -> float:
        d_L = dist(z, h=h)
        return float(
            detection_probability.evaluate_without_bh_mass(
                d_L, possible_host.phiS, possible_host.qS
            )
            * galaxy_redshift_normal_distribution.pdf(z)
        )

    (
        single_host_likelihood_numerator_without_bh_mass,
        single_host_likelihood_numerator_without_bh_mass_error,
    ) = quad(
        numerator_integrant_without_bh_mass,
        redshift_lower_integration_limit,
        redshift_upper_integration_limit,
        epsabs=ABS_ERROR,
    )
    (
        single_host_likelihood_denominator_without_bh_mass,
        single_host_likelihood_denominator_without_bh_mass_error,
    ) = quad(
        denominator_integrant_without_bh_mass,
        redshift_lower_integration_limit,
        redshift_upper_integration_limit,
        epsabs=ABS_ERROR,
    )

    print(
        f"Numerator without bh m:{single_host_likelihood_numerator_without_bh_mass}, error estimation: {single_host_likelihood_numerator_without_bh_mass_error}",
        flush=True,
    )
    print(
        f"Denominator without bh m:{single_host_likelihood_denominator_without_bh_mass}, error estimation {single_host_likelihood_denominator_without_bh_mass_error}",
        flush=True,
    )

    if evaluate_with_bh_mass:
        galaxy_mass_normal_distribution = norm(loc=possible_host.M, scale=possible_host.M_error)
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

            return float(
                detection_probability.evaluate_with_bh_mass(
                    d_L, M_z, possible_host.phiS, possible_host.qS
                )
                * detection_likelihood_gaussians_by_detection_index[detection_index][0].pdf(
                    [possible_host.phiS, possible_host.qS, luminosity_distance_fraction]
                )
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
                / (1 + z)  # delta function derivative
            )

        def denominator_integrant_with_bh_mass(M: float, z: float) -> float:
            d_L = dist(z, h=h)
            M_z = M * (1 + z)
            return float(
                detection_probability.evaluate_with_bh_mass(
                    d_L, M_z, possible_host.phiS, possible_host.qS
                )
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )

        start = time.time()
        (
            single_host_likelihood_numerator_with_bh_mass,
            single_host_likelihood_numerator_with_bh_mass_error,
        ) = quad(
            numerator_integrant_with_bh_mass,
            redshift_lower_integration_limit,
            redshift_upper_integration_limit,
            epsabs=ABS_ERROR,
        )

        (
            single_host_likelihood_denominator_with_bh_mass,
            single_host_likelihood_denominator_with_bh_mass_error,
        ) = dblquad(
            denominator_integrant_with_bh_mass,
            galaxy_redshift_normal_distribution.mean()
            - 5 * galaxy_redshift_normal_distribution.std(),
            galaxy_redshift_normal_distribution.mean()
            + 5 * galaxy_redshift_normal_distribution.std(),
            lambda m: (
                galaxy_mass_normal_distribution.mean() - 5 * galaxy_mass_normal_distribution.std()
            ),
            lambda m: (
                galaxy_mass_normal_distribution.mean() + 5 * galaxy_mass_normal_distribution.std()
            ),
            epsabs=ABS_ERROR,
        )
        end = time.time()
        print(f"Time taken for delta function approximation: {end - start}s", flush=True)

        print(
            f"Numerator with bh m:{single_host_likelihood_numerator_with_bh_mass}, error estimation: {single_host_likelihood_numerator_with_bh_mass_error}",
            flush=True,
        )
        print(
            f"Denominator with bh m:{single_host_likelihood_denominator_with_bh_mass}, error estimation {single_host_likelihood_denominator_with_bh_mass_error}",
            flush=True,
        )

        # monte carlo integration denominator 2D
        start = time.time()

        def denominator_integrant_with_bh_mass_vectorized(
            M: npt.NDArray[np.float64], z: npt.NDArray[np.float64]
        ) -> Any:
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

        numerator_integrant_from_samples = denominator_integrant_with_bh_mass_vectorized(
            M_samples, z_samples
        )

        sampling_pdf = galaxy_redshift_normal_distribution.pdf(
            z_samples
        ) * galaxy_mass_normal_distribution.pdf(M_samples)
        weights = numerator_integrant_from_samples / sampling_pdf

        integral = np.mean(weights)
        integral_error = np.std(weights) / np.sqrt(N_SAMPLES)
        end = time.time()
        print(f"Time taken for monte carlo integration: {end - start}s", flush=True)
        print(
            f"Monte Carlo denominator integral with bh mass: {integral}, error estimation: {integral_error}",
            flush=True,
        )
        print(
            f"Integration difference: {abs(single_host_likelihood_denominator_with_bh_mass - integral)}",
            flush=True,
        )

        return [
            single_host_likelihood_numerator_without_bh_mass,
            single_host_likelihood_denominator_without_bh_mass,
            single_host_likelihood_numerator_with_bh_mass,
            single_host_likelihood_denominator_with_bh_mass,
        ]
    return [
        single_host_likelihood_numerator_without_bh_mass,
        single_host_likelihood_denominator_without_bh_mass,
    ]


def child_process_init(
    redshift_lower_limit: float,
    redshift_upper_limit: float,
    bh_mass_lower_limit: float,
    bh_mass_upper_limit: float,
    current_detection_probability: DetectionProbability,
    current_detection_likelihood_gaussians_by_detection_index: dict[
        int,
        tuple[_multivariate.multivariate_normal_frozen, _multivariate.multivariate_normal_frozen],
    ],
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


def check_overflow(arr_1: np.ndarray, arr_2: np.ndarray) -> bool:
    try:
        arr = arr_1 * arr_2
    except RuntimeWarning:
        _LOGGER.warning("Overflow detected in multiplication")
        return True
    return bool(np.any(np.isinf(arr)))


def _get_closest_possible_host(
    detection: Detection, possible_hosts: list[HostGalaxy]
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
    return possible_hosts[int(np.argmin(distances))]


def _distance_spherical_coordinates(
    phi1: float, theta1: float, phi2: float, theta2: float
) -> float:
    return float(
        np.arccos(
            np.sin(theta1) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2) * np.cos(phi1 - phi2)
        )
    )


def compute_sigma_deviation(
    sigma: float, sigma_error: float, h_mean: float, h_mean_error: float
) -> tuple[float, float]:
    sigma_dev = (h_mean - H) / sigma
    sigma_dev_error = float(np.sqrt((sigma_error * sigma_dev) ** 2 + (h_mean_error) ** 2) / sigma)
    return sigma_dev, sigma_dev_error
