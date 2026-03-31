"""Pipeline B — science-grade Hubble constant posterior evaluation.

:class:`BayesianStatistics` loads saved Cramér-Rao bounds and orchestrates the
full Hubble-constant posterior evaluation using the real GLADE galaxy catalog,
simulation-based :class:`~master_thesis_code.bayesian_inference.simulation_detection_probability.SimulationDetectionProbability`,
full Fisher-matrix covariance, and multiprocessing.

Invoked via ``main.py:evaluate()`` / ``--evaluate`` CLI flag.
Output is written to ``simulations/posteriors/`` as JSON.

For the simpler dev cross-check pipeline, see **Pipeline A**
(:class:`~master_thesis_code.bayesian_inference.bayesian_inference.BayesianInference`).
"""

import json
import logging
import math
import multiprocessing as mp
import os
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import dblquad, fixed_quad, quad
from scipy.stats import _multivariate, multivariate_normal, norm

from master_thesis_code.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from master_thesis_code.constants import (
    CRAMER_RAO_BOUNDS_OUTPUT_PATH,
    INJECTION_DATA_DIR,
    PREPARED_CRAMER_RAO_BOUNDS_PATH,
    SNR_THRESHOLD,
    H,
)
from master_thesis_code.cosmological_model import LamCDMScenario, Model1CrossCheck
from master_thesis_code.datamodels.detection import (
    Detection,
    _sky_localization_uncertainty,
)
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
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

FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD = 0.10

# Module-level globals used by child_process_init for multiprocessing worker state
redshift_upper_integration_limit: float = 0.0
redshift_lower_integration_limit: float = 0.0
bh_mass_upper_integration_limit: float = 0.0
bh_mass_lower_integration_limit: float = 0.0
detection_probability: Any = None
detection_likelihood_gaussians_by_detection_index: Any = None


class BayesianStatistics:
    """Pipeline B — science-grade Hubble constant posterior evaluation.

    Loads saved Cramér-Rao bounds from CSV, constructs a simulation-based
    :class:`SimulationDetectionProbability`, builds multivariate-normal GW
    likelihoods from the full Fisher-matrix covariance, and evaluates
    per-detection posteriors over an H₀ grid using a multiprocessing pool.

    Invoked via ``main.py:evaluate()`` (``--evaluate`` CLI flag).
    Output is written to ``simulations/posteriors/`` as JSON.

    For the simpler dev cross-check pipeline, see **Pipeline A**
    (:class:`~master_thesis_code.bayesian_inference.bayesian_inference.BayesianInference`).
    """

    cramer_rao_bounds: pd.DataFrame
    detection: Detection
    cosmological_model: LamCDMScenario
    h: float
    Omega_m: float
    Omega_DE: float
    w_0: float
    w_a: float
    h_values: list[float]
    h_values_with_bh_mass: list[float]
    galaxy_weights: dict[str, dict[str, list[float]]]
    additional_galaxies_without_bh_mass: dict[str, dict[str, list[float]]]
    posterior_data: dict[int, list[float]]
    posterior_data_with_bh_mass: dict[int | str, Any]

    def __init__(self) -> None:
        self.h_values = []
        self.h_values_with_bh_mass = []
        self.galaxy_weights = {}
        self.additional_galaxies_without_bh_mass = {}
        self.posterior_data = {}
        self.posterior_data_with_bh_mass = {}
        self.cramer_rao_bounds = pd.read_csv(PREPARED_CRAMER_RAO_BOUNDS_PATH)
        self.true_cramer_rao_bounds = pd.read_csv(CRAMER_RAO_BOUNDS_OUTPUT_PATH)
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
        num_workers: int | None = None,
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

        _LOGGER.debug("Creating detection probability functions...")
        detection_probability = SimulationDetectionProbability(
            injection_data_dir=INJECTION_DATA_DIR,
            snr_threshold=SNR_THRESHOLD,
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

        if num_workers is None:
            try:
                available_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                available_cpus = os.cpu_count() or 1
            num_workers = max(1, available_cpus - 2)
        _LOGGER.debug(f"Using {num_workers} worker(s) for multiprocessing pool.")

        with mp.get_context("spawn").Pool(
            num_workers,
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

    # Sky localization weight (phi, theta) is inside the GW likelihood Gaussian.
    # Verified correct by Phase 14 derivation (Sec. 2.7): the 3D/4D GW Gaussian
    # naturally encodes the sky position weight -- this is NOT a source of error.
    def numerator_integrant_without_bh_mass(z: npt.NDArray[np.float64]) -> Any:
        d_L = dist_vectorized(z, h=h)
        # fraction = d_L_model / d_L_measured; matches covariance σ²/d_L_measured²
        luminosity_distance_fraction = d_L / detection.d_L
        phi = np.full_like(z, possible_host.phiS)
        theta = np.full_like(z, possible_host.qS)

        p_det = detection_probability.detection_probability_without_bh_mass_interpolated(
            d_L, phi, theta, h=h
        )
        return (
            p_det
            * detection_likelihood_gaussians_by_detection_index[detection_index][0].pdf(
                np.vstack([phi, theta, luminosity_distance_fraction]).T
            )
            * galaxy_redshift_normal_distribution.pdf(z)
        )

    def denominator_integrant_without_bh_mass(z: npt.NDArray[np.float64]) -> Any:
        d_L = dist_vectorized(z, h=h)
        phi = np.full_like(z, possible_host.phiS)
        theta = np.full_like(z, possible_host.qS)
        p_det = detection_probability.detection_probability_without_bh_mass_interpolated(
            d_L, phi, theta, h=h
        )
        return p_det * galaxy_redshift_normal_distribution.pdf(z)

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

        # --- Precompute conditional distribution for analytic M_z marginalization ---
        # Partition 4D covariance [phi, theta, d_L_frac, M_z_frac] into
        # observed (indices 0-2) and M_z_frac (index 3).
        # Eqs. (14.23)-(14.28) in derivations/dark_siren_likelihood.md
        # Ref: Bishop (2006) PRML Eq. 2.81-2.82 (multivariate normal conditioning)
        gaussian_4d = detection_likelihood_gaussians_by_detection_index[detection_index][1]
        cov_4d = np.asarray(gaussian_4d.cov)
        mu_obs_4d = np.asarray(gaussian_4d.mean)

        cov_obs = cov_4d[:3, :3]  # 3x3: phi, theta, d_L_frac
        cov_cross = cov_4d[3, :3]  # (3,): cross-covariance M_z_frac with obs
        cov_mz = cov_4d[3, 3]  # scalar: M_z_frac variance

        # Use pseudoinverse for robustness against near-singular Fisher matrices
        cov_obs_inv = np.linalg.pinv(cov_obs)

        # Conditional variance of M_z_frac given (phi, theta, d_L_frac)
        sigma2_cond = float(cov_mz - cov_cross @ cov_obs_inv @ cov_cross)
        sigma2_cond = max(sigma2_cond, 1e-30)  # floor to avoid numerical issues

        # Projection vector for conditional mean: μ_cond = μ_Mz + proj · (x_obs - μ_obs)
        proj = cov_cross @ cov_obs_inv  # (3,) vector

        # 3D marginal Gaussian for observed variables (marginalized over M_z_frac)
        # Marginal covariance is just the 3x3 submatrix of the 4D covariance
        gaussian_3d_marginal = multivariate_normal(
            mean=mu_obs_4d[:3], cov=cov_obs, allow_singular=True
        )

        def numerator_integrant_with_bh_mass(z: npt.NDArray[np.float64]) -> Any:
            d_L = dist_vectorized(z, h=h)
            luminosity_distance_fraction = d_L / detection.d_L
            phi = np.full_like(z, possible_host.phiS)
            theta = np.full_like(z, possible_host.qS)

            # NOTE: p_det uses the ML mass estimate (detection.M) rather than
            # M_gal*(1+z) at trial z. This is a known approximation, not a bug,
            # per Phase 14 analysis. The denominator uses M_gal*(1+z) correctly.
            p_det = detection_probability.detection_probability_with_bh_mass_interpolated(
                d_L, np.full_like(z, detection.M), phi, theta, h=h
            )

            # 3D marginal Gaussian: p(phi, theta, d_L_frac)
            gw_3d = gaussian_3d_marginal.pdf(
                np.vstack([phi, theta, luminosity_distance_fraction]).T
            )

            # Conditional mean of M_z_frac given (phi_gal, theta_gal, d_L_frac)
            x_obs = np.vstack([phi, theta, luminosity_distance_fraction]).T  # (N, 3)
            mu_cond = mu_obs_4d[3] + (x_obs - mu_obs_4d[:3]) @ proj  # (N,)

            # Galaxy mass in M_z_frac coordinates: M_z_frac = M_gal * (1+z) / M_z_det
            # Eq. (14.22) in derivations/dark_siren_likelihood.md
            # NOTE: (1+z) here is CORRECT -- it is the coordinate transform, not a Jacobian
            mu_gal_frac = possible_host.M * (1 + z) / detection.M
            sigma_gal_frac = possible_host.M_error * (1 + z) / detection.M

            # Analytic Gaussian product integral:
            # ∫ N(x; μ_cond, σ²_cond) · N(x; μ_gal, σ²_gal) dx
            #   = N(μ_cond; μ_gal, σ²_cond + σ²_gal)
            # Eq. (14.31) in derivations/dark_siren_likelihood.md
            sigma2_sum = sigma2_cond + sigma_gal_frac**2
            mz_integral = np.exp(-0.5 * (mu_cond - mu_gal_frac) ** 2 / sigma2_sum) / np.sqrt(
                2 * np.pi * sigma2_sum
            )

            # Eq. (14.32) in derivations/dark_siren_likelihood.md
            # No /(1+z) factor: Jacobian absorbed by Gaussian rescaling (Eq. 14.21)
            return p_det * gw_3d * mz_integral * galaxy_redshift_normal_distribution.pdf(z)

        single_host_likelihood_numerator_with_bh_mass = fixed_quad(
            numerator_integrant_with_bh_mass,
            numerator_integration_lower_redshift_limit,
            numerator_integration_upper_redshift_limit,
            n=FIXED_QUAD_N,
        )[0]

        # Eq. (14.33) in derivations/dark_siren_likelihood.md
        # Denominator: p_det(d_L, M_z, phi, theta) * p_gal(z) * p_gal(M)
        # No GW likelihood, no mz_integral, no /(1+z) -- confirmed correct by Phase 14
        def denominator_integrant_with_bh_mass_vectorized(
            M: npt.NDArray[np.float64], z: npt.NDArray[np.float64]
        ) -> Any:
            d_L = dist_vectorized(z, h=h)
            M_z = M * (1 + z)
            phi = np.full_like(M, possible_host.phiS)
            theta = np.full_like(M, possible_host.qS)
            p_det = detection_probability.detection_probability_with_bh_mass_interpolated(
                d_L, M_z, phi, theta, h=h
            )
            return (
                p_det
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )

        # MC importance sampling for 2D denominator integral over (z, M).
        # Proposal distribution: q(z, M) = p_gal(z) * p_gal(M).
        # After cancellation, weights = p_det (each in [0, 1]).
        # Relative MC error ~ std(p_det) / (sqrt(N) * mean(p_det)) ~ 1% for N=10000.
        # Numerator uses fixed_quad (1D over z, mass analytically marginalized) --
        # the quadrature-vs-MC asymmetry is a numerical choice, not a physics error.
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

    # Sky localization weight (phi, theta) is inside the GW likelihood Gaussian.
    # Verified correct by Phase 14 derivation (Sec. 2.7) -- not a source of error.
    def numerator_integrant_without_bh_mass(z: float) -> float:
        d_L = dist(z, h=h)
        luminosity_distance_fraction = d_L / detection.d_L
        return float(
            detection_probability.detection_probability_without_bh_mass_interpolated(
                d_L, possible_host.phiS, possible_host.qS, h=h
            )
            * detection_likelihood_gaussians_by_detection_index[detection_index][0].pdf(
                [possible_host.phiS, possible_host.qS, luminosity_distance_fraction]
            )
            * galaxy_redshift_normal_distribution.pdf(z)
        )

    def denominator_integrant_without_bh_mass(z: float) -> float:
        d_L = dist(z, h=h)
        return float(
            detection_probability.detection_probability_without_bh_mass_interpolated(
                d_L, possible_host.phiS, possible_host.qS, h=h
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
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L, M_z, possible_host.phiS, possible_host.qS, h=h
                )
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
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L, M_z, possible_host.phiS, possible_host.qS, h=h
                )
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

        # Analytic marginalization over M_z_frac (same as production path)
        # Ref: Bishop (2006) PRML Eq. 2.81-2.82
        gaussian_4d_test = detection_likelihood_gaussians_by_detection_index[detection_index][1]
        cov_4d_test = np.asarray(gaussian_4d_test.cov)
        mu_obs_4d_test = np.asarray(gaussian_4d_test.mean)
        cov_obs_test = cov_4d_test[:3, :3]
        cov_cross_test = cov_4d_test[3, :3]
        cov_mz_test = cov_4d_test[3, 3]
        cov_obs_inv_test = np.linalg.pinv(cov_obs_test)
        sigma2_cond_test = float(cov_mz_test - cov_cross_test @ cov_obs_inv_test @ cov_cross_test)
        sigma2_cond_test = max(sigma2_cond_test, 1e-30)
        proj_test = cov_cross_test @ cov_obs_inv_test
        gaussian_3d_marginal_test = multivariate_normal(
            mean=mu_obs_4d_test[:3], cov=cov_obs_test, allow_singular=True
        )

        def numerator_integrant_with_bh_mass(z: float) -> float:
            d_L = dist(z, h=h)
            luminosity_distance_fraction = d_L / detection.d_L

            x_obs_test = np.array(
                [possible_host.phiS, possible_host.qS, luminosity_distance_fraction]
            )
            gw_3d = float(gaussian_3d_marginal_test.pdf(x_obs_test))

            mu_cond = float(mu_obs_4d_test[3] + proj_test @ (x_obs_test - mu_obs_4d_test[:3]))
            mu_gal_frac = possible_host.M * (1 + z) / detection.M
            sigma_gal_frac = possible_host.M_error * (1 + z) / detection.M
            sigma2_sum = sigma2_cond_test + sigma_gal_frac**2
            mz_integral = float(
                np.exp(-0.5 * (mu_cond - mu_gal_frac) ** 2 / sigma2_sum)
                / np.sqrt(2 * np.pi * sigma2_sum)
            )

            # Eq. (14.32) in derivations/dark_siren_likelihood.md
            # No /(1+z) factor: Jacobian absorbed by Gaussian rescaling (Eq. 14.21)
            return float(
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L, detection.M, possible_host.phiS, possible_host.qS, h=h
                )
                * gw_3d
                * mz_integral
                * galaxy_redshift_normal_distribution.pdf(z)
            )

        def denominator_integrant_with_bh_mass(M: float, z: float) -> float:
            d_L = dist(z, h=h)
            M_z = M * (1 + z)
            return float(
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L, M_z, possible_host.phiS, possible_host.qS, h=h
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
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L, M_z, phi, theta, h=h
                )
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
    current_detection_probability: SimulationDetectionProbability,
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
