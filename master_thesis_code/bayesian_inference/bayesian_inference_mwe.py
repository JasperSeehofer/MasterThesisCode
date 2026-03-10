import multiprocessing as mp
from dataclasses import dataclass, field
from statistics import NormalDist
from time import time
from typing import Any

import emcee
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

# import error function
from scipy.special import erf
from scipy.stats import truncnorm

from master_thesis_code.bayesian_inference.scientific_plotter import ScientificPlotter
from master_thesis_code.constants import (
    GALAXY_REDSHIFT_ERROR_COEFFICIENT,
    OMEGA_M,
    TRUE_HUBBLE_CONSTANT,
    H,
)
from master_thesis_code.constants import (
    OMEGA_DE as OMEGA_LAMBDA,
)
from master_thesis_code.constants import (
    SPEED_OF_LIGHT_KM_S as SPEED_OF_LIGHT,
)
from master_thesis_code.physical_relations import (
    dist,
)
from master_thesis_code.physical_relations import (
    dist_to_redshift as dist_to_redshift,  # re-exported: test_bayesian_inference_mwe imports it
)
from master_thesis_code.physical_relations import (
    dist_vectorized as _dist_vectorized,
)
from master_thesis_code.physical_relations import (
    lambda_cdm_analytic_distance as lambda_cdm_analytic_distance,  # re-exported
)
from master_thesis_code.physical_relations import (
    redshifted_mass as redshifted_mass,  # re-exported: test_bayesian_inference_mwe imports it
)
from master_thesis_code.physical_relations import (
    redshifted_mass_inverse as redshifted_mass_inverse,  # re-exported
)

FRACTIONAL_LUMINOSITY_ERROR: float = 0.1
FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR: float = 0.1
FRACTIONAL_MEASURED_MASS_ERROR: float = 1e-8  # TODO: Check with parameter estimation
SKY_LOCALIZATION_ERROR: float = 2 / 180 * np.pi  # radians


def dist_array(
    redshifts: npt.NDArray[np.float64],
    h: float = H,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_LAMBDA,
) -> npt.NDArray[np.float64]:
    """Vectorized luminosity distance in Gpc over an array of redshifts.

    Delegates to physical_relations.dist_vectorized for a canonical, unit-consistent
    implementation.  Returns Gpc (same unit as the scalar dist()).
    """
    return np.asarray(_dist_vectorized(redshifts, h=h, Omega_m=Omega_m, Omega_de=Omega_de))


@dataclass(unsafe_hash=True)
class Galaxy:
    redshift: float
    central_black_hole_mass: float  # same as massive black hole
    right_ascension: float
    declination: float

    @classmethod
    def with_random_skylocalization(
        cls, redshift: float, central_black_hole_mass: float
    ) -> "Galaxy":
        # get spherically uniform distributed sky localization
        right_ascension = np.random.uniform(0, 2 * np.pi)
        declination = np.arccos(np.random.uniform(-1, 1))
        return cls(
            redshift=redshift,
            central_black_hole_mass=central_black_hole_mass,
            right_ascension=right_ascension,
            declination=declination,
        )

    @property
    def redshift_uncertainty(self) -> float:
        return min(GALAXY_REDSHIFT_ERROR_COEFFICIENT * (1 + self.redshift) ** 3, 0.015)

    @property
    def central_black_hole_mass_uncertainty(self) -> float:
        return FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR * self.central_black_hole_mass


@dataclass
class GalaxyCatalog:
    _use_truncnorm: bool
    _use_comoving_volume: bool

    lower_mass_limit: float = 10 ** (4)
    upper_mass_limit: float = 10 ** (7)
    redshift_lower_limit: float = 0.00001
    redshift_upper_limit: float = 0.55
    catalog: list[Galaxy] = field(default_factory=list)
    galaxy_distribution: list[NormalDist] = field(default_factory=list)
    galaxy_mass_distribution: list[NormalDist] = field(default_factory=list)

    def __init__(
        self,
        use_truncnorm: bool = True,
        use_comoving_volume: bool = True,
        h0: float = TRUE_HUBBLE_CONSTANT,
    ):
        self._use_truncnorm = use_truncnorm
        self._use_comoving_volume = use_comoving_volume
        self._h0 = h0
        self.catalog = []
        self.galaxy_distribution = []
        self.galaxy_mass_distribution = []
        self._comoving_volume_spline = self._build_comoving_volume_spline(h0)

    @staticmethod
    def _build_comoving_volume_spline(h0: float = TRUE_HUBBLE_CONSTANT) -> CubicSpline:
        """Precompute the comoving volume on a fine redshift grid and return a spline.

        The integral ∫₀ᶻ dz'/E(z') is computed once via cumulative_trapezoid so subsequent
        calls to comoving_volume() are O(log n) interpolation instead of O(100) integration.
        """
        _z_grid = np.linspace(0, 10.0, 4000)
        integrand = 1.0 / np.sqrt(OMEGA_M * (1 + _z_grid) ** 3 + OMEGA_LAMBDA)
        cumulative_integral = np.concatenate([[0.0], cumulative_trapezoid(integrand, _z_grid)])
        cv_grid = 4 * np.pi * (SPEED_OF_LIGHT / h0) ** 3 * cumulative_integral**2
        return CubicSpline(_z_grid, cv_grid)

    def comoving_volume(self, redshift: float) -> float:
        return float(self._comoving_volume_spline(redshift))

    def log_comoving_volume(self, redshift: float) -> float:
        try:
            redshift = redshift[0]  # type: ignore[index]
        except TypeError:
            pass
        if redshift < self.redshift_lower_limit or redshift > self.redshift_upper_limit:
            return float(-np.inf)
        return float(np.log(self.comoving_volume(redshift)))

    def get_samples_from_comoving_volume(self, number_of_samples: int) -> np.ndarray:
        # use emcee to sample the comoving volume distribution
        ndim = 1
        nwalkers = 5
        p0 = np.random.uniform(
            self.redshift_lower_limit, self.redshift_upper_limit, (nwalkers, ndim)
        )
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_comoving_volume)
        # burn-in
        p0, _, _ = sampler.run_mcmc(p0, 1000)
        sampler.reset()
        sampler.run_mcmc(p0, int(number_of_samples / nwalkers) + 1)
        samples = np.array(sampler.get_chain(flat=True)).flatten()
        # return required number of samples by using the last n samples
        reduced_samples = samples[-number_of_samples:]
        self.save_comoving_volume_sampling_plot(reduced_samples)
        return reduced_samples

    @staticmethod
    def save_comoving_volume_sampling_plot(samples: np.ndarray) -> None:
        # plot and save the sampling distribution for an explicit figure
        sample_fig, sample_ax = plt.subplots(figsize=(16, 9))
        sample_ax.set_title("Comoving Volume Sampling")
        sample_ax.set_xlabel("Redshift")
        sample_ax.set_ylabel("Density")
        sample_ax.hist(samples, bins=20, density=True)

        sample_fig.savefig("comoving_volume_sampling.png")
        plt.close(sample_fig)

    def plot_comoving_volume(self) -> None:
        redshifts = np.linspace(self.redshift_lower_limit, self.redshift_upper_limit, 100)
        comoving_volumes = [self.comoving_volume(redshift) for redshift in redshifts]
        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.plot(redshifts, np.array(comoving_volumes), label="Comoving volume")
        _plotter.show_and_close()

    def create_random_catalog(self, number_of_galaxies: int) -> None:
        # draw mass from uniform in log space
        print(
            f"Creating random galaxy catalog with {number_of_galaxies} galaxies in the redshift range ({self.redshift_lower_limit}, {self.redshift_upper_limit}) / ({dist(self.redshift_lower_limit)}, {dist(self.redshift_upper_limit)}) Gpc."
        )
        if self._use_comoving_volume:
            redshift_samples = self.get_samples_from_comoving_volume(number_of_galaxies)
            assert len(redshift_samples) == number_of_galaxies
            for redshift in redshift_samples:
                self.catalog.append(
                    Galaxy.with_random_skylocalization(
                        redshift=redshift,
                        central_black_hole_mass=10
                        ** np.random.uniform(
                            np.log10(self.lower_mass_limit),
                            np.log10(self.upper_mass_limit),
                        ),
                    )
                )
        else:
            for i in range(number_of_galaxies):
                self.catalog.append(
                    Galaxy.with_random_skylocalization(
                        redshift=np.random.uniform(
                            self.redshift_lower_limit, self.redshift_upper_limit
                        ),
                        central_black_hole_mass=10
                        ** np.random.uniform(
                            np.log10(self.lower_mass_limit),
                            np.log10(self.upper_mass_limit),
                        ),
                    )
                )
        self.setup_galaxy_distribution()
        self.setup_galaxy_mass_distribution()

    def remove_all_galaxies(self) -> None:
        self.catalog = []
        self.galaxy_distribution = []
        self.galaxy_mass_distribution = []

    def add_random_galaxy(self) -> None:
        galaxy = Galaxy.with_random_skylocalization(
            redshift=np.random.uniform(self.redshift_lower_limit, self.redshift_upper_limit),
            central_black_hole_mass=np.random.uniform(self.lower_mass_limit, self.upper_mass_limit),
        )
        self.catalog.append(galaxy)
        self.append_galaxy_to_galaxy_distribution(galaxy)
        self.append_galaxy_to_galaxy_mass_distribution(galaxy)

    def add_host_galaxy(self) -> Galaxy:
        galaxy = Galaxy.with_random_skylocalization(
            redshift=np.random.uniform(self.redshift_lower_limit, self.redshift_upper_limit),
            central_black_hole_mass=np.random.uniform(self.lower_mass_limit, self.upper_mass_limit),
        )
        self.catalog.append(galaxy)
        self.append_galaxy_to_galaxy_distribution(galaxy)
        self.append_galaxy_to_galaxy_mass_distribution(galaxy)
        return galaxy

    def setup_galaxy_distribution(self) -> None:
        if not self._use_truncnorm:
            self.galaxy_distribution = [
                NormalDist(mu=galaxy.redshift, sigma=galaxy.redshift_uncertainty)
                for galaxy in self.catalog
            ]
        else:
            self.galaxy_distribution = [
                truncnorm(
                    a=(self.redshift_lower_limit - galaxy.redshift) / galaxy.redshift_uncertainty,
                    b=(self.redshift_upper_limit - galaxy.redshift) / galaxy.redshift_uncertainty,
                )
                for galaxy in self.catalog
            ]

    def append_galaxy_to_galaxy_mass_distribution(self, galaxy: Galaxy) -> None:
        if not self._use_truncnorm:
            self.galaxy_mass_distribution.append(
                NormalDist(
                    mu=galaxy.central_black_hole_mass,
                    sigma=FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR * galaxy.central_black_hole_mass,
                )
            )

        else:
            self.galaxy_mass_distribution.append(
                truncnorm(
                    a=(self.lower_mass_limit - galaxy.central_black_hole_mass)
                    / galaxy.central_black_hole_mass_uncertainty,
                    b=(self.upper_mass_limit - galaxy.central_black_hole_mass)
                    / galaxy.central_black_hole_mass_uncertainty,
                )
            )

    def append_galaxy_to_galaxy_distribution(self, galaxy: Galaxy) -> None:
        if not self._use_truncnorm:
            self.galaxy_distribution.append(
                NormalDist(galaxy.redshift, galaxy.redshift_uncertainty)
            )
        else:
            self.galaxy_distribution.append(
                truncnorm(
                    a=(self.redshift_lower_limit - galaxy.redshift) / galaxy.redshift_uncertainty,
                    b=(self.redshift_upper_limit - galaxy.redshift) / galaxy.redshift_uncertainty,
                )
            )

    def evaluate_galaxy_distribution(self, redshift: float) -> npt.NDArray[np.float64]:
        redshift_uncertainty = min(GALAXY_REDSHIFT_ERROR_COEFFICIENT * (1 + redshift) ** 3, 0.015)
        p_background: float = 1.0

        if self._use_comoving_volume:
            p_background = self.comoving_volume(redshift)

        normal_dist = NormalDist(mu=redshift, sigma=redshift_uncertainty)

        if self._use_truncnorm:
            normalization = (
                normal_dist.cdf(self.redshift_upper_limit)
                - normal_dist.cdf(self.redshift_lower_limit)
            ) * normal_dist.stdev
            return np.array(
                [
                    normal_dist.pdf(galaxy.redshift) / normalization
                    # Adjust for background
                    for galaxy in self.catalog
                ]
            ) / len(self.catalog)

        return np.array(
            [
                normal_dist.pdf(galaxy.redshift)
                # Adjust for background
                for galaxy in self.catalog
            ]
        ) / len(self.catalog)

        """           
        return np.array([
                    distribution.pdf(redshift)
                    / (1 - distribution.cdf(self.redshift_lower_limit))
                    / distribution.stdev
                    for distribution in self.galaxy_distribution
                ]) / len(self.catalog)
        return np.array([distribution.pdf(redshift) for distribution in self.galaxy_distribution]) / len(self.catalog)
        """

    def setup_galaxy_mass_distribution(self) -> None:
        self.galaxy_mass_distribution = [
            NormalDist(
                mu=galaxy.central_black_hole_mass,
                sigma=FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR * 10**5.5,
            )
            for galaxy in self.catalog
        ]

    def evaluate_galaxy_mass_distribution(self, mass: float) -> npt.NDArray[np.float64]:
        # use truncated normal distribution

        if self._use_truncnorm:
            return np.array(
                [
                    distribution.pdf(mass)
                    / distribution.stdev
                    / (
                        distribution.cdf(self.upper_mass_limit)
                        - distribution.cdf(self.lower_mass_limit)
                    )
                    for distribution in self.galaxy_mass_distribution
                ]
            ) / len(self.catalog)

        # without truncnorm
        return np.array(
            [distribution.pdf(mass) for distribution in self.galaxy_mass_distribution]
        ) / len(self.catalog)

    def get_possible_host_galaxies(self) -> list[Galaxy]:
        return [
            galaxy
            for galaxy in self.catalog
            if dist(galaxy.redshift, TRUE_HUBBLE_CONSTANT)
            <= BayesianInference.luminosity_distance_threshold
        ]

    def get_unique_host_galaxies_from_catalog(self, number_of_host_galaxies: int) -> list[Galaxy]:
        if len(self.get_possible_host_galaxies()) < number_of_host_galaxies:
            print("Not enough possible host galaxies in catalog.")
            return []
        possible: list[Any] = self.get_possible_host_galaxies()
        return list(np.random.choice(possible, number_of_host_galaxies, replace=False))

    def add_unique_host_galaxies_from_catalog(
        self, number_of_host_galaxies_to_add: int, used_host_galaxies: list[Galaxy]
    ) -> list[Galaxy]:
        if (
            len(self.get_possible_host_galaxies()) - len(used_host_galaxies)
            < number_of_host_galaxies_to_add
        ):
            print("Not enough possible host galaxies in catalog.")
            return used_host_galaxies
        filtered: list[Any] = [
            galaxy
            for galaxy in self.get_possible_host_galaxies()
            if galaxy not in used_host_galaxies
        ]
        new_host_galaxies = np.random.choice(
            filtered, number_of_host_galaxies_to_add, replace=False
        )
        used_host_galaxies.extend(new_host_galaxies)
        return used_host_galaxies

    def plot_galaxy_catalog(self) -> None:
        redshifts = np.linspace(self.redshift_lower_limit, self.redshift_upper_limit, 100)
        galaxy_distribution = [
            np.sum(self.evaluate_galaxy_distribution(redshift)) for redshift in redshifts
        ]
        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.plot(redshifts, np.array(galaxy_distribution), label="Galaxy distribution")
        _plotter.show_and_close()

    def plot_galaxy_catalog_mass_distribution(self) -> None:
        masses = np.geomspace(self.lower_mass_limit, self.upper_mass_limit, 10000)
        galaxy_mass_distribution = [
            np.sum(self.evaluate_galaxy_mass_distribution(mass)) for mass in masses
        ]
        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.plot(masses, np.array(galaxy_mass_distribution), label="Galaxy mass distribution")
        _plotter.axis.set_xscale("log")
        _plotter.show_and_close()


@dataclass
class EMRIDetection:
    measured_luminosity_distance: float
    measured_redshifted_mass: float
    measured_right_ascension: float
    measured_declination: float

    @classmethod
    def from_host_galaxy(
        cls, host_galaxy: Galaxy, use_measurement_noise: bool = True
    ) -> "EMRIDetection":
        if not use_measurement_noise:
            measured_luminosity_distance = dist(host_galaxy.redshift, TRUE_HUBBLE_CONSTANT)
            measured_redshifted_mass = redshifted_mass(
                mass=host_galaxy.central_black_hole_mass,
                redshift=host_galaxy.redshift,
            )
        else:
            measured_luminosity_distance = np.random.normal(
                loc=dist(host_galaxy.redshift, TRUE_HUBBLE_CONSTANT),
                scale=FRACTIONAL_LUMINOSITY_ERROR
                * dist(
                    host_galaxy.redshift,
                    TRUE_HUBBLE_CONSTANT,
                ),
            )
            measured_redshifted_mass = np.random.normal(
                loc=redshifted_mass(
                    mass=host_galaxy.central_black_hole_mass,
                    redshift=host_galaxy.redshift,
                ),
                scale=FRACTIONAL_MEASURED_MASS_ERROR
                * redshifted_mass(
                    mass=host_galaxy.central_black_hole_mass,
                    redshift=host_galaxy.redshift,
                ),
            )

        return cls(
            measured_luminosity_distance=measured_luminosity_distance,
            measured_redshifted_mass=measured_redshifted_mass,
            measured_right_ascension=host_galaxy.right_ascension,
            measured_declination=host_galaxy.declination,
        )

    @classmethod
    def plot_detection_distribution(cls, host_galaxies: list[Galaxy]) -> None:
        detection_distribution = [
            NormalDist(mu=galaxy.redshift, sigma=FRACTIONAL_LUMINOSITY_ERROR * galaxy.redshift)
            for galaxy in host_galaxies
        ]
        redshifts = np.linspace(
            GalaxyCatalog.redshift_lower_limit, GalaxyCatalog.redshift_upper_limit, 1000
        )
        detection_probabilities = [
            np.sum([distribution.pdf(redshift) for distribution in detection_distribution])
            for redshift in redshifts
        ]
        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.plot(redshifts, np.array(detection_probabilities))
        _plotter.show_and_close()

    @classmethod
    def plot_detection_sky_distribution(cls, host_galaxies: list[Galaxy]) -> None:
        right_ascensions = [galaxy.right_ascension for galaxy in host_galaxies]
        declinations = [galaxy.declination for galaxy in host_galaxies]

        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.scatter(np.array(right_ascensions), np.array(declinations), "o")
        _plotter.show_and_close()


@dataclass
class BayesianInference:
    galaxy_catalog: GalaxyCatalog
    emri_detections: list[EMRIDetection]

    luminosity_distance_threshold = 1.55  # Gpc
    number_of_redshift_steps = 1000
    redshift_values: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    galaxy_distribution_at_redshifts: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )
    galaxy_detection_mass_distribution_at_redshifts: list = field(default_factory=list)
    detection_skylocalization_weight_by_galaxy: list = field(default_factory=list)
    use_bh_mass = False
    use_selection_effects_correction = True

    def __post_init__(self) -> None:
        self.redshift_values = np.linspace(
            self.galaxy_catalog.redshift_lower_limit,
            self.galaxy_catalog.redshift_upper_limit,
            self.number_of_redshift_steps,
        )
        self.galaxy_distribution_at_redshifts = np.array(
            [
                self.galaxy_catalog.evaluate_galaxy_distribution(redshift)
                for redshift in self.redshift_values
            ]
        )
        self.comoving_volume_at_redshifts = np.array(
            [self.galaxy_catalog.comoving_volume(redshift) for redshift in self.redshift_values]
        )
        self.galaxy_detection_mass_distribution_at_redshifts = [
            np.array(
                [
                    self.galaxy_catalog.evaluate_galaxy_mass_distribution(
                        redshifted_mass_inverse(
                            redshifted_mass=emri_detection.measured_redshifted_mass,
                            redshift=redshift,
                        )
                    )
                    for redshift in self.redshift_values
                ]
            )
            for emri_detection in self.emri_detections
        ]

        self.detection_skylocalization_weight_by_galaxy = [
            np.array(
                [
                    NormalDist(
                        mu=emri_detection.measured_right_ascension,
                        sigma=SKY_LOCALIZATION_ERROR,
                    ).pdf(galaxy.right_ascension)
                    * NormalDist(
                        mu=emri_detection.measured_declination,
                        sigma=SKY_LOCALIZATION_ERROR,
                    ).pdf(galaxy.declination)
                    for galaxy in self.galaxy_catalog.catalog
                ]
            )
            for emri_detection in self.emri_detections
        ]

    def gw_detection_probability(self, redshift: float, hubble_constant: float) -> float:
        return float(
            (
                1
                + erf(
                    (self.luminosity_distance_threshold - dist(redshift, hubble_constant))
                    / np.sqrt(2)
                    / FRACTIONAL_LUMINOSITY_ERROR
                    / dist(redshift, hubble_constant)
                )
            )
            / 2
        )

    def gw_likelihood(
        self,
        measured_luminosity_distance: float,
        redshift: float,
        hubble_constant: float,
    ) -> float:
        # TODO: Should I use truncated normal dists here?
        mu_luminosity_distance = dist(redshift, hubble_constant)
        sigma_luminosity_distance = FRACTIONAL_LUMINOSITY_ERROR * mu_luminosity_distance

        distribution = NormalDist(mu=mu_luminosity_distance, sigma=sigma_luminosity_distance)

        luminosity_distance_lower_limit = dist(GalaxyCatalog.redshift_lower_limit, hubble_constant)
        luminosity_distance_upper_limit = dist(GalaxyCatalog.redshift_upper_limit, hubble_constant)

        """return (
            distribution.pdf(measured_luminosity_distance)
            / distribution.stdev
            / (
                distribution.cdf(luminosity_distance_upper_limit)
                - distribution.cdf(luminosity_distance_lower_limit)
            )
        )"""
        return distribution.pdf(measured_luminosity_distance)

    def likelihood(
        self,
        hubble_constant: float,
        measured_luminosity_distance: float,
        measured_redshifted_mass: float,
        detection_index: int,
    ) -> float:
        # Compute all luminosity distances at once — replaces 2000+ scalar dist() calls.
        mu_d = dist_array(self.redshift_values, hubble_constant)

        # GW detection probability: P_det(z) = (1 + erf(x)) / 2
        # where x = (D_threshold - mu_d) / (sqrt(2) * sigma_d)
        p_det_array = (
            1.0
            + erf(
                (self.luminosity_distance_threshold - mu_d)
                / (np.sqrt(2.0) * FRACTIONAL_LUMINOSITY_ERROR * mu_d)
            )
        ) / 2.0

        # GW likelihood: Gaussian PDF with mu=mu_d, sigma=sigma_d
        sigma_d = FRACTIONAL_LUMINOSITY_ERROR * mu_d
        gw_likelihood_array = np.exp(
            -0.5 * ((measured_luminosity_distance - mu_d) / sigma_d) ** 2
        ) / (sigma_d * np.sqrt(2.0 * np.pi))

        # Galaxy sky-localisation weight per redshift bin: matrix-vector product
        # galaxy_distribution_at_redshifts: (n_z, n_galaxies); weights: (n_galaxies,)
        galaxy_skylocalization_weights = (
            self.galaxy_distribution_at_redshifts
            @ self.detection_skylocalization_weight_by_galaxy[detection_index]
        )

        if not self.use_bh_mass:
            nominator = np.trapezoid(
                gw_likelihood_array * p_det_array * galaxy_skylocalization_weights,
                self.redshift_values,
            )
        else:
            nominator = np.trapezoid(
                gw_likelihood_array
                * p_det_array
                * NormalDist(
                    mu=measured_redshifted_mass,
                    sigma=FRACTIONAL_MEASURED_MASS_ERROR * measured_redshifted_mass,
                ).pdf(measured_redshifted_mass)
                * np.array(
                    [
                        np.sum(
                            const_redshift_distribution
                            * const_redshift_mass_distribution
                            * self.detection_skylocalization_weight_by_galaxy[detection_index]
                        )
                        for const_redshift_distribution, const_redshift_mass_distribution in zip(
                            self.galaxy_distribution_at_redshifts,
                            self.galaxy_detection_mass_distribution_at_redshifts[detection_index],
                        )
                    ]
                ),
                self.redshift_values,
            )

        denominator = np.trapezoid(
            p_det_array * galaxy_skylocalization_weights,
            self.redshift_values,
        )
        if not self.use_selection_effects_correction:
            denominator = 1.0
        return float(nominator / denominator)

    def posterior(self, hubble_constant: float) -> list[float]:
        return [
            self.likelihood(
                hubble_constant=hubble_constant,
                measured_luminosity_distance=emri_detection.measured_luminosity_distance,
                measured_redshifted_mass=emri_detection.measured_redshifted_mass,
                detection_index=index,
            )
            for index, emri_detection in enumerate(self.emri_detections)
        ]

    def plot_gw_detection_probability(self) -> None:
        gw_detection_probabilities = [
            self.gw_detection_probability(redshift, TRUE_HUBBLE_CONSTANT)
            for redshift in self.redshift_values
        ]
        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.plot(self.redshift_values, np.array(gw_detection_probabilities))
        _plotter.show_and_close()


if __name__ == "__main__":
    galaxy_catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=True)
    NUMBER_OF_GALAXIES = 1000
    STEPS = 20
    NUMBER_OF_NEW_DETECTIONS_PER_STEP = 3
    compare_with_truncnorm = False

    plotter = ScientificPlotter(figure_size=(16, 9))
    plotter.figure.suptitle(
        f"Galaxy Catalog: {NUMBER_OF_GALAXIES}, Detections: {NUMBER_OF_NEW_DETECTIONS_PER_STEP * STEPS}"
    )
    plotter.set_colormap_from_range((0, STEPS * NUMBER_OF_NEW_DETECTIONS_PER_STEP - 1))

    # fractional_luminosity_errors = np.linspace(0.05, 0.3, STEPS)

    for i in range(STEPS):
        start_time = time()

        # FRACTIONAL_LUMINOSITY_ERROR = fractional_luminosity_errors[i]
        # galaxy_catalog.setup_galaxy_distribution()

        while True:
            galaxy_catalog.remove_all_galaxies()
            galaxy_catalog.create_random_catalog(NUMBER_OF_GALAXIES)
            if (
                len(galaxy_catalog.get_possible_host_galaxies())
                > STEPS * NUMBER_OF_NEW_DETECTIONS_PER_STEP
            ):
                print(
                    f"Galaxy catalog set up with {len(galaxy_catalog.catalog)} galaxies with {len(galaxy_catalog.get_possible_host_galaxies())} possible host galaxies."
                )
                break

        host_galaxies = galaxy_catalog.get_unique_host_galaxies_from_catalog(
            number_of_host_galaxies=NUMBER_OF_NEW_DETECTIONS_PER_STEP * STEPS,
        )

        emri_detections = [
            EMRIDetection.from_host_galaxy(host_galaxy) for host_galaxy in host_galaxies
        ]
        bayesian_inference = BayesianInference(
            galaxy_catalog=galaxy_catalog, emri_detections=emri_detections
        )

        # Inference
        hubble_values = np.linspace(0.6, 0.8, 60)
        with mp.Pool() as pool:
            posterior_distribution = pool.map(bayesian_inference.posterior, hubble_values)

        likelihoods = np.array(posterior_distribution).T
        # plot new individual likelihood

        """
        for k in range(NUMBER_OF_NEW_DETECTIONS_PER_STEP):
            index = -(k + 1)
            plotter.plot_colored(
                hubble_values,
                likelihoods[index] / max(likelihoods[index]),
                color=i * NUMBER_OF_NEW_DETECTIONS_PER_STEP + k,
                line_style="dashed",
                kwargs={"linewidth": 0.5},
            )
        """

        # plot combined likelihood
        combined_posterior = np.prod(likelihoods, axis=0)
        plotter.plot_colored(
            hubble_values,
            combined_posterior / max(combined_posterior),
            color=i * NUMBER_OF_NEW_DETECTIONS_PER_STEP,
            line_style="solid",
            kwargs={"linewidth": 1},
        )

        # evaluate with bh mass information
        bayesian_inference.use_bh_mass = True
        with mp.Pool() as pool:
            posterior_distribution = pool.map(bayesian_inference.posterior, hubble_values)

        likelihoods = np.array(posterior_distribution).T
        likelihoods = likelihoods / np.max(likelihoods)

        # plot combined posterior
        combined_posterior = np.prod(likelihoods, axis=0)

        plotter.plot_colored(
            hubble_values,
            combined_posterior / np.max(combined_posterior),
            color=i * NUMBER_OF_NEW_DETECTIONS_PER_STEP,
            line_style="dotted",
            label=rf"iteration ${i}$",
            kwargs={"linewidth": 1.5},
        )

        if compare_with_truncnorm:
            galaxy_catalog_with_truncnorm = galaxy_catalog
            galaxy_catalog_with_truncnorm._use_truncnorm = True
            bayesian_inference_with_truncnorm = BayesianInference(
                galaxy_catalog=galaxy_catalog_with_truncnorm,
                emri_detections=emri_detections,
            )
            with mp.Pool() as pool:
                posterior_distribution_with_truncnorm = pool.map(
                    bayesian_inference_with_truncnorm.posterior, hubble_values
                )
            likelihoods_with_truncnorm = np.array(posterior_distribution_with_truncnorm).T
            combined_posterior_with_truncnorm = np.prod(likelihoods_with_truncnorm, axis=0)
            plotter.plot_colored(
                hubble_values,
                combined_posterior_with_truncnorm / max(combined_posterior_with_truncnorm),
                color=i * NUMBER_OF_NEW_DETECTIONS_PER_STEP,
                line_style="dashdot",
            )
        print(f"Finished iteration {i + 1} of {STEPS} in {time() - start_time:.2f}s.")

    plt.vlines(
        TRUE_HUBBLE_CONSTANT,
        0,
        1,
        color="black",
        linestyles="dashed",
        label="True Hubble Constant",
    )
    plotter.show_colorbar(label="Number of detections")
    plotter.show_and_close()
