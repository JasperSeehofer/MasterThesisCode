from __future__ import annotations

from typing import List, Union
import numpy as np
import numpy.typing as npt
import multiprocessing as mp

# import error funciton
from scipy.special import erf
from statistics import NormalDist
from scipy.stats import truncnorm
from scipy.stats.distributions import truncnorm_gen
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from time import time
import emcee

from scientific_plotter import ScientificPlotter

FRACTIONAL_LUMINOSITY_ERROR = 0.05
FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR = 0.1
TRUE_HUBBLE_CONSTANT = 0.7  # km/s/Mpc/100
SPEED_OF_LIGHT = 300000.0  # km/s
OMEGA_M = 0.25
OMEGA_LAMBDA = 0.75


def dist(redshift: float, hubble_constant: float) -> float:
    return redshift * SPEED_OF_LIGHT / (hubble_constant * 100)  # Mpc


def dist_to_redshift(luminosity_distance: float, hubble_constant: float) -> float:
    return luminosity_distance * (hubble_constant * 100) / SPEED_OF_LIGHT


def redshifted_mass(mass: float, redshift: float) -> float:
    return mass * (1 + redshift)


def redshifted_mass_inverse(redshifted_mass: float, redshift: float) -> float:
    return redshifted_mass / (1 + redshift)


@dataclass
class Galaxy:
    redshift: float
    central_black_hole_mass: float  # same as massive black hole

    @property
    def redshift_uncertainty(self) -> float:
        return min(0.013 * (1 + self.redshift) ** 3, 0.015)


class GalaxyCatalog:
    _use_truncnorm: bool

    lower_mass_limit: float = 10 ** (4)
    upper_mass_limit: float = 10 ** (7)
    redshift_lower_limit: float = 0.00001
    redshift_upper_limit: float = 0.55
    catalog: List[Galaxy] = []
    galaxy_distribution: List[NormalDist] = []

    def __init__(self, use_truncnorm: bool = True, use_comoving_volume: bool = True):
        self._use_truncnorm = use_truncnorm
        self._use_comoving_volume = use_comoving_volume

    @property
    def mean_redshift(self) -> float:
        return np.mean([galaxy.redshift for galaxy in self.catalog])

    def comoving_volume(self, redshift: float) -> float:
        redshifts = np.linspace(0, redshift, 100)
        integral = np.trapz(
            [(1 / np.sqrt(OMEGA_M * (1 + z) ** 3 + OMEGA_LAMBDA)) for z in redshifts],
            redshifts,
        )
        res = 4 * np.pi * (SPEED_OF_LIGHT / TRUE_HUBBLE_CONSTANT) ** 3 * integral**2
        return res

    def log_comoving_volume(self, redshift: float) -> float:
        try:
            redshift = redshift[0]
        except TypeError:
            pass
        if redshift < self.redshift_lower_limit or redshift > self.redshift_upper_limit:
            return -np.inf
        return np.log(self.comoving_volume(redshift))

    def get_samples_from_comoving_volume(self, number_of_samples: int) -> List[float]:
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
        samples = sampler.get_chain(flat=True)
        # return required number of samples by using the last n samples
        reduced_samples = samples[-number_of_samples:]
        return samples

    def plot_comoving_volume_sampling(self) -> None:
        samples = self.get_samples_from_comoving_volume(1000)
        plt.hist(samples, bins=20)
        plt.show()
        plt.close()

    def plot_comoving_volume(self) -> None:
        redshifts = np.linspace(
            self.redshift_lower_limit, self.redshift_upper_limit, 100
        )
        comoving_volumes = [self.comoving_volume(redshift) for redshift in redshifts]
        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.plot(redshifts, comoving_volumes, label="Comoving volume")
        _plotter.show_and_close()

    def create_random_catalog(self, number_of_galaxies: int) -> None:
        # draw mass from uniform in log space
        if self._use_comoving_volume:
            redshift_samples = self.get_samples_from_comoving_volume(number_of_galaxies)
            for redshift in redshift_samples:
                self.catalog.append(
                    Galaxy(
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
                    Galaxy(
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

    def add_random_galaxy(self) -> None:
        galaxy = Galaxy(
            redshift=np.random.uniform(
                self.redshift_lower_limit, self.redshift_upper_limit
            ),
            central_black_hole_mass=np.random.uniform(
                self.lower_mass_limit, self.upper_mass_limit
            ),
        )
        self.catalog.append(galaxy)
        self.append_galaxy_to_galaxy_distribution(galaxy)

    def add_host_galaxy(self) -> Galaxy:
        galaxy = Galaxy(
            redshift=np.random.uniform(
                self.redshift_lower_limit, self.redshift_upper_limit
            ),
            central_black_hole_mass=np.random.uniform(
                self.lower_mass_limit, self.upper_mass_limit
            ),
        )
        self.catalog.append(galaxy)
        self.append_galaxy_to_galaxy_distribution(galaxy)

    def setup_galaxy_distribution(self) -> None:
        self.galaxy_distribution = [
            NormalDist(mu=galaxy.redshift, sigma=galaxy.redshift_uncertainty)
            for galaxy in self.catalog
        ]

    def append_galaxy_to_galaxy_distribution(self, galaxy: Galaxy) -> None:
        if not self._use_truncnorm:
            self.galaxy_distribution.append(
                NormalDist(galaxy.redshift, galaxy.redshift_uncertainty)
            )
        else:
            self.galaxy_distribution.append(
                truncnorm(
                    a=(self.redshift_lower_limit - galaxy.redshift)
                    / galaxy.redshift_uncertainty,
                    b=(self.redshift_upper_limit - galaxy.redshift)
                    / galaxy.redshift_uncertainty,
                )
            )

    def evaluate_galaxy_distribution(self, redshift: float) -> float:
        if self._use_truncnorm:
            return np.sum(
                [
                    distribution.pdf(redshift)
                    / (1 - distribution.cdf(self.redshift_lower_limit))
                    / distribution.stdev
                    for distribution in self.galaxy_distribution
                ]
            ) / len(self.catalog)
        return np.sum(
            [distribution.pdf(redshift) for distribution in self.galaxy_distribution]
        ) / len(self.catalog)

    def setup_galaxy_mass_distribution(self) -> None:
        self.galaxy_mass_distribution = [
            NormalDist(
                mu=galaxy.central_black_hole_mass,
                sigma=FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR
                * galaxy.central_black_hole_mass,
            )
            for galaxy in self.catalog
        ]

    def evaluate_galaxy_mass_distribution(self, mass: float) -> float:
        # use truncated normal distribution

        return np.sum(
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

    def get_possible_host_galaxies(self) -> List[Galaxy]:
        return [
            galaxy
            for galaxy in self.catalog
            if dist(galaxy.redshift, TRUE_HUBBLE_CONSTANT)
            <= BayesianInference.luminosity_distance_threshold
        ]

    def get_unique_host_galaxies_from_catalog(
        self, number_of_host_galaxies: int
    ) -> List[Galaxy]:

        if len(self.get_possible_host_galaxies()) < number_of_host_galaxies:
            print("Not enough possible host galaxies in catalog.")
            return []
        return np.random.choice(
            self.get_possible_host_galaxies(), number_of_host_galaxies, replace=False
        )

    def add_unique_host_galaxies_from_catalog(
        self, number_of_host_galaxies_to_add: int, used_host_galaxies: List[Galaxy]
    ) -> List[Galaxy]:
        if (
            len(self.get_possible_host_galaxies()) - len(used_host_galaxies)
            < number_of_host_galaxies_to_add
        ):
            print("Not enough possible host galaxies in catalog.")
            return used_host_galaxies
        new_host_galaxies = np.random.choice(
            [
                galaxy
                for galaxy in self.get_possible_host_galaxies()
                if galaxy not in used_host_galaxies
            ],
            number_of_host_galaxies_to_add,
            replace=False,
        )
        used_host_galaxies.extend(new_host_galaxies)
        return used_host_galaxies

    def plot_galaxy_catalog(self) -> None:
        redshifts = np.linspace(
            self.redshift_lower_limit, self.redshift_upper_limit, 100
        )
        galaxy_distribution = [
            self.evaluate_galaxy_distribution(redshift) for redshift in redshifts
        ]
        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.plot(redshifts, galaxy_distribution, label="Galaxy distribution")
        _plotter.show_and_close()

    def plot_galaxy_catalog_mass_distribution(self) -> None:
        masses = np.geomspace(self.lower_mass_limit, self.upper_mass_limit, 100)
        galaxy_mass_distribution = [
            self.evaluate_galaxy_mass_distribution(mass) for mass in masses
        ]
        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.plot(
            masses, galaxy_mass_distribution, label="Galaxy mass distribution"
        )
        _plotter.axis.set_xscale("log")
        _plotter.show_and_close()


@dataclass
class EMRIDetection:
    measured_luminosity_distance: float
    measured_redshifted_mass: float

    @classmethod
    def from_host_galaxy(cls, host_galaxy: Galaxy) -> EMRIDetection:
        return cls(
            measured_luminosity_distance=dist(
                host_galaxy.redshift, TRUE_HUBBLE_CONSTANT
            ),
            measured_redshifted_mass=redshifted_mass(
                mass=host_galaxy.central_black_hole_mass,
                redshift=host_galaxy.redshift,
            ),
        )


@dataclass
class BayesianInference:
    galaxy_catalog: GalaxyCatalog
    emri_detections: List[EMRIDetection]

    luminosity_distance_threshold = 1550.0  # Mpc
    number_of_redshift_steps = 1000
    redshift_values: npt.ArrayLike = np.array([])
    galaxy_distribution_at_redshifts: npt.ArrayLike = np.array([])
    galaxy_detection_mass_distribution_at_redshifts: list = field(default_factory=list)
    use_bh_mass = False

    def __post_init__(self):
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

    def gw_detection_probability(
        self, redshift: float, hubble_constant: float
    ) -> float:
        return (
            1
            + erf(
                (self.luminosity_distance_threshold - dist(redshift, hubble_constant))
                / np.sqrt(2)
                / FRACTIONAL_LUMINOSITY_ERROR
                / dist(redshift, hubble_constant)
            )
        ) / 2

    def gw_likelihood(
        self,
        meassured_luminosity_distance: float,
        redshift: float,
        hubble_constant: float,
    ) -> float:
        mu = dist(redshift, hubble_constant)
        sigma = FRACTIONAL_LUMINOSITY_ERROR * dist(redshift, hubble_constant)
        return NormalDist(mu=mu, sigma=sigma).pdf(meassured_luminosity_distance)

    def likelihood(
        self,
        hubble_constant: float,
        meassured_luminosity_distance: float,
        detection_index: int,
    ) -> float:
        if not self.use_bh_mass:
            nominator = np.trapz(
                np.array(
                    [
                        self.gw_likelihood(
                            hubble_constant=hubble_constant,
                            meassured_luminosity_distance=meassured_luminosity_distance,
                            redshift=redshift,
                        )
                        for redshift in self.redshift_values
                    ]
                )
                * self.galaxy_distribution_at_redshifts,
                self.redshift_values,
            )
        else:
            nominator = np.trapz(
                np.array(
                    [
                        self.gw_likelihood(
                            hubble_constant=hubble_constant,
                            meassured_luminosity_distance=meassured_luminosity_distance,
                            redshift=redshift,
                        )
                        for redshift in self.redshift_values
                    ]
                )
                * self.galaxy_distribution_at_redshifts
                * self.galaxy_detection_mass_distribution_at_redshifts[detection_index],
                self.redshift_values,
            )
        denominator = np.trapz(
            np.array(
                [
                    self.gw_detection_probability(
                        hubble_constant=hubble_constant, redshift=redshift
                    )
                    for redshift in self.redshift_values
                ]
            )
            * self.galaxy_distribution_at_redshifts,
            self.redshift_values,
        )
        return nominator / denominator

    def posterior(self, hubble_constant: float) -> List[float]:
        return [
            self.likelihood(
                hubble_constant=hubble_constant,
                meassured_luminosity_distance=emri_detection.measured_luminosity_distance,
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
        _plotter.plot(self.redshift_values, gw_detection_probabilities)
        _plotter.show_and_close()


if __name__ == "__main__":
    galaxy_catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=True)
    NUMBER_OF_GALAXIES = 70
    STEPS = 20
    NUMBER_OF_NEW_DETECTIONS_PER_STEP = 1
    compare_with_truncnorm = False

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

    galaxy_catalog.plot_galaxy_catalog()
    galaxy_catalog.plot_galaxy_catalog_mass_distribution()

    plotter = ScientificPlotter(figure_size=(16, 9))
    plotter.figure.suptitle(
        f"Galaxy Catalog: {NUMBER_OF_GALAXIES}, Detections: {NUMBER_OF_NEW_DETECTIONS_PER_STEP*STEPS}"
    )
    plotter.set_colormap_from_range((0, STEPS * NUMBER_OF_NEW_DETECTIONS_PER_STEP - 1))

    host_galaxies: List[Galaxy] = []

    for i in range(STEPS):
        start_time = time()
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
        hubble_values = np.linspace(0.6, 0.8, 80)
        with mp.Pool() as pool:
            posterior_distribution = pool.map(
                bayesian_inference.posterior, hubble_values
            )

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
            posterior_distribution = pool.map(
                bayesian_inference.posterior, hubble_values
            )

        likelihoods = np.array(posterior_distribution).T
        likelihoods = likelihoods / np.max(likelihoods)

        # plot combined posterior
        combined_posterior = np.prod(likelihoods, axis=0)

        plotter.plot_colored(
            hubble_values,
            combined_posterior / max(combined_posterior),
            color=i * NUMBER_OF_NEW_DETECTIONS_PER_STEP,
            line_style="dotted",
            kwargs={"linewidth": 1},
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
            likelihoods_with_truncnorm = np.array(
                posterior_distribution_with_truncnorm
            ).T
            combined_posterior_with_truncnorm = np.prod(
                likelihoods_with_truncnorm, axis=0
            )
            plotter.plot_colored(
                hubble_values,
                combined_posterior_with_truncnorm
                / max(combined_posterior_with_truncnorm),
                color=i * NUMBER_OF_NEW_DETECTIONS_PER_STEP,
                line_style="dashdot",
            )
        print(f"Finished iteration {i+1} of {STEPS} in {time()-start_time:.2f}s.")

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
