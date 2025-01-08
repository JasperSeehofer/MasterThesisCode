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
from scipy.optimize import fsolve
from scipy.special import hyp2f1
import emcee

from scientific_plotter import ScientificPlotter

FRACTIONAL_LUMINOSITY_ERROR = 0.1
FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR = 0.1
FRACTIONAL_MEASURED_MASS_ERROR = 1e-8 # TODO: Check with parameter estimation
SKY_LOCALIZATION_ERROR = 5/180*np.pi  # in radians
TRUE_HUBBLE_CONSTANT = 0.7  # km/s/Mpc/100
SPEED_OF_LIGHT = 300000.0  # km/s
OMEGA_M = 0.25
OMEGA_LAMBDA = 0.75
W_0 = -1.0
W_A = 0.0
GPC_TO_MPC = float(1e3)
KM_TO_M = float(1e3)

def dist(
    redshift: float,
    h: float = TRUE_HUBBLE_CONSTANT,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_LAMBDA,
    w_0: float = W_0,
    w_a: float = W_A,
    offset_for_root_finding: float = 0.0,
) -> float:
    """
    Calculate the luminosity distance in Gpc.
    """
    if not (isinstance(redshift, float) or isinstance(redshift, int)):
        redshift = redshift[0]

    H_0 = h * 100.0  # Hubble constant in m/s*Mpc

    # Hubble parameter
    """
    zs = np.linspace(0, redshift, 1000)
    hubble = np.sqrt(
        Omega_m * (1 + zs) ** 3
        + Omega_de
        * (1 + zs) ** (3 * (1 + w_0 + w_a))
        * np.exp(-3 * w_a * zs / (1 + zs))
    )

    # integral
    integral = np.trapz(1 / hubble, zs)
    """
    # use analytic version of the integral
    integral = lambda_cdm_analytic_distance(redshift, Omega_m, Omega_de)

    # luminosity distance in Gpc
    result = SPEED_OF_LIGHT / H_0 * (1 + redshift) * integral - offset_for_root_finding

    return result

def lambda_cdm_analytic_distance(
    redshift: float, Omega_m: float = OMEGA_M, Omega_de: float = OMEGA_LAMBDA
) -> float:
    return (
        (1 + redshift)
        * np.sqrt(1 + (Omega_m * (1 + redshift) ** 3) / Omega_de)
        * hyp2f1(1 / 3, 1 / 2, 4 / 3, -((Omega_m * (1 + redshift) ** 3) / Omega_de))
    ) / np.sqrt(Omega_de + Omega_m * (1 + redshift) ** 3) - (
        np.sqrt((Omega_m + Omega_de) / Omega_de)
        * hyp2f1(1 / 3, 1 / 2, 4 / 3, -(Omega_m / Omega_de))
    ) / np.sqrt(
        Omega_m + Omega_de
    )

def dist_to_redshift(
    distance: float,
    h: float = TRUE_HUBBLE_CONSTANT,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_LAMBDA,
    w_0: float = W_0,
    w_a: float = W_A,
) -> float:
    """
    Calculate the redshift for a given luminosity distance.
    """
    return fsolve(
        dist,
        1,
        args=(
            h,
            Omega_m,
            Omega_de,
            w_0,
            w_a,
            distance,
        ),
    )[0]


def redshifted_mass(mass: float, redshift: float) -> float:
    return mass * (1 + redshift)


def redshifted_mass_inverse(redshifted_mass: float, redshift: float) -> float:
    return redshifted_mass / (1 + redshift)


@dataclass
class Galaxy:
    redshift: float
    central_black_hole_mass: float  # same as massive black hole
    right_ascension: float 
    declination: float

    @classmethod
    def with_random_skylocalization(cls, redshift: float, central_black_hole_mass: float) -> Galaxy:
        # get spherically uniform distributed sky localization
        right_ascension = np.random.uniform(0, 2*np.pi)
        declination = np.arccos(np.random.uniform(-1, 1))
        return cls(
            redshift=redshift,
            central_black_hole_mass=central_black_hole_mass,
            right_ascension=right_ascension,
            declination=declination,
        )

    @property
    def redshift_uncertainty(self) -> float:
        return min(0.013 * (1 + self.redshift) ** 3, 0.015)


class GalaxyCatalog:
    _use_truncnorm: bool
    _use_comoving_volume: bool

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
            redshift_samples = self.get_samples_from_comoving_volume(number_of_galaxies)[:number_of_galaxies]
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
        # TODO: also add to mass distribution
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
        
        redshift_uncertainty = min(0.013*(1+redshift)**3, 0.015)
        p_background = 1

        if self._use_comoving_volume:
            p_background = self.comoving_volume(redshift)

        normal_dist = NormalDist(mu=redshift, sigma=redshift_uncertainty)

        if self._use_truncnorm:
            normalization = (normal_dist.cdf(self.redshift_upper_limit) - normal_dist.cdf(self.redshift_lower_limit))*normal_dist.stdev
            return np.array([
                normal_dist.pdf(galaxy.redshift)
                / normalization
                # Adjust for background
                for galaxy in self.catalog
            ]) / len(self.catalog)

        return np.array([
            normal_dist.pdf(galaxy.redshift)
            # Adjust for background
            for galaxy in self.catalog
        ]) / len(self.catalog)
        

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
                sigma=FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR
                * 10**5.5,
            )
            for galaxy in self.catalog
        ]

    def evaluate_galaxy_mass_distribution(self, mass: float) -> float:
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
        return np.array([
                distribution.pdf(mass)
                for distribution in self.galaxy_mass_distribution
            ]) / len(self.catalog)

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
            np.sum(self.evaluate_galaxy_distribution(redshift)) for redshift in redshifts
        ]
        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.plot(redshifts, galaxy_distribution, label="Galaxy distribution")
        _plotter.show_and_close()

    def plot_galaxy_catalog_mass_distribution(self) -> None:
        masses = np.geomspace(self.lower_mass_limit, self.upper_mass_limit, 10000)
        galaxy_mass_distribution = [
            np.sum(self.evaluate_galaxy_mass_distribution(mass)) for mass in masses
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
    measured_right_ascension: float
    measured_declination: float

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
            measured_right_ascension=host_galaxy.right_ascension,
            measured_declination=host_galaxy.declination,
        )
    
    @classmethod
    def plot_detection_distribution(cls, host_galaxies: List[Galaxy]) -> None:
        detection_distribution = [
            NormalDist(mu=galaxy.redshift, sigma=FRACTIONAL_LUMINOSITY_ERROR*galaxy.redshift)
            for galaxy in host_galaxies
        ]
        redshifts = np.linspace(GalaxyCatalog.redshift_lower_limit, GalaxyCatalog.redshift_upper_limit, 1000)
        detection_probabilities = [
            np.sum([distribution.pdf(redshift) for distribution in detection_distribution])
            for redshift in redshifts
        ]
        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.plot(redshifts, detection_probabilities)
        _plotter.show_and_close()

    @classmethod
    def plot_detection_sky_distribution(cls, host_galaxies: List[Galaxy]) -> None:
        right_ascensions = [galaxy.right_ascension for galaxy in host_galaxies]
        declinations = [galaxy.declination for galaxy in host_galaxies]

        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.scatter(right_ascensions, declinations, "o")
        _plotter.show_and_close()


@dataclass
class BayesianInference:
    galaxy_catalog: GalaxyCatalog
    emri_detections: List[EMRIDetection]

    luminosity_distance_threshold = 1550.0  # Mpc
    number_of_redshift_steps = 1000
    redshift_values: npt.ArrayLike = np.array([])
    galaxy_distribution_at_redshifts: npt.ArrayLike = np.array([])
    galaxy_detection_mass_distribution_at_redshifts: list = field(default_factory=list)
    detection_skylocalization_weight_by_galaxy: list = field(default_factory=list)
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
        
        self.detection_skylocalization_weight_by_galaxy = [
            np.array([
                NormalDist(
                mu=emri_detection.measured_right_ascension,
                sigma=SKY_LOCALIZATION_ERROR,
                ).pdf(galaxy.right_ascension) * 
                NormalDist(
                    mu=emri_detection.measured_declination,
                    sigma=SKY_LOCALIZATION_ERROR,
                ).pdf(galaxy.declination)
                for galaxy in self.galaxy_catalog.catalog
        ])
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
        measured_luminosity_distance: float,
        redshift: float,
        hubble_constant: float,
    ) -> float:
        # TODO: Should I use truncated normal dists here?
        mu_luminosity_distance = dist(redshift, hubble_constant)
        sigma_luminosity_distance = FRACTIONAL_LUMINOSITY_ERROR * dist(redshift, hubble_constant)

        distribution = NormalDist(
            mu=mu_luminosity_distance, sigma=sigma_luminosity_distance
        )

        luminosity_distance_lower_limit = dist(
            GalaxyCatalog.redshift_lower_limit, hubble_constant
        )
        luminosity_distance_upper_limit = dist(
            GalaxyCatalog.redshift_upper_limit, hubble_constant
        )

        return distribution.pdf(measured_luminosity_distance)/distribution.stdev/(
            distribution.cdf(luminosity_distance_upper_limit)
            - distribution.cdf(luminosity_distance_lower_limit)
        )
        

    def likelihood(
        self,
        hubble_constant: float,
        measured_luminosity_distance: float,
        measured_redshifted_mass: float,
        detection_index: int,
    ) -> float:
        if not self.use_bh_mass:
            nominator = np.trapz(
                np.array(
                    [
                        self.gw_likelihood(
                            hubble_constant=hubble_constant,
                            measured_luminosity_distance=measured_luminosity_distance,
                            redshift=redshift,
                        )
                        for redshift in self.redshift_values
                    ]
                )
                * np.array(
                    [
                        np.sum(
                            const_redshift
                            * self.detection_skylocalization_weight_by_galaxy[detection_index]
                        )
                        for const_redshift in self.galaxy_distribution_at_redshifts
                    ]
                ),
                self.redshift_values,
            )
                
        else:
            nominator = np.trapz(
                np.array(
                    [
                        self.gw_likelihood(
                            hubble_constant=hubble_constant,
                            measured_luminosity_distance=measured_luminosity_distance,
                            redshift=redshift,
                        )
                        for redshift in self.redshift_values
                    ]
                )
                * NormalDist(mu=measured_redshifted_mass, sigma=FRACTIONAL_MEASURED_MASS_ERROR*measured_redshifted_mass).pdf(
                    measured_redshifted_mass
                )
                * np.array(
                    [
                        np.sum(
                            const_redshift_distribution
                            * const_redshift_mass_distribution
                            * self.detection_skylocalization_weight_by_galaxy[detection_index]
                        )
                        for const_redshift_distribution, const_redshift_mass_distribution in zip(
                            self.galaxy_distribution_at_redshifts, 
                            self.galaxy_detection_mass_distribution_at_redshifts[detection_index])
                    ]
                ),
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
            * np.sum(
                self.galaxy_distribution_at_redshifts,
                axis=1,
            ),
            self.redshift_values,
        )
        return nominator / denominator

    def posterior(self, hubble_constant: float) -> List[float]:
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
        _plotter.plot(self.redshift_values, gw_detection_probabilities)
        _plotter.show_and_close()


if __name__ == "__main__":
    galaxy_catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False)
    NUMBER_OF_GALAXIES = 50
    STEPS = 5
    NUMBER_OF_NEW_DETECTIONS_PER_STEP = 2
    compare_with_truncnorm = False

    plotter = ScientificPlotter(figure_size=(16, 9))
    plotter.figure.suptitle(
        f"Galaxy Catalog: {NUMBER_OF_GALAXIES}, Detections: {NUMBER_OF_NEW_DETECTIONS_PER_STEP*STEPS}"
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
