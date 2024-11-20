from __future__ import annotations

from typing import List, Union
import numpy as np
# import error funciton
from scipy.special import erf
from statistics import NormalDist
from scipy.stats import truncnorm
from scipy.stats.distributions import truncnorm_gen
from dataclasses import dataclass
import matplotlib.pyplot as plt
from time import time

from scientific_plotter import ScientificPlotter

FRACTIONAL_LUMINOSITY_ERROR = 0.01
TRUE_HUBBLE_CONSTANT = 0.7 # km/s/Mpc/100
SPEED_OF_LIGHT = 300000.0 # km/s

def dist(redshift: float, hubble_constant: float) -> float:
    return redshift * SPEED_OF_LIGHT / (hubble_constant * 100) # Mpc


@dataclass
class Galaxy:
    redshift: float
    central_black_hole_mass: float # same as massive black hole

    @property
    def redshift_uncertainty(self) -> float:
        # min(0.013 * (1 + self.redshift)**3, 0.015)
        return 0.013 * (1 + self.redshift)**3


class GalaxyCatalog:
    _use_truncnorm: bool

    lower_mass_limit: float = 10**(4)
    upper_mass_limit: float = 10**(7)
    redshift_lower_limit: float = 0.01
    redshift_upper_limit: float = 1.0
    catalog: List[Galaxy] = []
    galaxy_distribution: List[Union[truncnorm_gen, NormalDist]] = []
    

    def __init__(self, use_truncnorm: bool = True):
        self._use_truncnorm = use_truncnorm

    def create_random_catalog(self, number_of_galaxies: int) -> None:
        for i in range(number_of_galaxies):
            self.catalog.append(
                Galaxy(
                    redshift=np.random.uniform(
                        self.redshift_lower_limit, 
                        self.redshift_upper_limit
                        ), 
                    central_black_hole_mass=np.random.uniform(
                        self.lower_mass_limit, 
                        self.upper_mass_limit
                        )
                    )
                )
        self.setup_galaxy_distribution()

    def remove_all_galaxies(self) -> None:
        self.catalog = []
        self.galaxy_distribution = []

    def add_random_galaxy(self) -> None:
        galaxy = Galaxy(
            redshift=np.random.uniform(
                self.redshift_lower_limit, 
                self.redshift_upper_limit
                ), 
            central_black_hole_mass=np.random.uniform(
                self.lower_mass_limit, 
                self.upper_mass_limit
                )
        )
        self.catalog.append(galaxy)
        self.append_galaxy_to_galaxy_distribution(galaxy)
    
    def add_host_galaxy(self) -> Galaxy:
        galaxy = Galaxy(
            redshift=np.random.uniform(
                self.redshift_lower_limit, 
                self.redshift_upper_limit
                ), 
            central_black_hole_mass=np.random.uniform(
                self.lower_mass_limit, 
                self.upper_mass_limit
                )
        )
        self.catalog.append(galaxy)
        self.append_galaxy_to_galaxy_distribution(galaxy)

    def setup_galaxy_distribution(self) -> float:
        if not self._use_truncnorm:
            self.galaxy_distribution = [
                    NormalDist(mu=galaxy.redshift, sigma=galaxy.redshift_uncertainty)
                    for galaxy in self.catalog
            ]
        else:
            self.galaxy_distribution = [
                    truncnorm(
                        a=(self.redshift_lower_limit - galaxy.redshift)/galaxy.redshift_uncertainty,
                        b=(self.redshift_upper_limit - galaxy.redshift)/galaxy.redshift_uncertainty
                    )
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
                    a=(self.redshift_lower_limit - galaxy.redshift)/galaxy.redshift_uncertainty,
                    b=(self.redshift_upper_limit - galaxy.redshift)/galaxy.redshift_uncertainty
                )
            )

    def evaluate_galaxy_distribution(self, redshift: float) -> float:
        if not self._use_truncnorm:
            return np.sum(
                [
                    distribution.pdf(redshift)
                    for distribution in self.galaxy_distribution
                ]
            )

@dataclass
class EMRIDetection:
    meassured_luminosity_distance: float

@dataclass
class BayesianInference:
    galaxy_catalog: GalaxyCatalog
    emri_detections: List[EMRIDetection]

    luminosity_distance_threshold = 1550.0 # Mpc


    def gw_detection_probability(self, redshift: float, hubble_constant: float) -> float:
        return (1 + erf((self.luminosity_distance_threshold - dist(redshift, hubble_constant)) / np.sqrt(2)*FRACTIONAL_LUMINOSITY_ERROR*dist(redshift, hubble_constant))) / 2
    
    def gw_likelihood(self,  meassured_luminosity_distance: float, redshift: float, hubble_constant: float) -> float:
        mu = dist(redshift, hubble_constant)
        sigma = FRACTIONAL_LUMINOSITY_ERROR*dist(redshift, hubble_constant)
        return NormalDist(mu=mu, sigma=sigma).pdf(meassured_luminosity_distance)
    
    def likelihood(self, hubble_constant: float, meassured_luminosity_distance: float) -> float:
        redshifts = np.linspace(
            self.galaxy_catalog.redshift_lower_limit, 
            self.galaxy_catalog.redshift_upper_limit, 
            1000
        )
        nominator = np.trapz(
            [
                self.gw_likelihood(
                    hubble_constant=hubble_constant, meassured_luminosity_distance=meassured_luminosity_distance, redshift=redshift
                ) * self.galaxy_catalog.evaluate_galaxy_distribution(redshift) for redshift in redshifts],
            redshifts
        )
        denominator = np.trapz(
            [
                self.gw_detection_probability(
                    hubble_constant=hubble_constant,
                    redshift=redshift
                ) * self.galaxy_catalog.evaluate_galaxy_distribution(redshift)
                for redshift in redshifts],
            redshifts
        )
        return nominator / denominator
    

if __name__ == "__main__":
    galaxy_catalog = GalaxyCatalog(use_truncnorm=False)
    plotter = ScientificPlotter(figure_size=(16, 9))
    plotter.set_colormap_from_range((galaxy_catalog.redshift_lower_limit, galaxy_catalog.redshift_upper_limit))

    for i in range(5):
        galaxy_catalog.remove_all_galaxies()
        galaxy_catalog.create_random_catalog(1)

        print("Galaxy catalog set up...")

        # check if no galaxy in catalog can be host galaxy
        if not any([dist(galaxy.redshift, TRUE_HUBBLE_CONSTANT) <= BayesianInference.luminosity_distance_threshold for galaxy in galaxy_catalog.catalog]):
            print("no possible host found, continue")
            continue

        emri_detection = EMRIDetection(meassured_luminosity_distance=dist(galaxy_catalog.catalog[0].redshift, TRUE_HUBBLE_CONSTANT))

        bayesian_inference = BayesianInference(galaxy_catalog=galaxy_catalog, emri_detections=[emri_detection])

        # Inference
        hubble_values = np.linspace(0.6, 0.8, 200)
        likelihoods = [
            bayesian_inference.likelihood(hubble_constant=hubble, meassured_luminosity_distance=emri_detection.meassured_luminosity_distance)
            for hubble in hubble_values
        ]

        plotter.plot_colored(hubble_values, likelihoods / max(likelihoods), color=galaxy_catalog.catalog[0].redshift)
        print(f"evaluated step {i+1}")
    plt.vlines(TRUE_HUBBLE_CONSTANT, 0, 1, color='black', linestyles="dashed", label='True Hubble Constant')
    plotter.show_colorbar(label='Redshift')
    plotter.show_and_close()



    