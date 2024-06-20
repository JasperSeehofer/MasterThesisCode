from enum import Enum
import pandas as pd
import numpy as np
from collections.abc import Iterable
from dataclasses import dataclass
import logging
import os
from typing import Tuple, List, Optional
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from master_thesis_code.constants import H0, C
from master_thesis_code.physical_relations import (
    dist,
    dist_to_redshift,
    dist_to_redshift_error_proagation,
    convert_redshifted_mass_to_true_mass,
)

_LOGGER = logging.getLogger()


GPC_TO_MPC = 10**3
RADIAN_TO_DEGREE = 360 / 2 / np.pi
REDUCED_CATALOGUE_FILE_PATH = (
    "./master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv"
)


alpha = 7.45 * np.log(10)
beta = 1.05
d_alpha = 0.08 * np.log(10)
d_beta = 0.11


@dataclass
class ParameterSample:
    M: float
    a: float
    redshift: float
    mu: float = 10
    phi_S: float = np.random.random() * 2 * np.pi
    theta_S: float = np.arccos(np.random.random() * 2 - 1)

    def get_distance(self) -> float:
        return dist(self.redshift)


@dataclass
class HostGalaxy:
    phiS: float
    qS: float
    z: float
    z_error: float
    M: float
    M_error: float
    catalog_index: int

    def __init__(self, parameters: pd.Series) -> None:
        self.phiS = parameters[InternalCatalogColumns.PHI_S]
        self.qS = parameters[InternalCatalogColumns.THETA_S]
        self.z = parameters[InternalCatalogColumns.REDSHIFT]
        self.z_error = parameters[InternalCatalogColumns.REDSHIFT_ERROR]
        self.M = parameters[InternalCatalogColumns.BH_MASS]
        self.M_error = parameters[InternalCatalogColumns.BH_MASS_ERROR]
        self.catalog_index = parameters.name


class CatalogueColumns(Enum):
    RIGHT_ASCENSION = 8  # in deg
    DECLINATION = 9  # in deg
    REDSHIFT = 27
    REDSHIFT_PECULIAR_VELOCITY_ERROR = 30
    REDSHIFT_MEASUREMENT_ERROR = 31
    REDSHIFT_FLAG = 34  # flag whether redshift is measured or estimated from distance
    STELLAR_MASS = 35  # in 10^10 solar masses
    STELLAR_MASS_ABSOULTE_ERROR = 36  # in 10^10 solar masses


# IMPORTANT: needs to be in the correct order as above.
class InternalCatalogColumns:
    PHI_S = "RIGHT_ASCENSION"
    THETA_S = "DECLINATION"
    REDSHIFT = "REDSHIFT"
    REDSHIFT_ERROR = "REDSHIFT_MEASUREMENT_ERROR"
    BH_MASS = "STELLAR_MASS"
    BH_MASS_ERROR = "STELLAR_MASS_ABSOULTE_ERROR"


@dataclass
class GalaxyCatalogueHandler:
    reduced_galaxy_catalog: pd.DataFrame

    def __init__(self):
        try:
            self.reduced_galaxy_catalog = self.read_reduced_galaxy_catalog()
            _LOGGER.info("Successfully loaded reduced galaxy catalog.")
        except FileNotFoundError:
            _LOGGER.info(
                "Reduced galaxy catalog not found. Looking for GLADE+.txt in ./galaxy_catalogue directory."
            )
            try:
                self.parse_to_reduced_catalog(
                    galaxy_catalogue_file_path="./master_thesis_code/galaxy_catalogue/GLADE+.txt"
                )
                self.reduced_galaxy_catalog = self.read_reduced_galaxy_catalog()
                _LOGGER.info("Successfully reduced and loaded galaxy catalog.")
            except FileNotFoundError as e:
                _LOGGER.error(
                    "No reduced galaxy catalog or GLADE+.txt export was found. Please provide galaxy catalog and restart."
                )
                raise FileNotFoundError

        _LOGGER.info(
            "Mapping catalog to spherical coordinates and using empirical relation to estimate BH mass and convert to redshifted mass."
        )
        self._map_stellar_masses_to_BH_masses()
        self._map_angles_to_spherical_coordinates()
        self._remove_galaxies_without_mass_information()
        # self._map_BH_masses_to_redshifted_masses()
        self._show_catalog_information()

    def _show_catalog_information(self) -> None:
        bh_mass_not_given = len(
            self.reduced_galaxy_catalog[
                self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS].isna()
            ]
        )
        _LOGGER.info(
            f"Galaxies without stellar mass estimation {bh_mass_not_given/len(self.reduced_galaxy_catalog)*100}%"
        )
        bh_mass_given_statistics = self.reduced_galaxy_catalog[
            ~self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS].isna()
        ].describe()
        _LOGGER.info(
            f"Galaxies with stellar mass estimation statistics\n: {bh_mass_given_statistics}"
        )

    def visualize_galaxy_catalog(self) -> None:
        figures_directory = "./saved_figures/galaxy_catalogue/"
        if not os.path.exists(figures_directory):
            os.makedirs(figures_directory)

        # visualize mass distribution
        fig, ax = plt.subplots()
        ax.hist(self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS], bins=200)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("BH mass in solar masses")
        ax.set_ylabel("Number of galaxies with BH mass")
        plt.savefig(figures_directory + "estimated_BH_mass_distribution.png")
        plt.close()

        # visualize redshift distribution
        fig, ax = plt.subplots()
        ax.hist(self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT], bins=200)
        ax.set_xlabel("Redshift")
        ax.set_ylabel("Number of galaxies with redshift")
        ax.set_yscale("log")
        plt.savefig(figures_directory + "redshift_distribution.png")
        plt.close()

    def parse_to_reduced_catalog(self, galaxy_catalogue_file_path: str) -> None:
        iterator = pd.read_csv(
            filepath_or_buffer=galaxy_catalogue_file_path,
            sep=" ",
            header=None,
            usecols=[column.value for column in CatalogueColumns],
            names=[column.name for column in CatalogueColumns],
            chunksize=10_000,
        )

        _LOGGER.info(f"Start reducing galaxy catalog.")
        next_progress_threshold = 5
        for count, chunk in enumerate(iterator):
            progress = int(count * 10_000 / 230_000)

            if progress >= next_progress_threshold:
                _LOGGER.info(f"Progress: {progress}")
                next_progress_threshold += 5

            # 1, 3 are measured redshifts, 2 is estimated from distance
            chunk = chunk[
                (chunk[CatalogueColumns.REDSHIFT_FLAG.name] == 1)
                | (chunk[CatalogueColumns.REDSHIFT_FLAG.name] == 3)
            ]

            chunk = chunk.fillna(
                {CatalogueColumns.REDSHIFT_PECULIAR_VELOCITY_ERROR.name: 0.0}
            )

            # propagating errors of redshift
            chunk[CatalogueColumns.REDSHIFT_MEASUREMENT_ERROR.name] = np.sqrt(
                chunk[CatalogueColumns.REDSHIFT_MEASUREMENT_ERROR.name] ** 2
                + chunk[CatalogueColumns.REDSHIFT_PECULIAR_VELOCITY_ERROR.name] ** 2
            )

            chunk = chunk.drop(
                columns=[
                    CatalogueColumns.REDSHIFT_PECULIAR_VELOCITY_ERROR.name,
                    CatalogueColumns.REDSHIFT_FLAG.name,
                ]
            )

            chunk.to_csv(
                REDUCED_CATALOGUE_FILE_PATH, header=False, mode="a", index=False
            )

    def read_reduced_galaxy_catalog(self) -> pd.DataFrame:
        return pd.read_csv(
            REDUCED_CATALOGUE_FILE_PATH,
            names=[
                column.name
                for column in CatalogueColumns
                if column.value not in [30, 34]
            ],
        )

    def get_host_galaxy_by_index(self, index: int) -> HostGalaxy:
        return HostGalaxy(self.reduced_galaxy_catalog.loc[index])

    def get_possible_hosts(
        self,
        M_z,
        M_z_error,
        z_min,
        z_max,
        phi,
        phi_error,
        theta,
        theta_error,
        cutoff_multiplier: float = 2,
    ) -> Optional[Tuple[List[HostGalaxy], List[HostGalaxy]]]:

        _LOGGER.info(
            "Searching for possible hosts within:"
            f"\nM = {M_z} +/+ {M_z_error*cutoff_multiplier}"
            f"\n {z_min} <= z <= {z_max}"
            f"\nphi = {phi} +/- {phi_error*cutoff_multiplier}"
            f"\ntheta = {theta} +/- {theta_error*cutoff_multiplier}"
        )

        possible_host_galaxies = self.reduced_galaxy_catalog.loc[
            (
                theta - theta_error * cutoff_multiplier
                <= self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S]
            )
            & (
                self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S]
                <= theta + theta_error * cutoff_multiplier
            )
            & (
                phi - phi_error * cutoff_multiplier
                <= self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S]
            )
            & (
                self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S]
                <= phi + phi_error * cutoff_multiplier
            )
            & (
                z_min
                <= self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT]
                + self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT_ERROR]
            )
            & (
                z_max
                >= self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT]
                - self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT_ERROR]
            )
        ]

        if possible_host_galaxies.empty:
            _LOGGER.warning("No possible hosts. Returning None.")
            return None

        possible_host_galaxies_with_BH_mass = possible_host_galaxies[
            (
                (
                    (M_z - M_z_error * cutoff_multiplier) / (1 + z_max)
                    <= possible_host_galaxies[InternalCatalogColumns.BH_MASS]
                    + possible_host_galaxies[InternalCatalogColumns.BH_MASS_ERROR]
                )
                & (
                    possible_host_galaxies[InternalCatalogColumns.BH_MASS]
                    - possible_host_galaxies[InternalCatalogColumns.BH_MASS_ERROR]
                    <= (M_z + M_z_error * cutoff_multiplier) / (1 + z_min)
                )
            )
        ]

        possible_host_galaxies = [
            HostGalaxy(parameters)
            for _, parameters in possible_host_galaxies.iterrows()
        ]

        possible_host_galaxies_with_BH_mass = [
            HostGalaxy(parameters)
            for _, parameters in possible_host_galaxies_with_BH_mass.iterrows()
        ]
        return (possible_host_galaxies, possible_host_galaxies_with_BH_mass)

    def _remove_galaxies_without_mass_information(self) -> None:
        self.reduced_galaxy_catalog = self.reduced_galaxy_catalog[
            ~self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS].isna()
        ]

    def _map_stellar_masses_to_BH_masses(self) -> None:
        BH_mass, BH_mass_error = _empiric_stellar_mass_to_BH_mass_relation(
            self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS],
            self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS_ERROR],
        )
        self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS] = BH_mass
        self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS_ERROR] = (
            BH_mass_error
        )

    def _map_angles_to_spherical_coordinates(self) -> None:
        self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S] = (
            self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S] * np.pi / 180
        )
        self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S] = (
            self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S] * np.pi / 180
            - np.pi / 2
        ) * (-1)

    def _map_BH_masses_to_redshifted_masses(self) -> None:
        self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS] = (
            self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS]
            * (1 + self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT])
        )
        self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS_ERROR] = np.sqrt(
            (
                self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS_ERROR]
                * (1 + self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT])
            )
            ** 2
            + (
                self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS]
                * self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT_ERROR]
            )
            ** 2
        )

    def get_random_hosts_in_mass_range(
        self, lower_limit: float, upper_limit: float, max_dist: float = 4.5
    ) -> Iterable:
        NUMBER_OF_HOSTS = 400
        thetas = np.arccos(np.random.uniform(-1.0, 1.0, NUMBER_OF_HOSTS))
        phis = np.random.uniform(0.0, 2 * np.pi, NUMBER_OF_HOSTS)

        restricted_galaxy_catalogue = self.reduced_galaxy_catalog[
            (self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS] >= lower_limit)
            & (
                self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS]
                <= upper_limit
            )
            & (self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT] <= max_dist)
        ]

        _LOGGER.debug(
            f"restricted_galaxy_catalogue: {restricted_galaxy_catalogue.shape[0]} galaxies."
        )
        restricted_galaxy_catalogue = restricted_galaxy_catalogue.sample(frac=1)
        return_list = []
        for theta, phi in zip(thetas, phis):

            closest_host_index = (
                (restricted_galaxy_catalogue[InternalCatalogColumns.PHI_S] / phi - 1)
                ** 2
                + (
                    restricted_galaxy_catalogue[InternalCatalogColumns.THETA_S] / theta
                    - 1
                )
                ** 2
            ).idxmin()
            host: pd.Series = restricted_galaxy_catalogue.loc[closest_host_index]
            return_list.append(HostGalaxy(host))
        return iter(return_list)

    def get_hosts_from_parameter_samples(
        self, parameter_samples: List[ParameterSample], max_redshift: float = 3
    ) -> Iterable:
        host_galaxies = []
        _LOGGER.info(
            f"Searching for closest host galaxies for {len(parameter_samples)} parameter samples."
        )
        if len(parameter_samples) > 500:
            _LOGGER.debug("number of samples larger than 500, reducing to 500.")
            parameter_samples = parameter_samples[-500:]
        counter = 0
        for parameter_sample in parameter_samples:
            _LOGGER.debug(
                f"closest host searches progess: {counter/len(parameter_samples)*100}%"
            )
            counter += 1
            closest_host = self._get_closest_host_galaxy(parameter_sample)
            if closest_host is None:
                continue
            if closest_host.z > max_redshift:
                continue
            host_galaxies.append(closest_host)

        _LOGGER.info(
            f"Found {len(host_galaxies)} host galaxies below maximal redshift {max_redshift}."
        )

        return iter(host_galaxies)

    def _get_closest_host_galaxy(
        self, parameter_sample: ParameterSample
    ) -> HostGalaxy | None:
        # sort by distance to redshift and mass
        closest_host_index = self._get_closest_redshift_mass_host_index(
            parameter_sample
        )

        # for now ignore phi, theta
        """
        closest_host_index = (
            (redshift_mass_subset[InternalCatalogColumns.PHI_S] - parameter_sample.phi_S)
            ** 2
            + (
                redshift_mass_subset[InternalCatalogColumns.THETA_S]
                - parameter_sample.theta_S
            )
            ** 2
        ).idxmin()
        """

        host_galaxy = HostGalaxy(self.reduced_galaxy_catalog.loc[closest_host_index])
        _LOGGER.debug(
            f"Found closest host galaxy: z deviation: {np.abs(host_galaxy.z - parameter_sample.redshift)/parameter_sample.redshift}%, M deviation: {np.abs(host_galaxy.M - parameter_sample.M)/parameter_sample.M}%"
        )
        return host_galaxy

    def _get_closest_redshift_mass_host_index(
        self, parameter_sample: ParameterSample
    ) -> int:
        return (
            (
                self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT]
                / parameter_sample.redshift
                - 1
            )
            ** 2
            + (
                np.log10(self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS])
                / np.log10(parameter_sample.M)
                - 1
            )
            ** 2
        ).idxmin()


def _polar_angle_to_declination(polar_angle: float) -> float:
    return np.pi / 2 - polar_angle


def _empiric_stellar_mass_to_BH_mass_relation(
    stellar_mass: float, stellar_mass_error
) -> Tuple:
    BH_mass = np.exp(alpha + beta * np.log(stellar_mass / 10))
    BH_mass_error = BH_mass * np.sqrt(
        d_alpha**2
        + (np.log(stellar_mass / 10) * d_beta) ** 2
        + (beta / stellar_mass / 10 * stellar_mass_error) ** 2
    )
    return (BH_mass, BH_mass_error)


def _empiric_MBH_to_M_stellar_relation(MBH_mass: float, MBH_mass_error: float) -> list:
    stellar_mass = 10 * np.exp((np.log(MBH_mass) - alpha) / beta)
    stellar_mass_error = stellar_mass * np.sqrt(
        (d_alpha / beta) ** 2
        + (beta * MBH_mass_error / MBH_mass) ** 2
        + ((np.log(MBH_mass) - alpha) / beta**2) ** 2 * d_beta**2
    )
    return [stellar_mass, stellar_mass_error]
