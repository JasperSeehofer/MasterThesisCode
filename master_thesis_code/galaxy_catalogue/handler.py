from enum import Enum
import pandas as pd
import numpy as np
from collections.abc import Iterable
from dataclasses import dataclass
import logging
from typing import Tuple, List
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from master_thesis_code.constants import H0, C

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
class HostGalaxy:
    phiS: float
    qS: float
    dist: float
    dist_error: float
    z: float
    z_error: float
    M: float
    M_error: float
    catalog_index: int

    def __init__(self, parameters: pd.Series) -> None:
        self.phiS = parameters[InternalCatalogColumns.PHI_S]
        self.qS = parameters[InternalCatalogColumns.THETA_S]
        self.dist = parameters[InternalCatalogColumns.LUMINOSITY_DISTANCE]
        self.dist_error = parameters[
            InternalCatalogColumns.LUMINOSITY_DISTANCE_ERROR
        ]  # in Mpc
        self.z = parameters[InternalCatalogColumns.REDSHIFT]
        self.z_error = parameters[InternalCatalogColumns.REDSHIFT_ERROR]
        self.M = parameters[InternalCatalogColumns.BH_MASS]
        self.M_error = parameters[InternalCatalogColumns.BH_MASS_ERROR]
        self.catalog_index = parameters.name


class CatalogueColumns(Enum):
    RIGHT_ASCENSION = 8  # in deg
    DECLINATION = 9  # in deg
    REDSHIFT = 27
    REDSHIFT_ERROR = 30
    LUMINOSITY_DISTANCE = 32  # in Mpc
    LUMINOSITY_DISTANCE_ERROR = 33  # in Mpc
    STELLAR_MASS = 35  # in 10^10 solar masses
    STELLAR_MASS_ABSOULTE_ERROR = 36  # in 10^10 solar masses


# IMPORTANT: needs to be in the correct order as above.
class InternalCatalogColumns:
    PHI_S = "RIGHT_ASCENSION"
    THETA_S = "DECLINATION"
    REDSHIFT = "REDSHIFT"
    REDSHIFT_ERROR = "REDSHIFT_ERROR"
    LUMINOSITY_DISTANCE = "LUMINOSITY_DISTANCE"
    LUMINOSITY_DISTANCE_ERROR = "LUMINOSITY_DISTANCE_ERROR"
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
            "Mapping catalog to spherical coordinates and using empirical relation to estimate BH mass."
        )
        self._map_stellar_masses_to_BH_masses()
        self._map_angles_to_spherical_coordinates()

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

            chunk.to_csv(
                REDUCED_CATALOGUE_FILE_PATH, header=False, mode="a", index=False
            )

    def read_reduced_galaxy_catalog(self) -> pd.DataFrame:
        return pd.read_csv(
            REDUCED_CATALOGUE_FILE_PATH,
            names=[column.name for column in CatalogueColumns],
        )

    def get_possible_hosts(
        self,
        M_z,
        M_z_error,
        dist,
        dist_error,
        phi,
        phi_error,
        theta,
        theta_error,
        cutoff_multiplier: float = 2,
    ) -> tuple[List[HostGalaxy], List[HostGalaxy]]:

        z = _convert_dist_to_redshift(dist * GPC_TO_MPC)
        z_error = _convert_dist_to_redshift(dist_error * GPC_TO_MPC)

        M, M_error = _convert_redshifted_mass_to_true_mass(M_z, M_z_error, z, z_error)

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
                z - z_error * cutoff_multiplier
                <= self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT]
                + self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT_ERROR]
            )
            & (
                self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT]
                - self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT_ERROR]
                <= z + z_error * cutoff_multiplier
            )
        ]

        if possible_host_galaxies.empty:
            _LOGGER.warning("No possible hosts. Returning empty lists.")
            return [], []

        possible_host_galaxies_with_BH_mass = possible_host_galaxies[
            (
                (
                    M - M_error * cutoff_multiplier
                    <= possible_host_galaxies[InternalCatalogColumns.BH_MASS]
                    + possible_host_galaxies[
                        CatalogueColumns.STELLAR_MASS_ABSOULTE_ERROR.name
                    ]
                )
                & (
                    possible_host_galaxies[CatalogueColumns.STELLAR_MASS.name]
                    - possible_host_galaxies[
                        CatalogueColumns.STELLAR_MASS_ABSOULTE_ERROR.name
                    ]
                    <= M + M_error * cutoff_multiplier
                )
            )
            | (possible_host_galaxies[InternalCatalogColumns.BH_MASS].isna())
        ]

        possible_host_galaxies = [
            HostGalaxy(parameters)
            for _, parameters in possible_host_galaxies.iterrows()
        ]

        possible_host_galaxies_with_BH_mass = [
            HostGalaxy(parameters)
            for _, parameters in possible_host_galaxies_with_BH_mass.iterrows()
        ]
        return possible_host_galaxies, possible_host_galaxies_with_BH_mass

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

    def get_random_hosts_in_mass_range(
        self, lower_limit: float, upper_limit: float, max_dist: float = 4.5
    ) -> Iterable:
        NUMBER_OF_HOSTS = 500
        thetas = np.arccos(np.random.uniform(-1.0, 1.0, NUMBER_OF_HOSTS))
        phis = np.random.uniform(0.0, 2 * np.pi, NUMBER_OF_HOSTS)
        distances = np.random.uniform(0.05, max_dist, NUMBER_OF_HOSTS) * GPC_TO_MPC

        restricted_galaxy_catalogue = self.reduced_galaxy_catalog[
            (self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS] >= lower_limit)
            & (
                self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS]
                <= upper_limit
            )
            & (
                self.reduced_galaxy_catalog[InternalCatalogColumns.LUMINOSITY_DISTANCE]
                <= max_dist * GPC_TO_MPC
            )
        ]

        return_list = []
        for theta, phi, distance in zip(thetas, phis, distances):

            closest_host_index = (
                (restricted_galaxy_catalogue[InternalCatalogColumns.PHI_S] / phi - 1)
                ** 2
                + (
                    restricted_galaxy_catalogue[InternalCatalogColumns.THETA_S] / theta
                    - 1
                )
                ** 2
                + (
                    restricted_galaxy_catalogue[
                        InternalCatalogColumns.LUMINOSITY_DISTANCE
                    ]
                    / distance
                    - 1
                )
                ** 2
            ).idxmin()
            host: pd.Series = restricted_galaxy_catalogue.loc[closest_host_index]

            return_list.append(
                HostGalaxy(
                    phiS=host[InternalCatalogColumns.PHI_S],
                    qS=host[InternalCatalogColumns.THETA_S],
                    dist=host[InternalCatalogColumns.LUMINOSITY_DISTANCE] * 1e-3,
                    dist_error=host[InternalCatalogColumns.LUMINOSITY_DISTANCE_ERROR]
                    * 1e-3,
                    z=host[InternalCatalogColumns.REDSHIFT],
                    z_error=host[InternalCatalogColumns.REDSHIFT_ERROR],
                    M=host[InternalCatalogColumns.BH_MASS],
                    M_error=host[InternalCatalogColumns.BH_MASS_ERROR],
                    catalog_index=host.name,
                )
            )
        return iter(return_list)


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


def _convert_dist_to_redshift(dist: float) -> float:
    return dist * H0 / C


def _convert_redshifted_mass_to_true_mass(
    M_z: float, M_z_error: float, z: float, z_error
) -> float:
    M = M_z / (1 + z)
    M_err = np.sqrt((M_z_error / (1 + z)) ** 2 + (M_z * z_error / (1 + z) ** 2) ** 2)
    return (M, M_err)
