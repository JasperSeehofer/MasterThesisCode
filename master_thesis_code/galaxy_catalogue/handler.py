from enum import Enum
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from typing import Tuple

_LOGGER = logging.getLogger()


GPC_TO_MPC, RADIAN_TO_DEGREE = 10**3, 360 / 2 / np.pi
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
    BH_MASS_COLUMN = "STELLAR_MASS"
    BH_MASS_ERROR_COLUMN = "STELLAR_MASS_ABSOULTE_ERROR"


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

    def parse_possible_hosts(self, cramer_rao_bounds_file_path: str) -> None:
        cramer_rao_bounds = pd.read_csv(cramer_rao_bounds_file_path)
        galaxy_catalogue = self.read_reduced_galaxy_catalog()

        for index, event in cramer_rao_bounds.iterrows():
            # true values
            phi_true = event["phiS"]
            theta_true = event["qS"]
            d_true = event["dist"]
            M_true = event["M"]

            # errors
            dphi = np.sqrt(event["delta_phiS_delta_phiS"])
            dtheta = np.sqrt(event["delta_qS_delta_qS"])
            dd = np.sqrt(event["delta_dist_delta_dist"])
            d_M = np.sqrt(event["delta_M_delta_M"])

            # transform to catalogue angles
            right_ascension = phi_true * RADIAN_TO_DEGREE
            declination = _polar_angle_to_declination(theta_true) * RADIAN_TO_DEGREE

            declination_error = dtheta * RADIAN_TO_DEGREE
            right_ascension_error = dphi * RADIAN_TO_DEGREE

            d_in_mpc = d_true * GPC_TO_MPC
            dd_in_mpc = dd * GPC_TO_MPC

            stellar_mass_estimate, stellar_mass_error = (
                _empiric_MBH_to_M_stellar_relation(M_true, d_M)
            )

            possible_host_galaxies = galaxy_catalogue.loc[
                (
                    declination - declination_error
                    <= galaxy_catalogue[CatalogueColumns.DECLINATION.name]
                )
                & (
                    galaxy_catalogue[CatalogueColumns.DECLINATION.name]
                    <= declination + declination_error
                )
                & (
                    right_ascension - right_ascension_error
                    <= galaxy_catalogue[CatalogueColumns.RIGHT_ASCENSION.name]
                )
                & (
                    galaxy_catalogue[CatalogueColumns.RIGHT_ASCENSION.name]
                    <= right_ascension + right_ascension_error
                )
                & (
                    d_in_mpc - dd_in_mpc
                    <= galaxy_catalogue[CatalogueColumns.LUMINOSITY_DISTANCE.name]
                    + galaxy_catalogue[CatalogueColumns.LUMINOSITY_DISTANCE_ERROR.name]
                )
                & (
                    galaxy_catalogue[CatalogueColumns.LUMINOSITY_DISTANCE.name]
                    - galaxy_catalogue[CatalogueColumns.LUMINOSITY_DISTANCE_ERROR.name]
                    <= d_in_mpc + dd_in_mpc
                )
            ]

            possible_host_galaxies_with_BH_mass = galaxy_catalogue.loc[
                (
                    declination - declination_error
                    <= galaxy_catalogue[CatalogueColumns.DECLINATION.name]
                )
                & (
                    galaxy_catalogue[CatalogueColumns.DECLINATION.name]
                    <= declination + declination_error
                )
                & (
                    right_ascension - right_ascension_error
                    <= galaxy_catalogue[CatalogueColumns.RIGHT_ASCENSION.name]
                )
                & (
                    galaxy_catalogue[CatalogueColumns.RIGHT_ASCENSION.name]
                    <= right_ascension + right_ascension_error
                )
                & (
                    d_in_mpc - dd_in_mpc
                    <= galaxy_catalogue[CatalogueColumns.LUMINOSITY_DISTANCE.name]
                    + galaxy_catalogue[CatalogueColumns.LUMINOSITY_DISTANCE_ERROR.name]
                )
                & (
                    galaxy_catalogue[CatalogueColumns.LUMINOSITY_DISTANCE.name]
                    - galaxy_catalogue[CatalogueColumns.LUMINOSITY_DISTANCE_ERROR.name]
                    <= d_in_mpc + dd_in_mpc
                )
                & (
                    stellar_mass_estimate - stellar_mass_error
                    <= galaxy_catalogue[CatalogueColumns.STELLAR_MASS.name]
                    + galaxy_catalogue[
                        CatalogueColumns.STELLAR_MASS_ABSOULTE_ERROR.name
                    ]
                )
                & (
                    galaxy_catalogue[CatalogueColumns.STELLAR_MASS.name]
                    - galaxy_catalogue[
                        CatalogueColumns.STELLAR_MASS_ABSOULTE_ERROR.name
                    ]
                    <= stellar_mass_estimate + stellar_mass_error
                )
            ]
            print(
                f"stellar_mass_estimation: {stellar_mass_estimate} +- {round(stellar_mass_error/stellar_mass_estimate*100, 2)}% in 10^10 solar masses."
            )
            print(
                f"found {len(possible_host_galaxies)} ({len(possible_host_galaxies_with_BH_mass)}) possible host galaxies (with BH mass)."
            )

    def _map_stellar_masses_to_BH_masses(self) -> None:
        BH_mass, BH_mass_error = _empiric_stellar_mass_to_BH_mass_relation(
            self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS_COLUMN],
            self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS_ERROR_COLUMN],
        )
        self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS_COLUMN] = BH_mass
        self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS_ERROR_COLUMN] = (
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

    def get_random_host_in_mass_range(
        self, lower_limit: float, upper_limit: float
    ) -> HostGalaxy:
        max_iter = 10000
        iter = 0
        while True:
            host = self.reduced_galaxy_catalog.sample().iloc[0]
            if (host[InternalCatalogColumns.BH_MASS_COLUMN] >= lower_limit) and (
                host[InternalCatalogColumns.BH_MASS_COLUMN] <= upper_limit
            ):
                return HostGalaxy(
                    phiS=host[InternalCatalogColumns.PHI_S],
                    qS=host[InternalCatalogColumns.THETA_S],
                    dist=host[InternalCatalogColumns.LUMINOSITY_DISTANCE],
                    dist_error=host[InternalCatalogColumns.LUMINOSITY_DISTANCE_ERROR],
                    z=host[InternalCatalogColumns.REDSHIFT],
                    z_error=host[InternalCatalogColumns.REDSHIFT_ERROR],
                    M=host[InternalCatalogColumns.BH_MASS_COLUMN],
                    M_error=host[InternalCatalogColumns.BH_MASS_ERROR_COLUMN],
                    catalog_index=int(host.name),
                )
            if max_iter <= iter:
                raise Exception("maximum iterations reached in search for host galaxy.")
            iter += 1


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
