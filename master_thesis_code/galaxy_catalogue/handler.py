from enum import Enum
import pandas as pd
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from master_thesis_code.constants import GPC_TO_MPC, RADIAN_TO_DEGREE

REDUCED_CATALOGUE_FILE_PATH = (
    "./master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv"
)


alpha = 7.45 * np.log(10)
beta = 1.05
d_alpha = 0.08 * np.log(10)
d_beta = 0.11


class CatalogueColumns(Enum):
    RIGHT_ASCENSION = 8  # in deg
    DECLINATION = 9  # in deg
    LUMINOSITY_DISTANCE = 32  # in Mpc
    LUMINOSITY_DISTANCE_ERROR = 33  # in Mpc
    STELLAR_MASS = 35  # in 10^10 solar masses
    STELLAR_MASS_ABSOULTE_ERROR = 36  # in 10^10 solar masses
    REDSHIFT = 27
    REDSHIT_ERROR = 30


@dataclass
class GalaxyCatalogueHandler:

    def parse_to_reduced_catalogue(self, galaxy_catalogue_file_path: str) -> None:
        iterator = pd.read_csv(
            filepath_or_buffer=galaxy_catalogue_file_path,
            sep=" ",
            header=None,
            usecols=[column.value for column in CatalogueColumns],
            chunksize=10_000,
        )
        for count, chunk in enumerate(iterator):
            print(f"{int(count*10_000/230_000)}% complete.")
            chunk.to_csv(
                REDUCED_CATALOGUE_FILE_PATH, header=False, mode="a", index=False
            )

    def read_reduced_galaxy_catalogue(self) -> pd.DataFrame:
        return pd.read_csv(
            REDUCED_CATALOGUE_FILE_PATH,
            names=[column.name for column in CatalogueColumns],
        )

    def parse_possible_hosts(self, cramer_rao_bounds_file_path: str) -> None:
        cramer_rao_bounds = pd.read_csv(cramer_rao_bounds_file_path)
        galaxy_catalogue = self.read_reduced_galaxy_catalogue()

        print(galaxy_catalogue[CatalogueColumns.STELLAR_MASS.name].describe())

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


def _polar_angle_to_declination(polar_angle: float) -> float:
    return np.pi / 2 - polar_angle


def _empiric_MBH_to_M_stellar_relation(MBH_mass: float, MBH_mass_error: float) -> list:
    print(MBH_mass, MBH_mass_error)
    stellar_mass = 10 * np.exp((np.log(MBH_mass) - alpha) / beta)
    stellar_mass_error = stellar_mass * np.sqrt(
        (d_alpha / beta) ** 2
        + (beta * MBH_mass_error / MBH_mass) ** 2
        + ((np.log(MBH_mass) - alpha) / beta**2) ** 2 * d_beta**2
    )
    return [stellar_mass, stellar_mass_error]


galaxy_catalogue_handler = GalaxyCatalogueHandler()
galaxy_catalogue_handler.parse_to_reduced_catalogue(
    galaxy_catalogue_file_path="master_thesis_code/galaxy_catalogue/GLADE+.txt"
)
# galaxy_catalogue_handler.parse_possible_hosts("./simulations/cramer_rao_bounds.csv")
