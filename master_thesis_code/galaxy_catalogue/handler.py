import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

# import normal distribution
from statistics import NormalDist

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord
from sklearn.neighbors import BallTree

from master_thesis_code.physical_relations import (
    dist,
    dist_to_redshift_error_proagation,
)

_LOGGER = logging.getLogger()
REDUCED_CATALOGUE_FILE_PATH = "./master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv"
M_min = 10**4
M_max = 10**6
Z_draw = 1.5


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
    phi_S: float = field(
        default_factory=lambda: float(np.random.default_rng().uniform(0, 2 * np.pi))
    )
    theta_S: float = field(
        default_factory=lambda: float(np.arccos(np.random.default_rng().uniform(-1, 1)))
    )

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HostGalaxy):
            return False
        return bool(
            self.catalog_index == other.catalog_index
        )  # Compare based on a unique identifier

    def __hash__(self) -> int:
        return hash(self.catalog_index)  # Use the unique identifier for hashing

    @classmethod
    def from_attributes(
        cls,
        phiS: float,
        qS: float,
        z: float,
        z_error: float,
        M: float,
        M_error: float,
    ) -> "HostGalaxy":
        parameters = pd.Series(
            {
                InternalCatalogColumns.PHI_S: phiS,
                InternalCatalogColumns.THETA_S: qS,
                InternalCatalogColumns.REDSHIFT: z,
                InternalCatalogColumns.REDSHIFT_ERROR: z_error,
                InternalCatalogColumns.BH_MASS: M,
                InternalCatalogColumns.BH_MASS_ERROR: M_error,
            }
        )
        return HostGalaxy(parameters)

    def draw_z_and_mass_from_gaussian(self) -> None:
        while True:
            self.z = NormalDist(mu=self.z, sigma=self.z_error).samples(1)[0]
            if (self.z >= 0) and (self.z <= Z_draw):
                break
        while True:
            self.M = NormalDist(mu=self.M, sigma=self.M_error).samples(1)[0]
            if (self.M >= M_min) and (self.M <= M_max):
                break


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
    catalog_ball_tree: BallTree
    catalog_4d_ball_tree: BallTree
    M_min: float
    M_max: float
    z_max: float

    def __init__(self, M_min: float, M_max: float, z_max: float) -> None:
        self.M_min = M_min
        self.M_max = M_max
        self.z_max = z_max
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
            except FileNotFoundError:
                _LOGGER.error(
                    "No reduced galaxy catalog or GLADE+.txt export was found. Please provide galaxy catalog and restart."
                )
                raise FileNotFoundError

        _LOGGER.info(
            "Mapping catalog to spherical coordinates and using empirical relation to estimate BH mass."
        )
        self._map_stellar_masses_to_BH_masses()
        self._rotate_equatorial_to_ecliptic()  # COORD-03 (Phase 36): equatorial J2000 -> ecliptic SSB
        self._map_angles_to_spherical_coordinates()
        self._remove_galaxies_without_mass_information()
        self.reduced_galaxy_catalog = self._get_pruned_galaxy_catalog(M_min, M_max, z_max)
        self.set_max_relative_errors()
        self._show_catalog_information()
        self.setup_galaxy_catalog_balltree()
        self.setup_4d_galaxy_catalog_balltree()

    def _get_pruned_galaxy_catalog(self, M_min: float, M_max: float, z_max: float) -> pd.DataFrame:
        mask = (
            (
                self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS]
                + self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS_ERROR]
                >= M_min
            )
            & (
                self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS]
                - self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS_ERROR]
                <= M_max
            )
            & (
                self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT]
                - self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT_ERROR]
                <= z_max
            )
        )
        return self.reduced_galaxy_catalog[mask]

    def set_max_relative_errors(self) -> None:
        self._max_relative_redshift_error = (
            self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT_ERROR]
            / self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT]
        ).max()
        self._max_relative_mass_error = (
            self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS_ERROR]
            / self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS]
        ).max()

    def _show_catalog_information(self) -> None:
        bh_mass_not_given = len(
            self.reduced_galaxy_catalog[
                self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS].isna()
            ]
        )
        _LOGGER.info(
            f"Galaxies without stellar mass estimation {bh_mass_not_given / len(self.reduced_galaxy_catalog) * 100}%"
        )
        bh_mass_given_statistics = self.reduced_galaxy_catalog[
            ~self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS].isna()
        ].describe()
        _LOGGER.info(
            f"Galaxies with stellar mass estimation statistics\n: {bh_mass_given_statistics}"
        )
        _LOGGER.info(f"Pruned galaxy catalog contains {len(self.reduced_galaxy_catalog)} galaxies.")

    def parse_to_reduced_catalog(self, galaxy_catalogue_file_path: str) -> None:
        iterator = pd.read_csv(
            filepath_or_buffer=galaxy_catalogue_file_path,
            sep=" ",
            header=None,
            usecols=[column.value for column in CatalogueColumns],
            names=[column.name for column in CatalogueColumns],
            chunksize=10_000,
        )

        _LOGGER.info("Start reducing galaxy catalog.")
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

            chunk = chunk.fillna({CatalogueColumns.REDSHIFT_PECULIAR_VELOCITY_ERROR.name: 0.0015})

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

            chunk.to_csv(REDUCED_CATALOGUE_FILE_PATH, header=False, mode="a", index=False)

    def parse_to_reduced_catalog_with_reduced_errors(self) -> None:
        catalog = pd.read_csv(
            REDUCED_CATALOGUE_FILE_PATH,
            names=[column.name for column in CatalogueColumns if column.value not in [30, 34]],
        )
        for index, row in catalog.iterrows():
            redshift = row[CatalogueColumns.REDSHIFT.name]
            redshift_error = row[CatalogueColumns.REDSHIFT_MEASUREMENT_ERROR.name]
            new_redshift_error = dist_to_redshift_error_proagation(redshift, redshift_error)
            catalog.at[index, CatalogueColumns.REDSHIFT_MEASUREMENT_ERROR.name] = new_redshift_error

    def read_reduced_galaxy_catalog(self) -> pd.DataFrame:
        return pd.read_csv(
            REDUCED_CATALOGUE_FILE_PATH,
            names=[column.name for column in CatalogueColumns if column.value not in [30, 34]],
        )

    def setup_galaxy_catalog_balltree(self) -> None:
        # expects the reduced galaxy catalog to be setup already
        # Columns were historically named RA/Dec (GLADE equatorial) but after
        # Plan 36-01 (_rotate_equatorial_to_ecliptic), they hold ecliptic polar-
        # angle pairs (θ_polar ∈ [0, π], φ ∈ [0, 2π)). The column constants
        # PHI_S / THETA_S are kept for backward compatibility; see CONTEXT.md D-13.
        phi = self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S].values
        theta = self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S].values

        # Eq. (standard spherical polar); COORD-02 fix per .planning/phases/36-coordinate-frame-fix/36-CONTEXT.md D-17
        data = _polar_to_cartesian(theta, phi)

        self.catalog_ball_tree = BallTree(data, metric="euclidean")
        self.reduced_galaxy_catalog = self.reduced_galaxy_catalog.reset_index()
        _LOGGER.debug(f"BallTree setup with {self.reduced_galaxy_catalog.shape[0]} galaxies.")

    def get_possible_hosts_from_ball_tree(
        self,
        phi: float,
        phi_sigma: float,
        theta: float,
        theta_sigma: float,
        M_z: float,
        M_z_sigma: float,
        z_min: float,
        z_max: float,
        sigma_multiplier: int = 2,
        cov_theta_phi: float = 0.0,
    ) -> tuple[list[HostGalaxy], list[HostGalaxy]] | None:
        """Find candidate host galaxies within the sky-Fisher error ellipse + mass-redshift cuts.

        The sky search radius is ``sigma_multiplier × √λ_max(Σ')`` where
        ``Σ' = J Σ Jᵀ``, ``J = diag(|sin θ|, 1)``, and ``Σ`` is the 2×2 Fisher
        sky covariance block ``[[σ_φ², C_θφ], [C_θφ, σ_θ²]]``. This is the
        chord-length-on-unit-sphere interpretation consistent with the Cartesian
        BallTree embedding (see :func:`_polar_to_cartesian`).

        Args:
            phi: ML estimate of ecliptic azimuth φ_S (rad).
            phi_sigma: 1-σ uncertainty on φ_S (rad).
            theta: ML estimate of ecliptic polar angle θ_S (rad, ∈ [0, π]).
            theta_sigma: 1-σ uncertainty on θ_S (rad).
            M_z: Redshifted central BH mass estimate (solar masses).
            M_z_sigma: 1-σ uncertainty on M_z (solar masses).
            z_min: Lower redshift bound for the galaxy search.
            z_max: Upper redshift bound for the galaxy search.
            sigma_multiplier: Number of σ to use as the search radius (default 2).
            cov_theta_phi: Off-diagonal Cramér-Rao element C_{θφ} (rad²).
                Default 0.0 reduces to the isotropic-ellipse case.
                Positioned at the signature tail so that Python's
                non-default-follows-default rule is respected.

        Returns:
            Tuple of (hosts_without_BH_mass_filter, hosts_with_BH_mass_filter) or None.

        References:
            .planning/phases/36-coordinate-frame-fix/36-CONTEXT.md D-21, D-22.
            master_thesis_code/datamodels/detection.py:15-40 (_sky_localization_uncertainty).
        """
        # Eq. (standard spherical polar); COORD-02 fix per .planning/phases/36-coordinate-frame-fix/36-CONTEXT.md D-17
        # _polar_to_cartesian expects array inputs; wrap scalars.
        query_point = _polar_to_cartesian(np.array([theta]), np.array([phi]))

        # Eq. (eigenvalue of J Σ Jᵀ on 2×2 Fisher sky block); COORD-04 per
        # .planning/phases/36-coordinate-frame-fix/36-CONTEXT.md D-22.
        # J = diag(|sin θ|, 1) rescales the azimuthal std to great-circle distance
        # on the unit sphere (ds² = dθ² + sin²θ dφ² — see detection.py:15-40).
        sigma_matrix = np.array([[phi_sigma**2, cov_theta_phi], [cov_theta_phi, theta_sigma**2]])
        jacobian = np.diag([abs(np.sin(theta)), 1.0])
        sigma_scaled = jacobian @ sigma_matrix @ jacobian.T
        lambda_max = float(np.linalg.eigvalsh(sigma_scaled).max())
        radius = float(sigma_multiplier * np.sqrt(max(lambda_max, 0.0)))

        indices = self.catalog_ball_tree.query_radius(query_point, r=radius)[0]

        candidate_hosts = self.reduced_galaxy_catalog.iloc[indices]

        redshift_filter_mask = (
            z_min
            <= candidate_hosts[InternalCatalogColumns.REDSHIFT]
            + candidate_hosts[InternalCatalogColumns.REDSHIFT_ERROR]
        ) & (
            z_max
            >= candidate_hosts[InternalCatalogColumns.REDSHIFT]
            - candidate_hosts[InternalCatalogColumns.REDSHIFT_ERROR]
        )
        candidate_hosts_without_bh_mass = candidate_hosts[redshift_filter_mask]

        mass_filter_mask = (
            (M_z - M_z_sigma * sigma_multiplier) / (1 + z_max)
            <= candidate_hosts_without_bh_mass[InternalCatalogColumns.BH_MASS]
            + candidate_hosts_without_bh_mass[InternalCatalogColumns.BH_MASS_ERROR]
        ) & (
            candidate_hosts_without_bh_mass[InternalCatalogColumns.BH_MASS]
            - candidate_hosts_without_bh_mass[InternalCatalogColumns.BH_MASS_ERROR]
            <= (M_z + M_z_sigma * sigma_multiplier) / (1 + z_min)
        )

        candidate_hosts_with_bh_mass = candidate_hosts_without_bh_mass[mass_filter_mask]
        possible_hosts_without_bh_mass = [
            HostGalaxy(parameters) for _, parameters in candidate_hosts_without_bh_mass.iterrows()
        ]
        possible_hosts_with_bh_mass = [
            HostGalaxy(parameters) for _, parameters in candidate_hosts_with_bh_mass.iterrows()
        ]
        if (len(possible_hosts_without_bh_mass) == 0) and (len(possible_hosts_with_bh_mass) == 0):
            _LOGGER.warning("No possible hosts. Returning None.")
            return None

        _LOGGER.info(
            f"Found {len(possible_hosts_without_bh_mass)} possible hosts without BH mass and {len(possible_hosts_with_bh_mass)} possible hosts with BH mass."
        )
        return (possible_hosts_without_bh_mass, possible_hosts_with_bh_mass)

    def setup_4d_galaxy_catalog_balltree(self) -> None:
        """Build the 5-D host-assignment BallTree (sky chord + z + log M).

        The sky sub-space uses spherical Cartesian embedding via
        ``_polar_to_cartesian(θ, φ)`` so that chord-length on the unit sphere
        is the sky metric — avoiding the COORD-02b flat-metric bug that
        collapsed equatorial points to a corner of the flat (φ/2π, θ/π)
        square. Redshift and log-mass axes are linearly normalized to [0, 1].

        Metric weights (planner's choice per Claude's Discretion in
        .planning/phases/36-coordinate-frame-fix/36-CONTEXT.md D-18):
        sky chord length ∈ [0, 2] + z_norm ∈ [0, 1] + log_M_norm ∈ [0, 1],
        euclidean on ℝ⁵. This gives the sky axes slightly more weight
        than z or M, which matches the physical intuition: two galaxies
        at the same sky position but different z are candidates for the
        same EMRI sky localization; two galaxies at the same z but
        different sky positions are not.

        Note: the attribute is named ``catalog_4d_ball_tree`` for backward
        compatibility; the tree is actually 5-D (3 sky Cartesian + z_norm +
        log_M_norm) after the COORD-02b fix.

        References:
            COORD-02b fix; .planning/phases/36-coordinate-frame-fix/36-CONTEXT.md D-17, D-18.
            .planning/REQUIREMENTS.md §Coordinate Frame Correctness COORD-02b.
        """
        # Sky sub-space: spherical Cartesian unit vectors (COORD-02b)
        phi = self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S].values
        theta = self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S].values
        # Eq. (standard spherical polar); CONTEXT.md D-17.
        sky_xyz = _polar_to_cartesian(theta, phi)  # shape (N, 3), unit vectors

        # Redshift axis: linear normalization to [0, 1]
        redshift_norm = (
            self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT] / self.z_max
        ).values

        # Mass axis: log normalization to [0, 1]
        log_mass = np.log10(self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS])
        log_mass_min = np.log10(self.M_min)
        log_mass_max = np.log10(self.M_max)
        mass_norm = ((log_mass - log_mass_min) / (log_mass_max - log_mass_min)).values

        # Combine into a 5-D array: [sky_x, sky_y, sky_z, z_norm, log_M_norm]
        data = np.hstack((sky_xyz, redshift_norm.reshape(-1, 1), mass_norm.reshape(-1, 1)))

        self.catalog_4d_ball_tree = BallTree(data, metric="euclidean")
        _LOGGER.info("5-D BallTree (3 sky Cartesian + z_norm + log_M_norm) built successfully.")

    def find_closest_galaxy_to_coordinates(
        self, phi: float, theta: float, redshift: float, mass: float
    ) -> HostGalaxy:
        """Return the catalog galaxy closest to (φ, θ, z, M) under the 5-D metric.

        Sky query embedded via the same ``_polar_to_cartesian`` helper used in
        ``setup_4d_galaxy_catalog_balltree`` — structural symmetry per D-17
        ensures tree data and query point live in the same 5-D space.

        References:
            COORD-02b fix; .planning/phases/36-coordinate-frame-fix/36-CONTEXT.md D-17, D-18.
        """
        # Sky sub-space: spherical Cartesian unit vector (COORD-02b)
        # Eq. (standard spherical polar); CONTEXT.md D-17.
        sky_xyz = _polar_to_cartesian(np.array([theta]), np.array([phi]))  # shape (1, 3)

        # Normalized z and log M, matching setup_4d_galaxy_catalog_balltree.
        redshift_norm = redshift / self.z_max
        log_mass_norm = (np.log10(mass) - np.log10(self.M_min)) / (
            np.log10(self.M_max) - np.log10(self.M_min)
        )

        # Combine into (1, 5) query point: [sky_x, sky_y, sky_z, z_norm, log_M_norm]
        query_point = np.hstack((sky_xyz, np.array([[redshift_norm, log_mass_norm]])))

        # Query the BallTree
        distance, index = self.catalog_4d_ball_tree.query(query_point, k=1)
        closest_galaxy = self.reduced_galaxy_catalog.iloc[index[0][0]]

        return HostGalaxy(closest_galaxy)

    def get_host_galaxy_by_index(self, index: int) -> HostGalaxy:
        return HostGalaxy(self.reduced_galaxy_catalog.loc[index])

    def get_possible_hosts(
        self,
        M_z: float,
        M_z_error: float,
        z_min: float,
        z_max: float,
        phi: float,
        phi_error: float,
        theta: float,
        theta_error: float,
        cutoff_multiplier: float = 2,
    ) -> tuple[list[HostGalaxy], list[HostGalaxy]] | None:
        _LOGGER.info(
            "Searching for possible hosts within:"
            f"\nM = {M_z} +/+ {M_z_error * cutoff_multiplier}"
            f"\n {z_min} <= z <= {z_max}"
            f"\nphi = {phi} +/- {phi_error * cutoff_multiplier}"
            f"\ntheta = {theta} +/- {theta_error * cutoff_multiplier}"
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
            HostGalaxy(parameters) for _, parameters in possible_host_galaxies.iterrows()
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
        self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS_ERROR] = BH_mass_error

    def _rotate_equatorial_to_ecliptic(self) -> None:
        """Rotate catalog RA/Dec from equatorial ICRS J2000 to ecliptic SSB.

        The GLADE+ catalog stores source positions as equatorial right ascension
        and declination at the J2000 epoch. LISA waveform conventions use the
        barycentric ecliptic frame. This method performs the vectorized
        astropy rotation once at catalog load time. After this call, the
        columns ``PHI_S`` / ``THETA_S`` hold ecliptic longitude / latitude
        (degrees, ranges ``[0, 360)`` and ``[-90, +90]``); a subsequent call
        to :meth:`_map_angles_to_spherical_coordinates` converts these to
        radians plus the standard polar-angle offset.

        Hard range assertions (D-15) fail loud rather than silently drift.

        References:
            astropy.coordinates.BarycentricTrueEcliptic(equinox='J2000').
            .planning/phases/36-coordinate-frame-fix/36-CONTEXT.md D-13, D-14, D-15.
        """
        ra_deg = self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S].values
        dec_deg = self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S].values

        # Eq. (astropy BarycentricTrueEcliptic(J2000));
        # .planning/phases/36-coordinate-frame-fix/36-CONTEXT.md D-13
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        ecl = coord.transform_to(BarycentricTrueEcliptic(equinox="J2000"))
        lon_deg = ecl.lon.to(u.deg).value % 360.0
        lat_deg = ecl.lat.to(u.deg).value

        # D-15: fail loud on out-of-range outputs; no silent coordinate drift.
        assert np.all((lon_deg >= 0) & (lon_deg < 360)), (
            f"Ecliptic longitude out of [0, 360): min={lon_deg.min()}, max={lon_deg.max()}"
        )
        assert np.all((lat_deg >= -90) & (lat_deg <= 90)), (
            f"Ecliptic latitude out of [-90, +90]: min={lat_deg.min()}, max={lat_deg.max()}"
        )

        self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S] = lon_deg
        self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S] = lat_deg

    def _map_angles_to_spherical_coordinates(self) -> None:
        """Convert ecliptic (lon, lat) in degrees to (φ, θ_polar) in radians.

        Expects :meth:`_rotate_equatorial_to_ecliptic` to have been called
        first (see Phase 36 COORD-03). ``θ_polar = π/2 − β`` ∈ ``[0, π]``.
        """
        self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S] = (
            self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S] * np.pi / 180
        )
        self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S] = (
            self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S] * np.pi / 180 - np.pi / 2
        ) * (-1)

    def _map_BH_masses_to_redshifted_masses(self) -> None:
        self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS] = self.reduced_galaxy_catalog[
            InternalCatalogColumns.BH_MASS
        ] * (1 + self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT])
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

    def get_random_hosts(self, number_of_hosts: int) -> Iterable:
        random_hosts = self.reduced_galaxy_catalog.sample(number_of_hosts)
        return iter([HostGalaxy(parameters) for _, parameters in random_hosts.iterrows()])

    def get_random_hosts_in_mass_range(
        self,
        lower_limit: float,
        upper_limit: float,
        max_dist: float = 4.5,
        number_of_hosts: int = 500,
        impose_isotropic: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Iterable:
        if rng is None:
            rng = np.random.default_rng()
        thetas = np.arccos(rng.uniform(-1.0, 1.0, number_of_hosts))
        phis = rng.uniform(0.0, 2 * np.pi, number_of_hosts)

        restricted_galaxy_catalogue = self.reduced_galaxy_catalog[
            (self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS] >= lower_limit)
            & (self.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS] <= upper_limit)
            & (self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT] <= max_dist)
        ]

        _LOGGER.debug(
            f"restricted_galaxy_catalogue: {restricted_galaxy_catalogue.shape[0]} galaxies."
        )
        if impose_isotropic:
            restricted_galaxy_catalogue = restricted_galaxy_catalogue.sample(frac=1)
            return_list = []
            for theta, phi in zip(thetas, phis):
                closest_host_index = (
                    (restricted_galaxy_catalogue[InternalCatalogColumns.PHI_S] / phi - 1) ** 2
                    + (restricted_galaxy_catalogue[InternalCatalogColumns.THETA_S] / theta - 1) ** 2
                ).idxmin()
                host: pd.Series = restricted_galaxy_catalogue.loc[closest_host_index]
                return_list.append(HostGalaxy(host))
            return iter(return_list)
        else:
            random_hosts = restricted_galaxy_catalogue.sample(number_of_hosts)
            return iter([HostGalaxy(parameters) for _, parameters in random_hosts.iterrows()])

    def get_hosts_from_parameter_samples(
        self, parameter_samples: list[ParameterSample]
    ) -> Iterable[HostGalaxy]:
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
                f"closest host searches progess: {counter / len(parameter_samples) * 100}%"
            )
            counter += 1

            # check if redshift and mass are within bounds
            if (
                parameter_sample.redshift > self.z_max
                or parameter_sample.redshift < 0
                or parameter_sample.M < self.M_min
                or parameter_sample.M > self.M_max
            ):
                _LOGGER.debug(f"Parameter sample out of bounds: {parameter_sample}. Skipping.")
                continue

            closest_host = self.find_closest_galaxy_to_coordinates(
                parameter_sample.phi_S,
                parameter_sample.theta_S,
                parameter_sample.redshift,
                parameter_sample.M,
            )
            host_galaxies.append(closest_host)

        _LOGGER.info(
            f"Found {len(host_galaxies)} host galaxies below maximal redshift {self.z_max} and within mass bounds [{self.M_min}, {self.M_max}]."
        )

        return iter(host_galaxies)

    def _get_closest_host_galaxy(self, parameter_sample: ParameterSample) -> HostGalaxy | None:
        # sort by distance to redshift and mass
        closest_host_index = self._get_closest_redshift_mass_host_index(parameter_sample)

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
        # check if host galaxy is within error bounds
        if (
            np.abs(host_galaxy.z - parameter_sample.redshift) / parameter_sample.redshift
            > self._max_relative_redshift_error
        ) or (
            np.abs(host_galaxy.M - parameter_sample.M) / parameter_sample.M
            > self._max_relative_mass_error
        ):
            _LOGGER.debug("Host galaxy not within error bounds. Returning None.")
            return None
        _LOGGER.debug(
            f"Found closest host galaxy: z deviation: {np.abs(host_galaxy.z - parameter_sample.redshift) / parameter_sample.redshift}%, M deviation: {np.abs(host_galaxy.M - parameter_sample.M) / parameter_sample.M}%"
        )
        return host_galaxy

    def _get_closest_redshift_mass_host_index(self, parameter_sample: ParameterSample) -> int:
        return int(
            (
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
        )


def _polar_to_cartesian(
    theta: npt.NDArray[np.float64], phi: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Map polar (θ, φ) to Cartesian unit vectors on S².

    Uses the standard physics convention where θ is the polar angle
    measured from the north pole: θ ∈ [0, π], θ=0 at north pole.

    Args:
        theta: Polar angle(s) in radians, shape (N,) or scalar.
        phi: Azimuthal angle(s) in radians, shape (N,) or scalar.

    Returns:
        (N, 3) array of unit vectors (x, y, z) = (sin θ cos φ, sin θ sin φ, cos θ).
        Each row satisfies ||v||₂ = 1 to floating-point precision.

    References:
        Standard spherical polar convention; see
        .planning/phases/36-coordinate-frame-fix/36-CONTEXT.md D-17.
    """
    # Eq. (standard spherical polar); .planning/phases/36-coordinate-frame-fix/36-CONTEXT.md D-17
    return np.vstack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))).T


def _polar_angle_to_declination(polar_angle: float) -> float:
    return np.pi / 2 - polar_angle


def _empiric_stellar_mass_to_BH_mass_relation(
    stellar_mass: float, stellar_mass_error: float
) -> tuple[float, float]:
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
