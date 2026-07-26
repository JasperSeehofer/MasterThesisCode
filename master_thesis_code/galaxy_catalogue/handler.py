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

from master_thesis_code.constants import (
    HOST_DRAW_Z_MAX,
    SIGMA_V_PV_RESIDUAL_CORRECTED_KM_S,
    SIGMA_V_PV_UNCORRECTED_KM_S,
    SPEED_OF_LIGHT_KM_S,
)
from master_thesis_code.emri_rate import R_eff_per_mbh
from master_thesis_code.physical_relations import dist

_LOGGER = logging.getLogger()
REDUCED_CATALOGUE_FILE_PATH = "./master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv"
M_min = 10**4
M_max = 10**6
Z_draw = 1.5


# Stellar-mass -> central-BH-mass relation. Reines & Volonteri (2015), ApJ 813, 82,
# arXiv:1508.06274, Eq. (5) (broad-line AGN, M_BH-M_*,total):
#   log10(M_BH/Msun) = (7.45 +/- 0.08) + (1.05 +/- 0.11) * log10(M_*/1e11 Msun)
# Constants are stored in natural-log units of ln(M_BH) (hence the * ln(10) factors).
alpha = 7.45 * np.log(10)
beta = 1.05
d_alpha = 0.08 * np.log(10)
d_beta = 0.11
# Intrinsic scatter epsilon_0 = 0.24 dex (Reines & Volonteri 2015, Sec. 4.1): the true rms of
# log10(M_BH) at fixed M_* once the calibration's virial measurement error (0.50 dex) is removed.
# This is the DOMINANT M_BH-prediction uncertainty; it was previously omitted from BH_mass_error.
sigma_int = 0.24 * np.log(10)


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
        catalog_index: int | None = None,
    ) -> "HostGalaxy":
        """Build a :class:`HostGalaxy` from explicit attribute values.

        Args:
            phiS: Ecliptic azimuthal sky angle (rad).
            qS: Ecliptic polar sky angle (rad, in ``[0, pi]``).
            z: Host redshift.
            z_error: 1-sigma redshift uncertainty.
            M: Source-frame MBH mass (solar masses).
            M_error: 1-sigma MBH mass uncertainty (solar masses).
            catalog_index: Catalog row index. Use ``-1`` to flag an
                out-of-catalog (dark) host that was drawn from the
                missing-galaxy population rather than read off a catalog row
                (see :mod:`master_thesis_code.dark_siren_injection`). Defaults
                to ``None`` (legacy behaviour for synthetic in-memory hosts).

        Returns:
            A :class:`HostGalaxy` whose :attr:`catalog_index` is set to
            ``catalog_index``.
        """
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
        # HostGalaxy.__init__ reads ``catalog_index`` from the Series name, so
        # set the name to thread the requested index (e.g. -1 for a dark host).
        parameters.name = catalog_index
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
    # NB: entries MUST stay in ascending column-value order (pandas read_csv applies
    # `names` to the `usecols` columns in ascending file order in parse_to_reduced_catalog).
    RIGHT_ASCENSION = 8  # in deg
    DECLINATION = 9  # in deg
    # Change 5 (pixelated completeness): apparent B-band magnitude (GLADE+ raw
    # 0-based col 10; NGC4736 = 8.8 verified). Feeds the per-HEALPix-pixel
    # magnitude-threshold completeness estimator (Gray-Messenger-Veitch 2022,
    # arXiv:2111.04629). MUST stay ascending-by-value (usecols/names alignment).
    APPARENT_B_MAG = 10  # in mag (apparent B-band; null for ~25-39% of rows)
    # CMB-frame redshift z_cmb (GLADE+ 0-based col 28). Previously col 27 (z_helio,
    # heliocentric): feeding the heliocentric value into d_L(z; H0) left the solar-motion
    # dipole (v_sun ~ 369.8 km/s) uncorrected in cz = H0*d_L — a coherent H0 systematic
    # (per-event up to +/-2.47% at z~0.05; ~+0.15% net over the detected sample). z_cmb
    # removes the solar dipole and, where GLADE+ flags it (col 29 == 1), is additionally
    # peculiar-velocity corrected, with the PV-correction error in col 30 (added in
    # quadrature below). Ref: Dálya et al. 2022, arXiv:2110.06184. See issue #15.
    REDSHIFT = 28
    # PV-correction flag (GLADE+ raw 0-based col 29): 1 = z_cmb is additionally
    # peculiar-velocity corrected (BORG, z<~0.05 ∩ 2M++ ∩ B-band), 0 = not.
    # Consumed at parse time to resolve the per-class PV width (issue #40b,
    # RATIFIED 2026-07-26); dropped before writing the reduced catalogue.
    REDSHIFT_PECULIAR_VELOCITY_CORRECTION_FLAG = 29
    REDSHIFT_PECULIAR_VELOCITY_ERROR = 30
    REDSHIFT_MEASUREMENT_ERROR = 31
    REDSHIFT_FLAG = 34  # measurement flag: 0=none, 1=PHOTOMETRIC z, 2=lum. distance,
    # 3=SPECTROSCOPIC z (Dálya et al. 2022, arXiv:2110.06184). NB flag 1 ≠ spectroscopic.
    STELLAR_MASS = 35  # in 10^10 solar masses
    STELLAR_MASS_ABSOULTE_ERROR = 36  # in 10^10 solar masses


# In-memory column keys for the loaded catalog. The sky columns use the
# frame-NEUTRAL physics symbols PHI_S/THETA_S, NOT "RIGHT_ASCENSION"/"DECLINATION":
# read_reduced_galaxy_catalog renames the raw equatorial RA/Dec columns to these,
# and _rotate_equatorial_to_ecliptic (COORD-03) rotates them IN PLACE to ecliptic.
# So after handler init these hold ecliptic φ (rad) and polar angle θ ∈ [0, π] (rad),
# BarycentricTrueEcliptic(J2000). The remaining keys below are CatalogueColumns names.
# See .planning/FRAME-AUDIT.md.
class InternalCatalogColumns:
    PHI_S = "PHI_S"  # sky azimuth φ: equatorial RA on read, ecliptic longitude after rotation
    THETA_S = "THETA_S"  # sky polar angle θ ∈ [0, π]; ecliptic colatitude after rotation
    B_MAG = "APPARENT_B_MAG"  # Change 5: apparent B-band magnitude (per-pixel m_th)
    REDSHIFT = "REDSHIFT"
    REDSHIFT_ERROR = "REDSHIFT_MEASUREMENT_ERROR"
    BH_MASS = "STELLAR_MASS"
    BH_MASS_ERROR = "STELLAR_MASS_ABSOULTE_ERROR"
    # GLADE+ redshift measurement flag, RETAINED as the trailing reduced-catalog
    # column (Dálya et al. 2022, arXiv:2110.06184): 1 = PHOTOMETRIC z (σ_z ≈ 0.035),
    # 3 = SPECTROSCOPIC z (σ_z ≈ 0.0017). Only {1, 3} survive the parse filter.
    # Used to split single-event H0 posteriors by host redshift provenance (paper
    # figure F4: photo-z hosts give flat/railing posteriors, spec-z hosts inform).
    REDSHIFT_FLAG = "REDSHIFT_FLAG"


def _reduced_catalog_column_names() -> list[str]:
    """On-disk column names of the headerless reduced-catalog CSV, in file order.

    Single source of truth shared by the writer (``parse_to_reduced_catalog``) and
    every reader (``read_reduced_galaxy_catalog`` and
    ``pixel_completeness.build_m_th_map``). The order is all
    :class:`CatalogueColumns` except the dropped PV-correction flag (raw col 29),
    peculiar-velocity error (raw col 30) and the redshift flag (raw col 34),
    followed by the RETAINED redshift flag as the TRAILING column. Keeping the flag last preserves the historical column
    order of the leading fields so existing positional readers stay aligned.

    Returns:
        Column names as written to / read from disk, e.g.
        ``[RIGHT_ASCENSION, DECLINATION, APPARENT_B_MAG, REDSHIFT,
        REDSHIFT_MEASUREMENT_ERROR, STELLAR_MASS, STELLAR_MASS_ABSOULTE_ERROR,
        REDSHIFT_FLAG]``.
    """
    names = [column.name for column in CatalogueColumns if column.value not in [29, 30, 34]]
    names.append(CatalogueColumns.REDSHIFT_FLAG.name)
    return names


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
        self._angles_mapped_to_ecliptic: bool = False
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

            # GLADE+ redshift/distance measurement flag (Dálya et al. 2022,
            # arXiv:2110.06184, raw col 35): 0 = none, 1 = PHOTOMETRIC redshift,
            # 2 = luminosity distance, 3 = SPECTROSCOPIC redshift. We keep 1 and 3
            # (any measured redshift, excluding distance-only 2).
            # ⚠ CAVEAT: flag 1 (photometric) DOMINATES the catalogue (~62%) and
            # carries a LARGE redshift error (σ_z ≈ 0.035, ~10–18× the EMRI GW
            # redshift precision σ_z^GW ≈ 0.037·z ≈ 0.002 at z≈0.05), whereas flag 3
            # (spectroscopic) has σ_z ≈ 0.0017. The photometric hosts make the
            # in-catalogue H0 likelihood photo-z-DOMINATED, which biases the dark-
            # siren H0 posterior to the grid edge (the seed-600 "railing").
            # See scripts/bridge_closure/BRIDGE-FINDINGS.md and
            # memory h0-railing-rootcause-photoz. Restricting to flag 3 alone is NOT
            # a valid fix on its own (it must be matched by the injection/host draw).
            chunk = chunk[
                (chunk[CatalogueColumns.REDSHIFT_FLAG.name] == 1)
                | (chunk[CatalogueColumns.REDSHIFT_FLAG.name] == 3)
            ]

            # [PHYSICS] Peculiar velocity counted EXACTLY ONCE, per PV-correction
            # class (issue #40b, RATIFIED 2026-07-26;
            # docs/derivations/hostz_pv_photoz_kernel.md §3.1):
            #  - BORG-corrected rows (raw col 29 == 1): the catalogue PV error is
            #    already sigma_tot = sigma_borg (+) sigma_vir (Dálya et al. 2022,
            #    arXiv:2110.06184, §2.2 Eq. 1) — folded in, plus the reconstruction
            #    residual (1+z)*150 km/s / c (Carrick et al. 2015, arXiv:1504.04627,
            #    §4.2.1).
            #  - Uncorrected rows: ONE full-dispersion term (1+z)*500 km/s / c
            #    (Laghi et al. 2021, arXiv:2102.01708, Sec. 3), REPLACING the former
            #    uncited 0.0015 fill.
            # RATIFY-3 check on the raw catalogue (2026-07-27): the flag is 1 /
            # 0 / null; 119,299 rows within the {1,3} parse filter are flagged
            # corrected but report NO PV error (99.7% photometric, median
            # z = 0.055, 374 spec-z). A row therefore counts as corrected only
            # if it is BOTH flagged AND carries a reported sigma_tot; flagged-
            # but-null rows take the conservative full-dispersion term (for the
            # dominant photo-z members the PV term is a ~2% width effect either
            # way). Null-flag rows (~19M, the bulk) are uncorrected by
            # construction.
            # (1+z) factor: Davis et al. (2011), arXiv:1012.2912, Eqs. (1)/(A1).
            # The former inference-time 200 km/s quadrature is retired
            # (SIGMA_V_PEC_KM_S = 0.0 default) — see constants.py.
            _pv_corrected = (
                chunk[CatalogueColumns.REDSHIFT_PECULIAR_VELOCITY_CORRECTION_FLAG.name] == 1
            ) & chunk[CatalogueColumns.REDSHIFT_PECULIAR_VELOCITY_ERROR.name].notna()
            _one_plus_z = 1.0 + chunk[CatalogueColumns.REDSHIFT.name]
            _sigma_z_pv_class = (
                _one_plus_z
                * np.where(
                    _pv_corrected,
                    SIGMA_V_PV_RESIDUAL_CORRECTED_KM_S,
                    SIGMA_V_PV_UNCORRECTED_KM_S,
                )
                / SPEED_OF_LIGHT_KM_S
            )
            _sigma_pv_catalogue = np.where(
                _pv_corrected,
                chunk[CatalogueColumns.REDSHIFT_PECULIAR_VELOCITY_ERROR.name].fillna(0.0),
                0.0,
            )
            chunk[CatalogueColumns.REDSHIFT_MEASUREMENT_ERROR.name] = np.sqrt(
                chunk[CatalogueColumns.REDSHIFT_MEASUREMENT_ERROR.name] ** 2
                + _sigma_pv_catalogue**2
                + _sigma_z_pv_class**2
            )

            # Drop the PV columns (folded into the redshift error above) but
            # RETAIN the redshift flag. Store it as the integer flag
            # (1 = photometric, 3 = spectroscopic) so it round-trips as "1"/"3"
            # rather than "1.0"/"3.0"; the {1, 3} filter above guarantees no NaNs.
            chunk = chunk.drop(
                columns=[
                    CatalogueColumns.REDSHIFT_PECULIAR_VELOCITY_CORRECTION_FLAG.name,
                    CatalogueColumns.REDSHIFT_PECULIAR_VELOCITY_ERROR.name,
                ]
            )
            chunk[CatalogueColumns.REDSHIFT_FLAG.name] = chunk[
                CatalogueColumns.REDSHIFT_FLAG.name
            ].astype(int)

            # Reorder so the retained flag is the TRAILING column, preserving the
            # historical order of the leading fields (positional-reader alignment).
            chunk = chunk[_reduced_catalog_column_names()]

            chunk.to_csv(REDUCED_CATALOGUE_FILE_PATH, header=False, mode="a", index=False)

    def read_reduced_galaxy_catalog(self) -> pd.DataFrame:
        """Load the reduced catalog (RAW equatorial ICRS degrees, pre-rotation).

        The on-disk sky columns are GLADE equatorial RA/Dec (deg). They are renamed
        to the frame-NEUTRAL symbols ``PHI_S``/``THETA_S`` so that the in-place
        equatorial→ecliptic rotation (``_rotate_equatorial_to_ecliptic``, COORD-03)
        does not leave a column literally named "RIGHT_ASCENSION" holding an ecliptic
        longitude. After that rotation these columns hold ecliptic φ (rad) and polar
        angle θ ∈ [0, π] (rad). See .planning/FRAME-AUDIT.md.
        """
        catalog = pd.read_csv(
            REDUCED_CATALOGUE_FILE_PATH,
            names=_reduced_catalog_column_names(),
        )
        return catalog.rename(
            columns={
                CatalogueColumns.RIGHT_ASCENSION.name: InternalCatalogColumns.PHI_S,
                CatalogueColumns.DECLINATION.name: InternalCatalogColumns.THETA_S,
            }
        )

    def setup_galaxy_catalog_balltree(self) -> None:
        # expects the reduced galaxy catalog to be setup already. The PHI_S/THETA_S
        # columns hold ECLIPTIC angles after _rotate_equatorial_to_ecliptic (COORD-03):
        # θ_polar ∈ [0, π], φ ∈ [0, 2π). See .planning/FRAME-AUDIT.md.
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
                Must be in BarycentricTrueEcliptic J2000 — the same frame as the
                BallTree (see migrate_crb_to_ecliptic.py).  phi_sigma and
                cov_theta_phi must also be in this ecliptic frame (_cov_frame guard
                in Detection.__init__ enforces this at load time).
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

    def draw_uniform_hosts(
        self,
        number_of_hosts: int,
        rng: np.random.Generator,
        z_max: float = HOST_DRAW_Z_MAX,
    ) -> list[HostGalaxy]:
        """Draw host galaxies uniformly at random from the in-catalog volume z < z_max.

        This is the self-consistent generative model for the equal-weight in-catalog
        likelihood term used by the current dark-siren inference: every catalog galaxy
        with redshift below ``z_max`` is an equally probable host, i.e. P(g) = const
        over the ``z < z_max`` catalog. Each returned :class:`HostGalaxy` carries the
        redshift, sky angles, BH mass, and per-quantity errors straight from its catalog
        row — there is NO nearest-neighbour snap (contrast
        :meth:`find_closest_galaxy_to_coordinates`) and NO overwrite of the catalog
        quantities.

        Sampling is WITH REPLACEMENT: hosts are i.i.d. draws from the uniform
        distribution over the eligible rows, so the same galaxy may be returned more
        than once. The truncation ``z < z_max`` is a deliberate population-depth
        choice matching ``Model1CrossCheck.max_redshift`` (issue #20): after the
        dt² fix the EMRI horizon reaches z ~ 1.5+, so p_det is NOT assumed zero
        beyond ``z_max`` — the injection-campaign ``z_cut`` derives from the same
        constant so draw, injections, and inference share one depth.

        Args:
            number_of_hosts: Number of host galaxies to draw (i.i.d., with replacement).
            rng: Seeded random generator. Threading the simulation-wide ``rng`` makes
                the host selection reproducible under ``--seed``.
            z_max: Exclusive upper redshift bound for eligible hosts. Defaults to
                :data:`~master_thesis_code.constants.HOST_DRAW_Z_MAX`.

        Returns:
            ``number_of_hosts`` hosts, each built exactly like
            :meth:`find_closest_galaxy_to_coordinates` builds its result (so z / sky /
            M / errors come straight from the catalog row).

        Raises:
            ValueError: If no catalog galaxy satisfies ``z < z_max``.

        References:
            Chen, Fishbach & Holz, "A Hitchhiker's Guide to ...", arXiv:2212.08694
            Eq. (9): equal-weight in-catalog term P(g) = const.
        """
        # Eq. (9) in Chen et al. (2024), arXiv:2212.08694: equal-weight in-catalog
        # term P(g) = const over the z < z_max catalog (research option A).
        eligible_catalog = self.reduced_galaxy_catalog[
            self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT] < z_max
        ]
        n_eligible = len(eligible_catalog)
        if n_eligible == 0:
            raise ValueError(
                f"No galaxy in the reduced catalog has redshift < z_max = {z_max}; "
                "cannot draw uniform in-catalog hosts."
            )

        # Uniform i.i.d. draw WITH REPLACEMENT over the eligible rows: each eligible
        # galaxy carries probability 1 / n_eligible (= const), the generative model
        # for the equal-weight in-catalog likelihood. Positional integers index the
        # eligible subset so each HostGalaxy is built from a genuine catalog row,
        # exactly as find_closest_galaxy_to_coordinates does (no snap, no overwrite).
        positions: npt.NDArray[np.int64] = rng.integers(0, n_eligible, size=number_of_hosts)
        return [HostGalaxy(eligible_catalog.iloc[int(position)]) for position in positions]

    def draw_rate_weighted_hosts(
        self,
        number_of_hosts: int,
        rng: np.random.Generator,
        z_max: float = HOST_DRAW_Z_MAX,
    ) -> list[HostGalaxy]:
        r"""Draw in-catalog hosts with probability ∝ the per-MBH EMRI-rate weight.

        This is the rate-weighted generative model (research "version B") that
        supersedes the equal-weight draw of :meth:`draw_uniform_hosts`. Every
        catalog galaxy with redshift below ``z_max`` is a candidate host, but the
        selection probability is now proportional to the per-MBH EMRI-rate weight

        .. math::

            P(g) \propto w(g) = \frac{R_\mathrm{eff}(M_g)}{1 + z_g},

        where :math:`R_\mathrm{eff}(M_g)` is the *effective per-MBH* EMRI rate
        (:func:`master_thesis_code.emri_rate.R_eff_per_mbh`; Babak et al. 2017,
        Eqs. 23, 26-27, 30-31, 34) evaluated at the SOURCE-FRAME catalog BH mass
        ``M_g`` — the exact column :attr:`HostGalaxy.M` reads — and ``1/(1+z_g)``
        is the source-to-detector time dilation. The *per-MBH* rate (NOT the
        comoving volume density :func:`~master_thesis_code.emri_rate.R_EMRI`) is
        the correct weight here because each catalog galaxy is ONE realised MBH:
        the mass function ``dn/dlog10 M`` is already sampled by the catalog
        itself, so only the per-object rate shape and the mild redshift dilation
        reweight the hosts. The overall normalization (including
        ``emri_rate.C_NORM``) cancels in ``p = w / Σ w`` and is irrelevant.

        The SAME weight ``w(g)`` reweights the in-catalog likelihood term of the
        Bayesian inference (``bayesian_statistics.p_Di``), so the draw and the
        inference share one population model (self-consistency).

        Sampling is WITH REPLACEMENT via :meth:`numpy.random.Generator.choice`
        with the normalized weights, so the same galaxy may be returned more than
        once. As in :meth:`draw_uniform_hosts`, each returned :class:`HostGalaxy`
        carries z / sky / M / errors straight from its catalog row — there is NO
        nearest-neighbour snap and NO overwrite of catalog quantities. The
        truncation ``z < z_max`` is a deliberate population-depth choice
        matching ``Model1CrossCheck.max_redshift`` (issue #20; see
        :meth:`draw_uniform_hosts` for the shared-depth rationale).

        Args:
            number_of_hosts: Number of host galaxies to draw (i.i.d., with
                replacement).
            rng: Seeded random generator. Threading the simulation-wide ``rng``
                makes the rate-weighted host selection reproducible under
                ``--seed``.
            z_max: Exclusive upper redshift bound for eligible hosts. Defaults to
                :data:`~master_thesis_code.constants.HOST_DRAW_Z_MAX`.

        Returns:
            ``number_of_hosts`` hosts, each built exactly like
            :meth:`draw_uniform_hosts` (z / sky / M / errors come straight from
            the catalog row), drawn with probability
            ``P(g) ∝ R_eff_per_mbh(M_g) / (1 + z_g)``.

        Raises:
            ValueError: If no catalog galaxy satisfies ``z < z_max``, or if the
                total weight ``Σ w(g)`` over the eligible rows is non-positive.

        References:
            Babak et al. (2017), arXiv:1703.09722, Eqs. (23), (26)-(27),
                (30)-(31), (34) — effective per-MBH EMRI rate ``R_eff(M)`` (see
                :mod:`master_thesis_code.emri_rate`).
            Gray et al. (2020), arXiv:1908.06050 — galaxy weighting of the
                in-catalog dark-siren likelihood by an astrophysical rate prior.
        """
        # w(g) = R_eff_per_mbh(M_g) / (1 + z_g): per-MBH effective EMRI rate
        # (Babak et al. 2017, arXiv:1703.09722) × source-frame time dilation. The
        # IDENTICAL weight reweights the inference in-catalog term
        # (bayesian_statistics.p_Di). Gray et al. (2020), arXiv:1908.06050.
        eligible_catalog = self.reduced_galaxy_catalog[
            self.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT] < z_max
        ]
        n_eligible = len(eligible_catalog)
        if n_eligible == 0:
            raise ValueError(
                f"No galaxy in the reduced catalog has redshift < z_max = {z_max}; "
                "cannot draw rate-weighted in-catalog hosts."
            )

        # SOURCE-FRAME catalog BH mass (the column HostGalaxy.M reads) and catalog
        # redshift. R_eff_per_mbh is strictly positive, so all weights are positive
        # for z >= 0; the non-positive-total guard below is defensive.
        masses = eligible_catalog[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64)
        redshifts = eligible_catalog[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64)
        weights: npt.NDArray[np.float64] = np.asarray(R_eff_per_mbh(masses), dtype=np.float64) / (
            1.0 + redshifts
        )
        total_weight = float(weights.sum())
        if not (total_weight > 0.0):
            raise ValueError(
                "Total rate weight Σ w(g) over the eligible catalog is non-positive "
                f"({total_weight}); cannot form the host-selection probability."
            )

        # P(g) = w(g) / Σ w(g); draw WITH REPLACEMENT. Positional integers index the
        # eligible subset so each HostGalaxy is built from a genuine catalog row,
        # exactly as draw_uniform_hosts / find_closest_galaxy_to_coordinates do
        # (no snap, no overwrite).
        probabilities: npt.NDArray[np.float64] = weights / total_weight
        positions: npt.NDArray[np.int64] = rng.choice(
            n_eligible, size=number_of_hosts, replace=True, p=probabilities
        )
        return [HostGalaxy(eligible_catalog.iloc[int(position)]) for position in positions]

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
        assert not self._angles_mapped_to_ecliptic, (
            "_map_angles_to_spherical_coordinates called twice — "
            "angles are already in ecliptic polar frame"
        )
        self._angles_mapped_to_ecliptic = True
        self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S] = (
            self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S] * np.pi / 180
        )
        self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S] = (
            self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S] * np.pi / 180 - np.pi / 2
        ) * (-1)

    # NOTE (redshifted-mass convention, Design B): the catalog BH masses remain
    # SOURCE-frame. The detector-frame lift M_z = M_source·(1+z) is applied once,
    # at injection time (parameter_space.set_host_galaxy_parameters and
    # main.py:injection_campaign), NOT to the whole catalog. The former
    # `_map_BH_masses_to_redshifted_masses` catalog-lift helper was dead code and
    # was removed: wiring it in would double-lift, since the inference already
    # lifts each candidate host by (1+z) (bayesian_statistics.py:1335).

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


def _empiric_stellar_mass_to_BH_mass_relation(
    stellar_mass: float, stellar_mass_error: float
) -> tuple[float, float]:
    BH_mass = np.exp(alpha + beta * np.log(stellar_mass / 10))
    # Error budget in ln(M_BH): intrinsic scatter (DOMINANT) + fit-parameter uncertainties +
    # propagated stellar-mass error. Reines & Volonteri (2015), arXiv:1508.06274, Sec. 4.1.
    # d(ln M_BH)/d(M_*) = beta / M_* -- the 1e11 pivot is a constant, so the previous extra
    # "/ 10" on the stellar-mass term was a bug (understated that term by 100x in variance).
    BH_mass_error = BH_mass * np.sqrt(
        sigma_int**2
        + d_alpha**2
        + (np.log(stellar_mass / 10) * d_beta) ** 2
        + (beta / stellar_mass * stellar_mass_error) ** 2
    )
    return (BH_mass, BH_mass_error)


def _empiric_MBH_to_M_stellar_relation(MBH_mass: float, MBH_mass_error: float) -> list:
    stellar_mass = 10 * np.exp((np.log(MBH_mass) - alpha) / beta)
    # Inverse error budget in ln(M_*): from ln(M_*) = ln(10) + (ln(M_BH) - alpha)/beta,
    # d(ln M_*)/d(ln M_BH) = 1/beta (the M_BH-error term is /beta, NOT *beta as before), and the
    # intrinsic scatter propagates as sigma_int/beta. Reines & Volonteri (2015), arXiv:1508.06274.
    # NOTE: this inverse is currently unused (no call sites); fixed for correctness.
    stellar_mass_error = stellar_mass * np.sqrt(
        (sigma_int / beta) ** 2
        + (d_alpha / beta) ** 2
        + (MBH_mass_error / (MBH_mass * beta)) ** 2
        + ((np.log(MBH_mass) - alpha) / beta**2) ** 2 * d_beta**2
    )
    return [stellar_mass, stellar_mass_error]
