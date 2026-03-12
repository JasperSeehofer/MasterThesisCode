"""Galaxy and GalaxyCatalog datamodels for Bayesian H₀ inference."""

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import emcee
import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.stats import rv_continuous, truncnorm

from master_thesis_code.constants import (
    FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR,
    GALAXY_REDSHIFT_ERROR_COEFFICIENT,
    LUMINOSITY_DISTANCE_THRESHOLD_GPC,
    OMEGA_M,
    TRUE_HUBBLE_CONSTANT,
)
from master_thesis_code.constants import (
    OMEGA_DE as OMEGA_LAMBDA,
)
from master_thesis_code.constants import (
    SPEED_OF_LIGHT_KM_S as SPEED_OF_LIGHT,
)
from master_thesis_code.physical_relations import dist


@dataclass(unsafe_hash=True)
class Galaxy:
    """A galaxy in the catalog with a central massive black hole.

    Attributes:
        redshift: Spectroscopic redshift :math:`z` (dimensionless).
        central_black_hole_mass: Central MBH mass in solar masses :math:`M_\\odot`.
        right_ascension: Ecliptic azimuthal angle in radians :math:`[0, 2\\pi)`.
        declination: Ecliptic polar angle in radians :math:`[0, \\pi]`.
    """

    redshift: float  # dimensionless
    central_black_hole_mass: float  # M_sun (massive black hole mass)
    right_ascension: float  # rad, ecliptic azimuthal angle
    declination: float  # rad, ecliptic polar angle

    @classmethod
    def with_random_skylocalization(
        cls,
        redshift: float,
        central_black_hole_mass: float,
        rng: np.random.Generator | None = None,
    ) -> "Galaxy":
        if rng is None:
            rng = np.random.default_rng()
        # get spherically uniform distributed sky localization
        right_ascension = float(rng.uniform(0, 2 * np.pi))
        declination = float(np.arccos(rng.uniform(-1, 1)))
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
    """Synthetic galaxy catalog for Bayesian H₀ inference.

    Holds a list of :class:`Galaxy` objects and provides redshift/mass probability
    distributions used by :class:`~master_thesis_code.bayesian_inference.bayesian_inference.BayesianInference`.

    Attributes:
        lower_mass_limit: Minimum central BH mass in :math:`M_\\odot`.
        upper_mass_limit: Maximum central BH mass in :math:`M_\\odot`.
        redshift_lower_limit: Minimum redshift in the catalog.
        redshift_upper_limit: Maximum redshift in the catalog.
        catalog: List of :class:`Galaxy` instances.
    """

    _use_truncnorm: bool
    _use_comoving_volume: bool

    lower_mass_limit: float = 10 ** (4)
    upper_mass_limit: float = 10 ** (7)
    redshift_lower_limit: float = 0.00001
    redshift_upper_limit: float = 0.55
    catalog: list[Galaxy] = field(default_factory=list)
    galaxy_distribution: list[NormalDist | rv_continuous] = field(default_factory=list)
    galaxy_mass_distribution: list[NormalDist | rv_continuous] = field(default_factory=list)

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
        self._comoving_volume_element_spline = self._build_comoving_volume_element_spline(h0)

    @staticmethod
    def _build_comoving_volume_element_spline(h0: float = TRUE_HUBBLE_CONSTANT) -> CubicSpline:
        """Precompute the comoving volume element dV_c/dz on a fine redshift grid.

        The integral I(z) = integral_0^z dz'/E(z') is computed once via cumulative_trapezoid so
        subsequent calls to comoving_volume_element() are O(log n) spline interpolation.
        """
        _z_grid = np.linspace(0, 10.0, 4000)
        e_z = np.sqrt(OMEGA_M * (1 + _z_grid) ** 3 + OMEGA_LAMBDA)
        integrand = 1.0 / e_z
        cumulative_integral = np.concatenate([[0.0], cumulative_trapezoid(integrand, _z_grid)])
        # Eq. 27 in Hogg (1999), arXiv:astro-ph/9905116
        cv_grid = 4 * np.pi * (SPEED_OF_LIGHT / h0) ** 3 * cumulative_integral**2 / e_z
        return CubicSpline(_z_grid, cv_grid)

    def comoving_volume_element(self, redshift: float) -> float:
        """Comoving volume element dV_c/dz at the given redshift."""
        return float(self._comoving_volume_element_spline(redshift))

    def log_comoving_volume_element(self, redshift: float) -> float:
        try:
            redshift = redshift[0]  # type: ignore[index]
        except TypeError:
            pass
        if redshift < self.redshift_lower_limit or redshift > self.redshift_upper_limit:
            return float(-np.inf)
        return float(np.log(self.comoving_volume_element(redshift)))

    def get_samples_from_comoving_volume_element(
        self, number_of_samples: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # use emcee to sample the comoving volume distribution
        ndim = 1
        nwalkers = 5
        p0 = rng.uniform(self.redshift_lower_limit, self.redshift_upper_limit, (nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_comoving_volume_element)
        # burn-in
        p0, _, _ = sampler.run_mcmc(p0, 1000)
        sampler.reset()
        sampler.run_mcmc(p0, int(number_of_samples / nwalkers) + 1)
        samples = np.array(sampler.get_chain(flat=True)).flatten()
        # return required number of samples by using the last n samples
        return samples[-number_of_samples:]

    def create_random_catalog(
        self, number_of_galaxies: int, rng: np.random.Generator | None = None
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        # draw mass from uniform in log space
        print(
            f"Creating random galaxy catalog with {number_of_galaxies} galaxies in the redshift range ({self.redshift_lower_limit}, {self.redshift_upper_limit}) / ({dist(self.redshift_lower_limit)}, {dist(self.redshift_upper_limit)}) Gpc."
        )
        if self._use_comoving_volume:
            redshift_samples = self.get_samples_from_comoving_volume_element(
                number_of_galaxies, rng=rng
            )
            assert len(redshift_samples) == number_of_galaxies
            for redshift in redshift_samples:
                self.catalog.append(
                    Galaxy.with_random_skylocalization(
                        redshift=redshift,
                        central_black_hole_mass=10
                        ** float(
                            rng.uniform(
                                np.log10(self.lower_mass_limit),
                                np.log10(self.upper_mass_limit),
                            )
                        ),
                        rng=rng,
                    )
                )
        else:
            for i in range(number_of_galaxies):
                self.catalog.append(
                    Galaxy.with_random_skylocalization(
                        redshift=float(
                            rng.uniform(self.redshift_lower_limit, self.redshift_upper_limit)
                        ),
                        central_black_hole_mass=10
                        ** float(
                            rng.uniform(
                                np.log10(self.lower_mass_limit),
                                np.log10(self.upper_mass_limit),
                            )
                        ),
                        rng=rng,
                    )
                )
        self.setup_galaxy_distribution()
        self.setup_galaxy_mass_distribution()

    def remove_all_galaxies(self) -> None:
        self.catalog = []
        self.galaxy_distribution = []
        self.galaxy_mass_distribution = []

    def add_random_galaxy(self, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        galaxy = Galaxy.with_random_skylocalization(
            redshift=float(rng.uniform(self.redshift_lower_limit, self.redshift_upper_limit)),
            central_black_hole_mass=float(
                rng.uniform(self.lower_mass_limit, self.upper_mass_limit)
            ),
            rng=rng,
        )
        self.catalog.append(galaxy)
        self.append_galaxy_to_galaxy_distribution(galaxy)
        self.append_galaxy_to_galaxy_mass_distribution(galaxy)

    def add_host_galaxy(self, rng: np.random.Generator | None = None) -> Galaxy:
        if rng is None:
            rng = np.random.default_rng()
        galaxy = Galaxy.with_random_skylocalization(
            redshift=float(rng.uniform(self.redshift_lower_limit, self.redshift_upper_limit)),
            central_black_hole_mass=float(
                rng.uniform(self.lower_mass_limit, self.upper_mass_limit)
            ),
            rng=rng,
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
                    loc=galaxy.redshift,
                    scale=galaxy.redshift_uncertainty,
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
                    loc=galaxy.central_black_hole_mass,
                    scale=galaxy.central_black_hole_mass_uncertainty,
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
                    loc=galaxy.redshift,
                    scale=galaxy.redshift_uncertainty,
                )
            )

    def evaluate_galaxy_distribution(self, redshift: float) -> npt.NDArray[np.float64]:
        redshift_uncertainty = min(GALAXY_REDSHIFT_ERROR_COEFFICIENT * (1 + redshift) ** 3, 0.015)
        p_background: float = 1.0

        if self._use_comoving_volume:
            p_background = self.comoving_volume_element(redshift)

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

    def setup_galaxy_mass_distribution(self) -> None:
        if not self._use_truncnorm:
            self.galaxy_mass_distribution = [
                NormalDist(
                    mu=galaxy.central_black_hole_mass,
                    # sigma(M) = f * M -- fractional uncertainty, consistent with append_galaxy_to_galaxy_mass_distribution
                    sigma=FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR * galaxy.central_black_hole_mass,
                )
                for galaxy in self.catalog
            ]
        else:
            self.galaxy_mass_distribution = [
                truncnorm(
                    a=(self.lower_mass_limit - galaxy.central_black_hole_mass)
                    / galaxy.central_black_hole_mass_uncertainty,
                    b=(self.upper_mass_limit - galaxy.central_black_hole_mass)
                    / galaxy.central_black_hole_mass_uncertainty,
                    loc=galaxy.central_black_hole_mass,
                    scale=galaxy.central_black_hole_mass_uncertainty,
                )
                for galaxy in self.catalog
            ]

    def evaluate_galaxy_mass_distribution(self, mass: float) -> npt.NDArray[np.float64]:
        # use truncated normal distribution

        if self._use_truncnorm:
            # scipy.stats.truncnorm.pdf() is already normalized over its support
            return np.array(
                [distribution.pdf(mass) for distribution in self.galaxy_mass_distribution]
            ) / len(self.catalog)

        # without truncnorm
        return np.array(
            [distribution.pdf(mass) for distribution in self.galaxy_mass_distribution]
        ) / len(self.catalog)

    def get_possible_host_galaxies(self) -> list[Galaxy]:
        return [
            galaxy
            for galaxy in self.catalog
            if dist(galaxy.redshift, TRUE_HUBBLE_CONSTANT) <= LUMINOSITY_DISTANCE_THRESHOLD_GPC
        ]

    def get_unique_host_galaxies_from_catalog(
        self, number_of_host_galaxies: int, rng: np.random.Generator | None = None
    ) -> list[Galaxy]:
        if rng is None:
            rng = np.random.default_rng()
        if len(self.get_possible_host_galaxies()) < number_of_host_galaxies:
            print("Not enough possible host galaxies in catalog.")
            return []
        possible: list[Any] = self.get_possible_host_galaxies()
        indices = rng.choice(len(possible), number_of_host_galaxies, replace=False)
        return [possible[i] for i in indices]

    def add_unique_host_galaxies_from_catalog(
        self,
        number_of_host_galaxies_to_add: int,
        used_host_galaxies: list[Galaxy],
        rng: np.random.Generator | None = None,
    ) -> list[Galaxy]:
        if rng is None:
            rng = np.random.default_rng()
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
        indices = rng.choice(len(filtered), number_of_host_galaxies_to_add, replace=False)
        new_host_galaxies = [filtered[i] for i in indices]
        used_host_galaxies.extend(new_host_galaxies)
        return used_host_galaxies
