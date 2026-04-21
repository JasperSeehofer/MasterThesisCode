"""Fixtures for coordinate-frame round-trip tests (Phase 35 + Phase 36).

Exports:
- ``synthetic_catalog_builder(n, sky_band, seed, already_rotated)``: build an
  in-memory ``GalaxyCatalogueHandler``-compatible DataFrame with controlled
  sky positions. Columns match ``InternalCatalogColumns`` verbatim
  (``"RIGHT_ASCENSION"``, ``"DECLINATION"``, ``"REDSHIFT"``,
  ``"REDSHIFT_MEASUREMENT_ERROR"``, ``"STELLAR_MASS"``,
  ``"STELLAR_MASS_ABSOULTE_ERROR"``).
- ``equatorial_to_ecliptic_astropy(ra_deg, dec_deg)``: ground-truth astropy
  wrapper returning ``(lambda_rad, beta_rad, theta_polar_rad)`` as a
  ``NamedTuple`` for clean unpacking.
- ``build_balltree(catalog)``: BallTree builder that delegates to
  ``GalaxyCatalogueHandler.setup_galaxy_catalog_balltree`` via an
  ``object.__new__`` shim (per repo idiom in
  ``test_catalog_only_diagnostic.py:37-73``).

Consumed by:
- ``master_thesis_code_test/test_coordinate_roundtrip.py`` (Phase 35 RED tests)
- Phase 36 post-fix regression tests (per ``.planning/.../35-CONTEXT.md`` D-09)

Run as a script to self-check (no side effects; imports only):
    uv run python -m master_thesis_code_test.fixtures.coordinate
"""

from typing import NamedTuple

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord
from sklearn.neighbors import BallTree

from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
)


class EclipticCoords(NamedTuple):
    """Astropy ground-truth ecliptic coordinates for a single equatorial input.

    Attributes:
        lambda_rad: Ecliptic longitude in radians (``ecl.lon``), range ``[0, 2*pi)``.
        beta_rad: Ecliptic latitude in radians (``ecl.lat``), range ``[-pi/2, pi/2]``.
        theta_polar_rad: Polar angle (``pi/2 - beta_rad``), range ``[0, pi]``.
    """

    lambda_rad: float
    beta_rad: float
    theta_polar_rad: float


def equatorial_to_ecliptic_astropy(ra_deg: float, dec_deg: float) -> EclipticCoords:
    """Convert J2000 equatorial (RA, Dec) to barycentric-true-ecliptic (lambda, beta).

    Uses ``astropy.coordinates.SkyCoord.transform_to(BarycentricTrueEcliptic(equinox='J2000'))``
    as the authoritative conversion. Returns both the ecliptic latitude
    (``beta``, measured from the ecliptic equator) and the polar angle
    (``theta_polar = pi/2 - beta``, measured from the north ecliptic pole)
    that the rest of the pipeline uses.

    Args:
        ra_deg: Right ascension in degrees (ICRS J2000).
        dec_deg: Declination in degrees (ICRS J2000).

    Returns:
        ``EclipticCoords(lambda_rad, beta_rad, theta_polar_rad)``.
    """
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ecl = coord.transform_to(BarycentricTrueEcliptic(equinox="J2000"))
    lambda_rad = float(ecl.lon.to(u.rad).value)
    beta_rad = float(ecl.lat.to(u.rad).value)
    theta_polar_rad = float(np.pi / 2 - beta_rad)
    return EclipticCoords(lambda_rad, beta_rad, theta_polar_rad)


def synthetic_catalog_builder(
    n: int,
    sky_band: str,
    *,
    seed: int = 42,
    already_rotated: bool = False,
) -> pd.DataFrame:
    """Build an in-memory catalog DataFrame for coordinate-frame tests.

    Columns exactly match ``InternalCatalogColumns``:
    ``"RIGHT_ASCENSION"``, ``"DECLINATION"``, ``"REDSHIFT"``,
    ``"REDSHIFT_MEASUREMENT_ERROR"``, ``"STELLAR_MASS"``,
    ``"STELLAR_MASS_ABSOULTE_ERROR"``.

    The ``already_rotated`` knob decides which stage of the pipeline
    the fixture injects into (per PATTERNS.md §"Caveat — the synthetic
    builder must choose a stage"):

    - ``already_rotated=False`` (default): positions in DEGREES,
      equatorial (RA/Dec). Test exercises the missing rotation inside
      ``_map_angles_to_spherical_coordinates`` (handler.py:486-492).
    - ``already_rotated=True``: positions in RADIANS, ecliptic
      (``phi=lambda``, ``theta=polar``). Test isolates the BallTree
      embedding bug by skipping the rotation stage.

    Args:
        n: Number of synthetic galaxies to produce.
        sky_band: One of ``"ecliptic_equator"``, ``"north_pole"``,
            ``"uniform"``. Determines how positions are sampled:
            - ``"ecliptic_equator"``: beta=0 (post-rotation theta=pi/2);
              longitudes drawn uniformly from [0, 2*pi).
            - ``"north_pole"``: beta=pi/2 (NCP in ecliptic frame);
              longitudes irrelevant, set to 0.
            - ``"uniform"``: isotropic cos(beta) sampling.
        seed: RNG seed (default 42, matches D-07).
        already_rotated: See discussion above.

    Returns:
        A ``pandas.DataFrame`` with ``n`` rows and the six columns listed
        above. Non-sky columns are filled with plausible dummy values
        (redshift ~ 0.1, mass ~ 1e10) that are not the focus of the test.
    """
    rng = np.random.default_rng(seed)
    beta: npt.NDArray[np.float64]
    lambda_rad_vals: npt.NDArray[np.float64]

    if sky_band == "ecliptic_equator":
        beta = np.zeros(n)
        lambda_rad_vals = rng.uniform(0.0, 2 * np.pi, size=n)
    elif sky_band == "north_pole":
        beta = np.full(n, np.pi / 2)
        lambda_rad_vals = np.zeros(n)
    elif sky_band == "uniform":
        # isotropic cos(beta) sampling
        beta = np.arcsin(rng.uniform(-1.0, 1.0, size=n))
        lambda_rad_vals = rng.uniform(0.0, 2 * np.pi, size=n)
    else:
        raise ValueError(
            f"Unknown sky_band {sky_band!r}; must be one of "
            "'ecliptic_equator', 'north_pole', 'uniform'."
        )

    if already_rotated:
        # Store ecliptic (phi, theta_polar) in RADIANS as the pipeline
        # expects post-_map_angles_to_spherical_coordinates.
        phi_vals: npt.NDArray[np.float64] = lambda_rad_vals
        theta_polar_vals: npt.NDArray[np.float64] = np.pi / 2 - beta
        df = pd.DataFrame(
            {
                InternalCatalogColumns.PHI_S: phi_vals,
                InternalCatalogColumns.THETA_S: theta_polar_vals,
                InternalCatalogColumns.REDSHIFT: np.full(n, 0.1),
                InternalCatalogColumns.REDSHIFT_ERROR: np.full(n, 0.01),
                InternalCatalogColumns.BH_MASS: np.full(n, 1.0e10),
                InternalCatalogColumns.BH_MASS_ERROR: np.full(n, 1.0e9),
            }
        )
    else:
        # Store equatorial (RA, Dec) in DEGREES as GLADE delivers them.
        # Caller must drive _map_angles_to_spherical_coordinates before use.
        # Note: for this fixture path we interpret (lambda, beta) as
        # already-ecliptic degrees; real Phase 36 fix will rotate from
        # equatorial properly. For Phase 35 RED tests, caller passes a
        # DataFrame with RA/Dec in degrees built directly with the knob
        # bypassed (see test file for direct construction).
        df = pd.DataFrame(
            {
                InternalCatalogColumns.PHI_S: np.rad2deg(lambda_rad_vals),
                InternalCatalogColumns.THETA_S: np.rad2deg(beta),
                InternalCatalogColumns.REDSHIFT: np.full(n, 0.1),
                InternalCatalogColumns.REDSHIFT_ERROR: np.full(n, 0.01),
                InternalCatalogColumns.BH_MASS: np.full(n, 1.0e10),
                InternalCatalogColumns.BH_MASS_ERROR: np.full(n, 1.0e9),
            }
        )
    return df


def build_balltree(catalog: pd.DataFrame) -> BallTree:
    """Build a ``BallTree`` from an in-memory catalog DataFrame.

    Delegates to ``GalaxyCatalogueHandler.setup_galaxy_catalog_balltree``
    via an ``object.__new__`` shim so the real production embedding is
    exercised (buggy or fixed). The shim bypasses the heavy ``__init__``
    that reads GLADE off disk.

    Args:
        catalog: DataFrame with at least ``"RIGHT_ASCENSION"`` and
            ``"DECLINATION"`` columns in radians (ecliptic phi/theta_polar
            if ``already_rotated=True`` upstream, otherwise still-equatorial
            degrees passed through ``_map_angles_to_spherical_coordinates``).

    Returns:
        The constructed ``BallTree``. Also available on the shim instance
        as ``catalog_ball_tree`` for callers that want to reuse the
        instance.
    """
    instance = object.__new__(GalaxyCatalogueHandler)
    instance.reduced_galaxy_catalog = catalog
    instance.setup_galaxy_catalog_balltree()
    return instance.catalog_ball_tree


__all__ = [
    "EclipticCoords",
    "equatorial_to_ecliptic_astropy",
    "synthetic_catalog_builder",
    "build_balltree",
]
