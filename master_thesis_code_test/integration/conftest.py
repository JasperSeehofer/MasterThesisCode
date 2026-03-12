"""Fixtures for evaluation pipeline integration tests."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
)
from master_thesis_code.physical_relations import dist_to_redshift

if TYPE_CHECKING:
    from master_thesis_code.cosmological_model import Model1CrossCheck

# Detected event specs (must match generate_fixtures.py)
_DETECTED_SPECS = [
    {"z": 0.05, "M_z": 5e4, "phiS": 1.0, "qS": 0.8},
    {"z": 0.10, "M_z": 1e5, "phiS": 2.0, "qS": 1.2},
    {"z": 0.15, "M_z": 2e5, "phiS": 3.0, "qS": 0.5},
    {"z": 0.20, "M_z": 3e5, "phiS": 4.0, "qS": 1.5},
    {"z": 0.25, "M_z": 5e5, "phiS": 5.0, "qS": 2.0},
]


def _redshift_for_distance(d_L: float) -> float:
    """Invert dist() to get the redshift for a given luminosity distance."""
    return float(dist_to_redshift(d_L))


def _build_galaxy_rows(n_detections: int = 5) -> list[dict]:
    """Build synthetic galaxy catalog rows near the first *n_detections* events.

    Creates 8 galaxies per detection:
    - 5 with matching BH mass (pass both redshift and mass filters)
    - 3 with mismatched BH mass (pass redshift filter only)

    The mass-mismatched galaxies ensure that ``possible_host_galaxies_reduced``
    in ``p_Di()`` is non-empty, which is required for the likelihood computation
    to produce nonzero results.
    """
    rng = np.random.default_rng(seed=99)
    rows: list[dict] = []
    for spec in _DETECTED_SPECS[:n_detections]:
        z = spec["z"]
        # Rest-frame mass = redshifted mass / (1+z)
        M_rest = spec["M_z"] / (1 + z)
        phi = spec["phiS"]
        theta = spec["qS"]

        # 5 galaxies with matching BH mass
        for j in range(5):
            dz = rng.normal(0, 0.005)
            dphi = rng.normal(0, 0.003)
            dtheta = rng.normal(0, 0.003)
            dM_frac = rng.normal(0, 0.05)

            galaxy_z = max(0.001, z + dz)
            galaxy_phi = (phi + dphi) % (2 * np.pi)
            galaxy_theta = np.clip(theta + dtheta, 0.01, np.pi - 0.01)
            galaxy_M = M_rest * (1 + dM_frac)
            galaxy_M = np.clip(galaxy_M, 1e4 + 1, 1e6 - 1)

            rows.append(
                {
                    InternalCatalogColumns.PHI_S: galaxy_phi,
                    InternalCatalogColumns.THETA_S: galaxy_theta,
                    InternalCatalogColumns.REDSHIFT: galaxy_z,
                    InternalCatalogColumns.REDSHIFT_ERROR: 0.002,
                    InternalCatalogColumns.BH_MASS: galaxy_M,
                    InternalCatalogColumns.BH_MASS_ERROR: galaxy_M * 0.1,
                }
            )

        # 3 galaxies with mismatched BH mass (far from detection M_z)
        # These pass the redshift filter but NOT the mass filter,
        # populating possible_host_galaxies_reduced in p_Di().
        for j in range(3):
            dz = rng.normal(0, 0.005)
            dphi = rng.normal(0, 0.003)
            dtheta = rng.normal(0, 0.003)

            galaxy_z = max(0.001, z + dz)
            galaxy_phi = (phi + dphi) % (2 * np.pi)
            galaxy_theta = np.clip(theta + dtheta, 0.01, np.pi - 0.01)
            # Mass far from the detection's M_z so it fails the mass filter
            galaxy_M = np.clip(M_rest * 0.1, 1e4 + 1, 1e6 - 1)

            rows.append(
                {
                    InternalCatalogColumns.PHI_S: galaxy_phi,
                    InternalCatalogColumns.THETA_S: galaxy_theta,
                    InternalCatalogColumns.REDSHIFT: galaxy_z,
                    InternalCatalogColumns.REDSHIFT_ERROR: 0.002,
                    InternalCatalogColumns.BH_MASS: galaxy_M,
                    InternalCatalogColumns.BH_MASS_ERROR: galaxy_M * 0.01,
                }
            )
    return rows


def build_galaxy_catalog_for_n_detections(n: int) -> GalaxyCatalogueHandler:
    """Build a GalaxyCatalogueHandler with synthetic galaxies for *n* detections.

    Creates 5*n galaxies positioned near the first *n* detected events.
    """
    handler = object.__new__(GalaxyCatalogueHandler)
    handler.M_min = 10**4.5
    handler.M_max = 10**6
    handler.z_max = 1.5

    rows = _build_galaxy_rows(n_detections=n)
    handler.reduced_galaxy_catalog = pd.DataFrame(rows)

    handler.set_max_relative_errors()
    handler.setup_galaxy_catalog_balltree()
    handler.setup_4d_galaxy_catalog_balltree()

    return handler


@pytest.fixture()
def fake_galaxy_catalog() -> GalaxyCatalogueHandler:
    """Build a GalaxyCatalogueHandler with synthetic galaxies, bypassing __init__.

    The handler has a real BallTree built from 25 synthetic galaxies
    positioned near the 5 detected events in the fixture CSVs.
    """
    return build_galaxy_catalog_for_n_detections(5)


@pytest.fixture(scope="session")
def cosmological_model() -> "Model1CrossCheck":
    """Session-scoped Model1CrossCheck (slow ~3s MCMC burn-in, never mutated)."""
    from master_thesis_code.cosmological_model import Model1CrossCheck

    return Model1CrossCheck(rng=np.random.default_rng(42))
