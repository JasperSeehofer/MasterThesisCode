"""Consistency tests for physical constants in constants.py."""

import numpy as np

from master_thesis_code.constants import (
    GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT,
    GPC_TO_MPC,
    HOST_DRAW_Z_MAX,
    KM_TO_M,
    OMEGA_DE,
    OMEGA_M,
    RADIAN_TO_DEGREE,
    C,
)


def test_flat_universe() -> None:
    """OMEGA_M + OMEGA_DE ≈ 1 (flat ΛCDM)."""
    assert abs(OMEGA_M + OMEGA_DE - 1.0) < 1e-10


def test_speed_of_light_value() -> None:
    """C must be close to 299 792 458 m/s (exact SI value)."""
    assert abs(C - 299_792_458.0) < 1.0


def test_gpc_to_mpc() -> None:
    """1 Gpc = 1000 Mpc."""
    assert GPC_TO_MPC == 1e3


def test_km_to_m() -> None:
    """1 km = 1000 m."""
    assert KM_TO_M == 1e3


def test_radian_to_degree() -> None:
    """360 degrees = 2π radians."""
    assert abs(RADIAN_TO_DEGREE * 2 * np.pi - 360.0) < 1e-10


def test_host_draw_depth_pin() -> None:
    """Pin the campaign population depth — any change is a /physics-change.

    Pre-#20 value: 0.5 (pre-dt² horizon justification). The Phase-2 campaign
    decision (issue #20, 2026-07-03) deliberately flips this to 1.5 in a
    [PHYSICS] commit that updates this pin in the same diff.
    """
    assert HOST_DRAW_Z_MAX == 0.5


def test_galaxy_catalog_depth_pin() -> None:
    """Pin the documented catalogue depth bound (currently unwired in code)."""
    assert GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT == 0.55


def test_host_draw_within_population_model() -> None:
    """Ordering constraint: the host draw must not exceed the population model
    depth (Model1CrossCheck.max_redshift = 1.5, cosmological_model.py) — the
    d_L pre-screen derivation relies on this ordering."""
    assert HOST_DRAW_Z_MAX <= 1.5
