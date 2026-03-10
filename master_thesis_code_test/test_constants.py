"""Consistency tests for physical constants in constants.py."""

import numpy as np

from master_thesis_code.constants import (
    GPC_TO_MPC,
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
