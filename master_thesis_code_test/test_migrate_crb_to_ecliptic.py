"""Tests for scripts/migrate_crb_to_ecliptic.py.

Round-trip correctness: synthetic equatorial CRB rows → migrate → compare
against the reference transform in galaxy_catalogue/handler.py.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord

from scripts.migrate_crb_to_ecliptic import migrate_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _reference_transform(
    phi_eq: npt.NDArray[np.float64],
    q_eq: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reproduce the handler.py:574-648 rotation for comparison."""
    ra_deg = np.degrees(phi_eq)
    dec_deg = np.degrees(np.pi / 2.0 - q_eq)
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ecl = coord.transform_to(BarycentricTrueEcliptic(equinox="J2000"))
    phi_ecl = np.radians(ecl.lon.to(u.deg).value) % (2.0 * np.pi)
    q_ecl = np.pi / 2.0 - np.radians(ecl.lat.to(u.deg).value)
    return phi_ecl, q_ecl


def _make_synthetic_csv(tmp_path: Path, n: int = 5) -> tuple[Path, pd.DataFrame]:
    """Write a minimal CRB CSV with random equatorial sky angles."""
    phi_eq = _RNG.uniform(0, 2 * np.pi, n)
    q_eq = np.arccos(_RNG.uniform(-1, 1, n))  # uniform on sphere
    df = pd.DataFrame(
        {
            "M": _RNG.uniform(1e5, 1e6, n),
            "mu": np.full(n, 10.0),
            "phiS": phi_eq,
            "qS": q_eq,
            "phiK": _RNG.uniform(0, 2 * np.pi, n),
            "qK": np.arccos(_RNG.uniform(-1, 1, n)),
            "luminosity_distance": _RNG.uniform(0.5, 5.0, n),
            "SNR": _RNG.uniform(20, 100, n),
            "host_galaxy_index": _RNG.integers(0, 1000, n),
            "generation_time": np.ones(n),
        }
    )
    path = tmp_path / "cramer_rao_bounds.csv"
    df.to_csv(path, index=False)
    return path, df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_angles_match_reference_transform(tmp_path: Path) -> None:
    path, original = _make_synthetic_csv(tmp_path)
    migrate_csv(path)

    migrated = pd.read_csv(path)
    phi_ref, q_ref = _reference_transform(
        original["phiS"].values.astype(np.float64),
        original["qS"].values.astype(np.float64),
    )

    np.testing.assert_allclose(migrated["phiS"].values, phi_ref, atol=1e-12)
    np.testing.assert_allclose(migrated["qS"].values, q_ref, atol=1e-12)


def test_coord_frame_column_added(tmp_path: Path) -> None:
    path, _ = _make_synthetic_csv(tmp_path)
    migrate_csv(path)

    df = pd.read_csv(path)
    assert "_coord_frame" in df.columns
    assert (df["_coord_frame"] == "ecliptic_BarycentricTrue_J2000").all()


def test_other_columns_unchanged(tmp_path: Path) -> None:
    path, original = _make_synthetic_csv(tmp_path)
    migrate_csv(path)

    migrated = pd.read_csv(path)
    unchanged = [c for c in original.columns if c not in ("qS", "phiS")]
    pd.testing.assert_frame_equal(
        migrated[unchanged].reset_index(drop=True),
        original[unchanged].reset_index(drop=True),
        check_exact=False,
        atol=1e-12,
    )


def test_backup_written(tmp_path: Path) -> None:
    path, original = _make_synthetic_csv(tmp_path)
    migrate_csv(path)

    backup = path.with_suffix(path.suffix + ".bak_equatorial")
    assert backup.exists(), "*.bak_equatorial backup should be created"

    backup_df = pd.read_csv(backup)
    pd.testing.assert_frame_equal(
        backup_df.reset_index(drop=True),
        original.reset_index(drop=True),
        check_exact=False,
        atol=1e-12,
    )


def test_idempotency_guard_raises(tmp_path: Path) -> None:
    path, _ = _make_synthetic_csv(tmp_path)
    migrate_csv(path)

    # Second call must be skipped (returns skipped, does NOT raise)
    result = migrate_csv(path)
    assert result["status"] == "skipped"


def test_dry_run_does_not_modify(tmp_path: Path) -> None:
    path, original = _make_synthetic_csv(tmp_path)
    original_bytes = path.read_bytes()

    migrate_csv(path, dry_run=True)

    assert path.read_bytes() == original_bytes, "dry-run must not modify the file"
    assert not path.with_suffix(path.suffix + ".bak_equatorial").exists()
