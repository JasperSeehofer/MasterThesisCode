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

from scripts.migrate_crb_to_ecliptic import (
    _PARAM_SYMBOLS,
    _compute_ecliptic_jacobian,
    migrate_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)

# Parameter ordering (must match ParameterSpace._parameters_to_dict)
_N_PARAMS = len(_PARAM_SYMBOLS)


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
    """Write a minimal CRB CSV with random equatorial sky angles and Fisher columns."""
    phi_eq = _RNG.uniform(0, 2 * np.pi, n)
    q_eq = np.arccos(_RNG.uniform(-1, 1, n))  # uniform on sphere
    rows: dict[str, object] = {
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
    # Add all lower-triangle Fisher covariance columns (positive-definite diagonal, small off-diag)
    sigma_val = 1e-4  # representative CRB magnitude (rad^2)
    for r in range(_N_PARAMS):
        for c in range(r + 1):
            key = f"delta_{_PARAM_SYMBOLS[r]}_delta_{_PARAM_SYMBOLS[c]}"
            if r == c:
                rows[key] = np.full(n, sigma_val)
            else:
                rows[key] = np.full(n, sigma_val * 0.1)

    df = pd.DataFrame(rows)
    path = tmp_path / "cramer_rao_bounds.csv"
    df.to_csv(path, index=False)
    return path, df


# ---------------------------------------------------------------------------
# Angle rotation tests
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


def test_cov_frame_column_added(tmp_path: Path) -> None:
    path, _ = _make_synthetic_csv(tmp_path)
    migrate_csv(path)

    df = pd.read_csv(path)
    assert "_cov_frame" in df.columns
    assert (df["_cov_frame"] == "ecliptic_BarycentricTrue_J2000").all()


def test_other_columns_unchanged(tmp_path: Path) -> None:
    path, original = _make_synthetic_csv(tmp_path)
    migrate_csv(path)

    migrated = pd.read_csv(path)
    # Exclude sky angles (rotated), Fisher columns (rotated), and frame tags (added)
    fisher_cols = {
        f"delta_{_PARAM_SYMBOLS[r]}_delta_{_PARAM_SYMBOLS[c]}"
        for r in range(_N_PARAMS)
        for c in range(r + 1)
    }
    unchanged = [c for c in original.columns if c not in ("qS", "phiS") and c not in fisher_cols]
    pd.testing.assert_frame_equal(
        migrated[unchanged].reset_index(drop=True),
        original[unchanged].reset_index(drop=True),
        check_exact=False,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Covariance rotation tests
# ---------------------------------------------------------------------------


def test_covariance_block_rotated(tmp_path: Path) -> None:
    """Sky-sky covariance block must satisfy Σ_ecl = J·Σ_eq·Jᵀ."""
    path, original = _make_synthetic_csv(tmp_path)

    q_eq = original["qS"].values.astype(np.float64)
    phi_eq = original["phiS"].values.astype(np.float64)
    J_all = _compute_ecliptic_jacobian(q_eq, phi_eq)

    migrate_csv(path)
    migrated = pd.read_csv(path)

    qs_i = _PARAM_SYMBOLS.index("qS")
    phis_i = _PARAM_SYMBOLS.index("phiS")

    for i in range(len(original)):
        J = J_all[i]  # (2, 2)

        # Original 2x2 sky-sky block
        sig_qq = float(original["delta_qS_delta_qS"].iloc[i])
        sig_pp = float(original["delta_phiS_delta_phiS"].iloc[i])
        sig_pq = float(original["delta_phiS_delta_qS"].iloc[i])
        sigma_eq = np.array([[sig_qq, sig_pq], [sig_pq, sig_pp]])

        sigma_ecl_expected = J @ sigma_eq @ J.T

        qq_ecl = float(migrated["delta_qS_delta_qS"].iloc[i])
        pp_ecl = float(migrated["delta_phiS_delta_phiS"].iloc[i])
        pq_ecl = float(migrated["delta_phiS_delta_qS"].iloc[i])
        sigma_ecl_actual = np.array([[qq_ecl, pq_ecl], [pq_ecl, pp_ecl]])

        # rtol=1e-6: central finite differences with δ=1e-7 give O(δ²/ε)≈1e-8 precision
        np.testing.assert_allclose(
            sigma_ecl_actual,
            sigma_ecl_expected,
            rtol=1e-6,
            err_msg=f"Sky-sky covariance mismatch at row {i}",
        )


def test_covariance_non_sky_diagonal_unchanged(tmp_path: Path) -> None:
    """Non-sky diagonal CRB entries must not change under sky-angle rotation."""
    path, original = _make_synthetic_csv(tmp_path)
    migrate_csv(path)
    migrated = pd.read_csv(path)

    non_sky = [p for p in _PARAM_SYMBOLS if p not in ("qS", "phiS")]
    for param in non_sky:
        key = f"delta_{param}_delta_{param}"
        if key in original.columns:
            np.testing.assert_allclose(
                migrated[key].values,
                original[key].values,
                atol=1e-14,
                err_msg=f"Non-sky diagonal entry {key} changed unexpectedly",
            )


def test_partial_migration_cov_only(tmp_path: Path) -> None:
    """Pos-only migrated CSV (has _coord_frame, no _cov_frame) gets covariance rotated."""
    path, original = _make_synthetic_csv(tmp_path)

    # Simulate state 2: migrate positions but NOT covariance (old script behaviour)
    backup = path.with_suffix(path.suffix + ".bak_equatorial")
    import shutil

    shutil.copy2(path, backup)  # backup holds original equatorial positions

    df = pd.read_csv(path)
    phi_ref, q_ref = _reference_transform(
        df["phiS"].values.astype(np.float64),
        df["qS"].values.astype(np.float64),
    )
    df["phiS"] = phi_ref
    df["qS"] = q_ref
    df["_coord_frame"] = "ecliptic_BarycentricTrue_J2000"
    df.to_csv(path, index=False)

    result = migrate_csv(path)

    assert result["status"] == "cov_rotated"
    migrated = pd.read_csv(path)
    assert "_cov_frame" in migrated.columns
    assert (migrated["_cov_frame"] == "ecliptic_BarycentricTrue_J2000").all()
    # Sky positions should be unchanged (already ecliptic)
    np.testing.assert_allclose(migrated["phiS"].values, phi_ref, atol=1e-12)
    np.testing.assert_allclose(migrated["qS"].values, q_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# Idempotency / backup tests
# ---------------------------------------------------------------------------


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


def test_idempotency_guard_skips_fully_migrated(tmp_path: Path) -> None:
    path, _ = _make_synthetic_csv(tmp_path)
    migrate_csv(path)

    result = migrate_csv(path)
    assert result["status"] == "skipped"


def test_dry_run_does_not_modify(tmp_path: Path) -> None:
    path, original = _make_synthetic_csv(tmp_path)
    original_bytes = path.read_bytes()

    migrate_csv(path, dry_run=True)

    assert path.read_bytes() == original_bytes, "dry-run must not modify the file"
    assert not path.with_suffix(path.suffix + ".bak_equatorial").exists()
