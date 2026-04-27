"""Migrate CRB CSVs from equatorial ICRS to BarycentricTrueEcliptic J2000.

The EMRI simulation stored sky angles qS (polar) and phiS (azimuth) in
equatorial ICRS.  Since Phase 36, galaxy_catalogue/handler.py builds its
BallTree in BarycentricTrueEcliptic J2000 — so evaluation requires ecliptic
angles.  This script applies the identical rotation to stored CRB CSVs.

Idempotency guards:
  - Already fully migrated (_coord_frame AND _cov_frame both present): skip.
  - Pos-only migrated (_coord_frame present, _cov_frame absent): rotate the
    Fisher covariance block only.  Equatorial positions are read from the
    .bak_equatorial backup (always present for pos-only files).
  - Unmigrated: full migration (rotate positions + covariance).

Always writes a `.bak_equatorial` backup before modifying the file.

Usage:
    uv run python scripts/migrate_crb_to_ecliptic.py <csv-or-directory> [--dry-run]

    Dry-run prints what would change without writing anything.
    When given a directory, recurses and transforms every
    *cramer_rao_bounds*.csv that has not yet been fully migrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord

_COORD_FRAME = "ecliptic_BarycentricTrue_J2000"
_COV_FRAME = "ecliptic_BarycentricTrue_J2000"

# Parameter ordering matches ParameterSpace._parameters_to_dict().
# The Fisher covariance is stored as delta_{params[row]}_delta_{params[col]}
# for the lower triangle (row >= col).
_PARAM_SYMBOLS: list[str] = [
    "M", "mu", "a", "p0", "e0", "x0", "luminosity_distance",
    "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0",
]
_QS_IDX = _PARAM_SYMBOLS.index("qS")    # 7
_PHIS_IDX = _PARAM_SYMBOLS.index("phiS")  # 8
_N_PARAMS = len(_PARAM_SYMBOLS)  # 14

# CRB column names (lower triangle of the 14x14 covariance)
_CRB_COLS: list[str] = [
    f"delta_{_PARAM_SYMBOLS[row]}_delta_{_PARAM_SYMBOLS[col]}"
    for row in range(_N_PARAMS)
    for col in range(row + 1)
]

_JACOBIAN_DELTA = 1e-7  # central-difference step size (radians)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _icrs_to_ecliptic(
    q_eq: npt.NDArray[np.float64],
    phi_eq: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """ICRS (q_eq colatitude, phi_eq azimuth) → BarycentricTrueEcliptic J2000."""
    coord = SkyCoord(
        ra=np.degrees(phi_eq) * u.deg,
        dec=np.degrees(np.pi / 2.0 - q_eq) * u.deg,
        frame="icrs",
    )
    ecl = coord.transform_to(BarycentricTrueEcliptic(equinox="J2000"))
    phi_ecl = np.radians(ecl.lon.to(u.deg).value) % (2.0 * np.pi)
    q_ecl = np.pi / 2.0 - np.radians(ecl.lat.to(u.deg).value)
    return q_ecl, phi_ecl


def _compute_ecliptic_jacobian(
    q_eq: npt.NDArray[np.float64],
    phi_eq: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Jacobian ∂(q_ecl, phi_ecl)/∂(q_eq, phi_eq) for N sources via central differences.

    Returns shape (N, 2, 2):
        J[i] = [[∂q_ecl/∂q_eq,   ∂q_ecl/∂phi_eq],
                [∂phi_ecl/∂q_eq, ∂phi_ecl/∂phi_eq]]

    Taylor An Introduction to Error Analysis §3.11 — the Jacobian matrix needed
    for error propagation under a nonlinear coordinate transform.

    Delta wraps are handled by remapping finite differences to (−π, π] before
    dividing, which is safe for delta=1e-7 rad (changes are of order 1e-7 << π).
    """
    d = _JACOBIAN_DELTA

    # ∂/∂q_eq
    q_p, phi_p = _icrs_to_ecliptic(q_eq + d, phi_eq)
    q_m, phi_m = _icrs_to_ecliptic(q_eq - d, phi_eq)
    dq_dqeq = (q_p - q_m) / (2.0 * d)
    raw = phi_p - phi_m
    dphi_dqeq = ((raw + np.pi) % (2.0 * np.pi) - np.pi) / (2.0 * d)

    # ∂/∂phi_eq
    q_p2, phi_p2 = _icrs_to_ecliptic(q_eq, phi_eq + d)
    q_m2, phi_m2 = _icrs_to_ecliptic(q_eq, phi_eq - d)
    dq_dphieq = (q_p2 - q_m2) / (2.0 * d)
    raw2 = phi_p2 - phi_m2
    dphi_dphieq = ((raw2 + np.pi) % (2.0 * np.pi) - np.pi) / (2.0 * d)

    n = len(q_eq)
    J = np.zeros((n, 2, 2), dtype=np.float64)
    J[:, 0, 0] = dq_dqeq
    J[:, 0, 1] = dq_dphieq
    J[:, 1, 0] = dphi_dqeq
    J[:, 1, 1] = dphi_dphieq
    return J


def _rotate_covariance_block(
    df: pd.DataFrame,
    J_all: npt.NDArray[np.float64],
) -> pd.DataFrame:
    """Apply Σ_ecl = R·Σ_eq·Rᵀ in-place on the lower-triangle CRB columns.

    R is the 14×14 embedding of J into the identity matrix at rows/cols
    [_QS_IDX, _PHIS_IDX].  Only CRB columns that exist in *df* are updated;
    rows missing any CRB column are left unchanged.

    # Eq. (3.47) in Taylor, An Introduction to Error Analysis (2nd ed., 1997)
    # Sigma_ecl = J_rot * Sigma_eq * J_rot^T, where J_rot is the Jacobian
    # of the coordinate transformation at each source position.
    """
    crb_cols_present = [c for c in _CRB_COLS if c in df.columns]
    if not crb_cols_present:
        return df

    df = df.copy()
    idx = df.index  # RangeIndex for CSV-read DataFrames

    for i in range(len(df)):
        J = J_all[i]  # shape (2, 2)

        # Build 14×14 rotation matrix R = I with J embedded
        R = np.eye(_N_PARAMS, dtype=np.float64)
        R[_QS_IDX : _PHIS_IDX + 1, _QS_IDX : _PHIS_IDX + 1] = J

        # Reconstruct symmetric 14×14 Σ from lower-triangle entries
        sigma = np.zeros((_N_PARAMS, _N_PARAMS), dtype=np.float64)
        for r in range(_N_PARAMS):
            for c in range(r + 1):
                key = f"delta_{_PARAM_SYMBOLS[r]}_delta_{_PARAM_SYMBOLS[c]}"
                if key in df.columns:
                    val = float(df.at[idx[i], key])
                    sigma[r, c] = val
                    sigma[c, r] = val

        sigma_ecl = R @ sigma @ R.T

        # Write back lower triangle
        for r in range(_N_PARAMS):
            for c in range(r + 1):
                key = f"delta_{_PARAM_SYMBOLS[r]}_delta_{_PARAM_SYMBOLS[c]}"
                if key in df.columns:
                    df.at[idx[i], key] = sigma_ecl[r, c]

    return df


def migrate_csv(path: Path, *, dry_run: bool = False) -> dict[str, object]:
    """Migrate *path* to ecliptic coordinates and rotate the Fisher covariance.

    Three states handled:
      1. Fully migrated (_coord_frame AND _cov_frame present)  → skip.
      2. Pos-only migrated (_coord_frame present, _cov_frame absent) → covariance
         rotation only; equatorial positions from .bak_equatorial backup.
      3. Unmigrated → full migration (position rotation + covariance rotation).

    Returns dict with keys: status, rows, path.
    """
    df = pd.read_csv(path)

    has_coord = "_coord_frame" in df.columns
    has_cov = "_cov_frame" in df.columns

    # State 1: fully migrated
    if has_coord and has_cov:
        print(f"SKIP  {path}  (fully migrated: _coord_frame + _cov_frame present)")
        return {"status": "skipped", "rows": len(df), "path": str(path)}

    if "qS" not in df.columns or "phiS" not in df.columns:
        raise ValueError(f"{path}: missing qS or phiS column — not a CRB CSV?")

    backup = path.with_suffix(path.suffix + ".bak_equatorial")

    # State 2: pos-only migrated — rotate covariance using backup equatorial positions
    if has_coord and not has_cov:
        if not backup.exists():
            raise FileNotFoundError(
                f"{path}: pos-only migrated but .bak_equatorial backup not found — "
                "cannot recover equatorial positions for Jacobian computation"
            )
        bak_df = pd.read_csv(backup)
        q_eq = bak_df["qS"].values.astype(np.float64)
        phi_eq = bak_df["phiS"].values.astype(np.float64)

        if dry_run:
            print(f"DRY   {path}  ({len(df)} rows)  cov-rotation-only (pos-only state)")
            return {"status": "would_cov_rotate", "rows": len(df), "path": str(path)}

        J_all = _compute_ecliptic_jacobian(q_eq, phi_eq)
        df = _rotate_covariance_block(df, J_all)
        df = df.copy()
        df["_cov_frame"] = _COV_FRAME
        df.to_csv(path, index=False)
        print(f"COV   {path}  ({len(df)} rows → covariance rotated to {_COV_FRAME})")
        return {"status": "cov_rotated", "rows": len(df), "path": str(path)}

    # State 3: unmigrated — full migration
    q_eq = df["qS"].values.astype(np.float64)
    phi_eq = df["phiS"].values.astype(np.float64)

    q_ecl, phi_ecl = _icrs_to_ecliptic(q_eq, phi_eq)

    # Validate ranges (hard fail — matches handler.py D-15 guard)
    assert np.all((phi_ecl >= 0) & (phi_ecl <= 2 * np.pi)), (
        f"phiS out of [0, 2π]: min={phi_ecl.min():.4f}, max={phi_ecl.max():.4f}"
    )
    assert np.all((q_ecl >= 0) & (q_ecl <= np.pi)), (
        f"qS out of [0, π]: min={q_ecl.min():.4f}, max={q_ecl.max():.4f}"
    )

    if dry_run:
        print(
            f"DRY   {path}  ({len(df)} rows)  "
            f"qS [{q_ecl.min():.4f}, {q_ecl.max():.4f}]  "
            f"phiS [{phi_ecl.min():.4f}, {phi_ecl.max():.4f}]"
        )
        return {"status": "would_transform", "rows": len(df), "path": str(path)}

    # Backup before overwriting
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"BACKUP {backup}")

    original_sha = _sha256(path)

    J_all = _compute_ecliptic_jacobian(q_eq, phi_eq)
    df = _rotate_covariance_block(df, J_all)
    df["phiS"] = phi_ecl
    df["qS"] = q_ecl
    df["_coord_frame"] = _COORD_FRAME
    df["_cov_frame"] = _COV_FRAME
    df.to_csv(path, index=False)

    receipt = {
        "migrated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_sha256": original_sha,
        "output_rows": len(df),
        "coord_frame": _COORD_FRAME,
        "cov_frame": _COV_FRAME,
    }
    receipt_path = path.with_suffix(path.suffix + ".migration.json")
    receipt_path.write_text(json.dumps(receipt, indent=2))

    print(f"OK    {path}  ({len(df)} rows → {_COORD_FRAME})")
    return {"status": "transformed", "rows": len(df), "path": str(path)}


def _find_csvs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*cramer_rao_bounds*.csv") if not p.name.endswith(".bak_equatorial")
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", help="CSV file or directory to migrate")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would change without writing"
    )
    args = parser.parse_args(argv)

    target = Path(args.target)
    csvs = _find_csvs(target) if target.is_dir() else [target]

    if not csvs:
        print(f"No *cramer_rao_bounds*.csv files found under {target}")
        sys.exit(0)

    results = [migrate_csv(p, dry_run=args.dry_run) for p in csvs]
    transformed = sum(1 for r in results if r["status"] == "transformed")
    cov_rotated = sum(1 for r in results if r["status"] == "cov_rotated")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(
        f"\nDone: {transformed} fully transformed, "
        f"{cov_rotated} covariance-rotated, "
        f"{skipped} skipped (already ecliptic)"
    )


if __name__ == "__main__":
    main()
