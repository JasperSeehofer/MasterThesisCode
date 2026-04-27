"""Migrate CRB CSVs from equatorial ICRS to BarycentricTrueEcliptic J2000.

The EMRI simulation stored sky angles qS (polar) and phiS (azimuth) in
equatorial ICRS.  Since Phase 36, galaxy_catalogue/handler.py builds its
BallTree in BarycentricTrueEcliptic J2000 — so evaluation requires ecliptic
angles.  This script applies the identical rotation to stored CRB CSVs.

Idempotency guard: aborts immediately if `_coord_frame` column is already
present (prevents double-rotation).  Always writes a `.bak_equatorial` backup
before modifying the file.

Usage:
    uv run python scripts/migrate_crb_to_ecliptic.py <csv-or-directory> [--dry-run]

    Dry-run prints what would change without writing anything.
    When given a directory, recurses and transforms every
    *cramer_rao_bounds*.csv that has not yet been migrated.
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
import pandas as pd
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord

_COORD_FRAME = "ecliptic_BarycentricTrue_J2000"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def migrate_csv(path: Path, *, dry_run: bool = False) -> dict[str, object]:
    """Transform qS/phiS in *path* from equatorial ICRS to ecliptic J2000.

    Returns a dict with keys: status ("transformed" | "skipped"), rows, path.
    Raises AssertionError if the file is already migrated.
    """
    df = pd.read_csv(path)

    if "_coord_frame" in df.columns:
        print(f"SKIP  {path}  (already has _coord_frame={df['_coord_frame'].iloc[0]!r})")
        return {"status": "skipped", "rows": len(df), "path": str(path)}

    if "qS" not in df.columns or "phiS" not in df.columns:
        raise ValueError(f"{path}: missing qS or phiS column — not a CRB CSV?")

    # qS is equatorial polar angle (colatitude): 0=north, pi=south
    # phiS is equatorial azimuth (RA equivalent) in radians
    ra_rad = df["phiS"].values
    dec_rad = np.pi / 2.0 - df["qS"].values  # colatitude → latitude

    # Rotate ICRS → BarycentricTrueEcliptic J2000 (matches handler.py:574-648)
    coord = SkyCoord(
        ra=np.degrees(ra_rad) * u.deg,
        dec=np.degrees(dec_rad) * u.deg,
        frame="icrs",
    )
    ecl = coord.transform_to(BarycentricTrueEcliptic(equinox="J2000"))

    phi_ecl = np.radians(ecl.lon.to(u.deg).value) % (2.0 * np.pi)
    q_ecl = np.pi / 2.0 - np.radians(ecl.lat.to(u.deg).value)

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
    backup = path.with_suffix(path.suffix + ".bak_equatorial")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"BACKUP {backup}")

    original_sha = _sha256(path)
    df["phiS"] = phi_ecl
    df["qS"] = q_ecl
    df["_coord_frame"] = _COORD_FRAME
    df.to_csv(path, index=False)

    receipt = {
        "migrated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_sha256": original_sha,
        "output_rows": len(df),
        "coord_frame": _COORD_FRAME,
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
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"\nDone: {transformed} transformed, {skipped} skipped (already ecliptic)")


if __name__ == "__main__":
    main()
