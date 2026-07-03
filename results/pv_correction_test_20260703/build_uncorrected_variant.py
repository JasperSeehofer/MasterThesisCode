"""Build the PV-UNCORRECTED GLADE+ reduced-catalogue variant (issue #16, handoff §7b).

Purpose
-------
The live reduced catalogue (master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv)
uses GLADE+ raw column 28 (0-based) = z_cmb, which for rows with flag2 (raw col 29) == 1 is
ADDITIONALLY corrected for peculiar velocity (Dálya et al. 2022, arXiv:2110.06184). To isolate
the impact of that PV *value* correction on the H0 posterior, this script builds a variant
reduced catalogue that is byte-for-byte identical to the production parse
(GalaxyCatalogueHandler.parse_to_reduced_catalog, handler.py:311-369) EXCEPT that for
flag2==1 rows the redshift column is replaced by the pure heliocentric->CMB *frame*
transform of z_helio (raw col 27), i.e. with the PV value-correction REMOVED. flag2==0
rows keep col 28 unchanged (for those rows col 28 already IS the pure frame transform).

Method
------
Two streaming passes over GLADE+.txt (space-separated, headerless, 'null' == NaN which
pandas' default na_values already handles):

  PASS 1 (validation): on flag2==0 rows (where z_cmb must equal the pure frame transform
  of z_helio up to GLADE+'s storage rounding), reconstruct z_cmb from z_helio + the Planck
  solar dipole (v_sun = 369.82 km/s toward galactic l = 264.021 deg, b = 48.253 deg,
  Planck 2018 I. arXiv:1807.06205; apex converted to ICRS via astropy) under five candidate
  conventions and measure |z_cmb_reconstructed - z_cmb_glade|:

      mult_plus  : (1+z_cmb) = (1+z_helio) * (1 + (v/c) cos(theta))
      mult_minus : (1+z_cmb) = (1+z_helio) * (1 - (v/c) cos(theta))
      add_plus   :    z_cmb  =    z_helio  +      (v/c) cos(theta)
      add_minus  :    z_cmb  =    z_helio  -      (v/c) cos(theta)
      mult_sr_plus: (1+z_cmb) = (1+z_helio) * gamma * (1 + (v/c) cos(theta))   [SR-exact]

  where theta is the angle between the galaxy direction and the solar apex (direction of
  solar motion in the CMB frame). The convention with the smallest median residual is
  selected; the script ABORTS (writes stats.json with status=FAILED, no variant CSV) if
  no convention reaches median < 1e-4.

  PASS 2 (build): replicate the production parse exactly — read the CatalogueColumns set,
  filter z_flag (raw col 34) in {1, 3}, fill NaN PV-error (raw col 30) with 0.0015 on ALL
  rows, fold it in quadrature into the measurement error (raw col 31), drop the PV-error
  column, cast the flag to int, reorder to
      [RA, Dec, B_mag, REDSHIFT, REDSHIFT_MEASUREMENT_ERROR,
       STELLAR_MASS, STELLAR_MASS_ABSOULTE_ERROR, REDSHIFT_FLAG]
  and append headerless — EXCEPT the REDSHIFT column, which for flag2==1 rows with finite
  z_helio is the validated frame-only transform of z_helio. flag2==1 rows with missing
  z_helio (counted in stats.json) keep col 28 as a fallback.

The output CSV is written to reduced_galaxy_catalogue_noPVcorr.csv in this directory.
The writer APPENDS per chunk (like the production writer), so the script refuses to run
if the output file already exists. Fully deterministic: no randomness anywhere.

Usage
-----
    uv run python results/pv_correction_test_20260703/build_uncorrected_variant.py

Runtime is dominated by two full scans of the 6.4 GB raw catalogue (a few minutes).
"""

import json
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.coordinates import SkyCoord

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RAW_CATALOGUE = REPO / "master_thesis_code" / "galaxy_catalogue" / "GLADE+.txt"
LIVE_REDUCED = REPO / "master_thesis_code" / "galaxy_catalogue" / "reduced_galaxy_catalogue.csv"
OUT_CSV = HERE / "reduced_galaxy_catalogue_noPVcorr.csv"
OUT_STATS = HERE / "stats.json"

# --------------------------------------------------------------------------------------
# Raw GLADE+ 0-based columns (verified against handler.py CatalogueColumns and the
# Dálya et al. 2022, arXiv:2110.06184 schema). Must be listed in ASCENDING order:
# pandas applies `names` to `usecols` in ascending file order.
# --------------------------------------------------------------------------------------
USECOLS = [8, 9, 10, 27, 28, 29, 30, 31, 34, 35, 36]
NAMES = [
    "RIGHT_ASCENSION",  # 8  RA (deg, ICRS)
    "DECLINATION",  # 9  Dec (deg, ICRS)
    "APPARENT_B_MAG",  # 10 apparent B magnitude
    "Z_HELIO",  # 27 heliocentric redshift
    "REDSHIFT",  # 28 z_cmb (PV-corrected where col 29 == 1)
    "PV_FLAG",  # 29 flag2: 1 = z_cmb corrected for peculiar velocity
    "REDSHIFT_PECULIAR_VELOCITY_ERROR",  # 30 PV-correction error (z units)
    "REDSHIFT_MEASUREMENT_ERROR",  # 31 z measurement error
    "REDSHIFT_FLAG",  # 34 z flag: 1 = photo-z, 3 = spec-z (kept)
    "STELLAR_MASS",  # 35 stellar mass (1e10 Msun)
    "STELLAR_MASS_ABSOULTE_ERROR",  # 36 stellar mass error (1e10 Msun)
]

# On-disk reduced-catalogue column order — MUST match handler._reduced_catalog_column_names().
REDUCED_ORDER = [
    "RIGHT_ASCENSION",
    "DECLINATION",
    "APPARENT_B_MAG",
    "REDSHIFT",
    "REDSHIFT_MEASUREMENT_ERROR",
    "STELLAR_MASS",
    "STELLAR_MASS_ABSOULTE_ERROR",
    "REDSHIFT_FLAG",
]

CHUNKSIZE = 1_000_000
PV_ERROR_FILL = 0.0015  # handler.py:348 — NaN PV error -> 0.0015 on ALL kept rows
EXPECTED_ROWS = 22_641_048  # live reduced catalogue row count (parity requirement)

# Planck 2018 solar dipole (arXiv:1807.06205): v_sun toward galactic (l, b).
V_SUN_KM_S = 369.82
APEX_L_DEG = 264.021
APEX_B_DEG = 48.253
SPEED_OF_LIGHT_KM_S = 299_792.458
BETA = V_SUN_KM_S / SPEED_OF_LIGHT_KM_S
GAMMA = 1.0 / np.sqrt(1.0 - BETA**2)

CONVENTIONS = ["mult_plus", "mult_minus", "add_plus", "add_minus", "mult_sr_plus"]
MEDIAN_ACCEPT = 1e-4  # abort threshold on the best convention's median residual


def apex_unit_vector_icrs() -> npt.NDArray[np.float64]:
    """Unit vector of the solar apex (CMB dipole direction) in ICRS cartesian coords."""
    apex = SkyCoord(l=APEX_L_DEG, b=APEX_B_DEG, unit="deg", frame="galactic").icrs
    ra = np.radians(apex.ra.deg)
    dec = np.radians(apex.dec.deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def cos_theta_to_apex(
    ra_deg: npt.NDArray[np.float64],
    dec_deg: npt.NDArray[np.float64],
    apex: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """cos(angle) between galaxy directions (ICRS RA/Dec in deg) and the solar apex."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return (
        np.cos(dec) * np.cos(ra) * apex[0]
        + np.cos(dec) * np.sin(ra) * apex[1]
        + np.sin(dec) * apex[2]
    )


def frame_only_z(
    z_helio: npt.NDArray[np.float64],
    cos_theta: npt.NDArray[np.float64],
    convention: str,
) -> npt.NDArray[np.float64]:
    """Pure heliocentric->CMB frame transform of z_helio under a given convention."""
    z_dip = BETA * cos_theta
    if convention == "mult_plus":
        return (1.0 + z_helio) * (1.0 + z_dip) - 1.0
    if convention == "mult_minus":
        return (1.0 + z_helio) * (1.0 - z_dip) - 1.0
    if convention == "add_plus":
        return z_helio + z_dip
    if convention == "add_minus":
        return z_helio - z_dip
    if convention == "mult_sr_plus":
        return (1.0 + z_helio) * GAMMA * (1.0 + z_dip) - 1.0
    raise ValueError(f"unknown convention: {convention}")


def read_chunks() -> "pd.io.parsers.TextFileReader":
    """Stream the raw catalogue with the same reader settings as the production parse."""
    return pd.read_csv(
        RAW_CATALOGUE,
        sep=" ",
        header=None,
        usecols=USECOLS,
        names=NAMES,
        dtype=np.float64,  # all kept columns are numeric; 'null' -> NaN (default na_values)
        chunksize=CHUNKSIZE,
    )


def count_lines(path: Path) -> int:
    """Fast buffered newline count (row-count parity check against the live CSV)."""
    n = 0
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(1 << 24)
            if not buf:
                return n
            n += buf.count(b"\n")


def pass1_validate() -> dict:
    """PASS 1: validate the frame transform on flag2==0 rows; gather flagged-row stats.

    Returns a dict with per-convention residual stats, the chosen convention, counts,
    the |z_glade_corrected - z_frame_only| distribution over flagged rows (evaluated
    later under the chosen convention), and the flagged-row redshift range.
    """
    apex = apex_unit_vector_icrs()
    resid: dict[str, list[npt.NDArray[np.float32]]] = {c: [] for c in CONVENTIONS}
    dz_flagged: dict[str, list[npt.NDArray[np.float32]]] = {c: [] for c in CONVENTIONS}
    n_total = 0
    n_flagged = 0
    n_flag0 = 0
    n_flag0_validated = 0
    n_flagged_no_zhelio = 0
    z_flagged_min = np.inf
    z_flagged_max = -np.inf

    for chunk in read_chunks():
        chunk = chunk[(chunk["REDSHIFT_FLAG"] == 1) | (chunk["REDSHIFT_FLAG"] == 3)]
        n_total += len(chunk)

        flagged = chunk["PV_FLAG"] == 1
        n_flagged += int(flagged.sum())
        n_flag0 += int((~flagged).sum())

        cos_t = cos_theta_to_apex(
            chunk["RIGHT_ASCENSION"].to_numpy(),
            chunk["DECLINATION"].to_numpy(),
            apex,
        )
        z_helio = chunk["Z_HELIO"].to_numpy()
        z_cmb = chunk["REDSHIFT"].to_numpy()
        fl = flagged.to_numpy()

        # Validation set: flag2==0 rows with both redshifts present.
        val = (~fl) & np.isfinite(z_helio) & np.isfinite(z_cmb)
        n_flag0_validated += int(val.sum())
        # Flagged-row correction magnitude set: flag2==1 with both redshifts present.
        cor = fl & np.isfinite(z_helio) & np.isfinite(z_cmb)
        n_flagged_no_zhelio += int((fl & ~np.isfinite(z_helio)).sum())
        if cor.any():
            z_flagged_min = min(z_flagged_min, float(np.min(z_cmb[cor])))
            z_flagged_max = max(z_flagged_max, float(np.max(z_cmb[cor])))

        for conv in CONVENTIONS:
            z_rec = frame_only_z(z_helio, cos_t, conv)
            resid[conv].append(np.abs(z_rec[val] - z_cmb[val]).astype(np.float32))
            dz_flagged[conv].append(np.abs(z_cmb[cor] - z_rec[cor]).astype(np.float32))

    stats: dict = {
        "n_rows_total": n_total,
        "n_flagged": n_flagged,
        "n_flag0": n_flag0,
        "n_flag0_validated": n_flag0_validated,
        "n_flagged_no_zhelio": n_flagged_no_zhelio,
        "transform_validation_residuals": {},
    }
    medians: dict[str, float] = {}
    for conv in CONVENTIONS:
        r = np.concatenate(resid[conv])
        medians[conv] = float(np.median(r))
        stats["transform_validation_residuals"][conv] = {
            "median": float(np.median(r)),
            "p99": float(np.percentile(r, 99)),
            "max": float(np.max(r)),
        }
    chosen = min(medians, key=lambda c: medians[c])
    stats["chosen_convention"] = chosen
    stats["chosen_median_residual"] = medians[chosen]

    d = np.concatenate(dz_flagged[chosen])
    stats["pv_correction_magnitude_flagged_rows"] = {
        "description": "|z_glade_corrected - z_frame_only(chosen)| over flag2==1 rows",
        "n": int(d.size),
        "median": float(np.median(d)),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "max": float(np.max(d)),
    }
    stats["flagged_rows_redshift_range"] = {
        "z_cmb_min": z_flagged_min,
        "z_cmb_max": z_flagged_max,
    }
    return stats


def pass2_build(convention: str) -> int:
    """PASS 2: write the variant CSV, replicating the production parse exactly except
    that flag2==1 rows get the frame-only transform of z_helio as REDSHIFT.

    Returns the number of rows written.
    """
    if OUT_CSV.exists():
        raise FileExistsError(f"{OUT_CSV} already exists — the writer appends; delete it first.")
    apex = apex_unit_vector_icrs()
    n_written = 0

    for chunk in read_chunks():
        # Same {1, 3} redshift-flag filter as the production parse (handler.py:343-346).
        chunk = chunk[(chunk["REDSHIFT_FLAG"] == 1) | (chunk["REDSHIFT_FLAG"] == 3)]

        # REDSHIFT override: flag2==1 rows with finite z_helio -> frame-only transform.
        replace = (chunk["PV_FLAG"] == 1) & np.isfinite(chunk["Z_HELIO"])
        if replace.any():
            cos_t = cos_theta_to_apex(
                chunk.loc[replace, "RIGHT_ASCENSION"].to_numpy(),
                chunk.loc[replace, "DECLINATION"].to_numpy(),
                apex,
            )
            chunk.loc[replace, "REDSHIFT"] = frame_only_z(
                chunk.loc[replace, "Z_HELIO"].to_numpy(), cos_t, convention
            )

        # Identical error handling to the production parse (handler.py:348-354):
        # fill NaN PV error with 0.0015 on ALL rows, fold in quadrature.
        chunk = chunk.fillna({"REDSHIFT_PECULIAR_VELOCITY_ERROR": PV_ERROR_FILL})
        chunk["REDSHIFT_MEASUREMENT_ERROR"] = np.sqrt(
            chunk["REDSHIFT_MEASUREMENT_ERROR"] ** 2
            + chunk["REDSHIFT_PECULIAR_VELOCITY_ERROR"] ** 2
        )

        # Flag round-trips as int (handler.py:361-363); reorder and append headerless.
        chunk["REDSHIFT_FLAG"] = chunk["REDSHIFT_FLAG"].astype(int)
        chunk = chunk[REDUCED_ORDER]
        chunk.to_csv(OUT_CSV, header=False, mode="a", index=False)
        n_written += len(chunk)

    return n_written


def main() -> None:
    t0 = time.time()
    stats: dict = {
        "raw_catalogue": str(RAW_CATALOGUE),
        "live_reduced_catalogue": str(LIVE_REDUCED),
        "output_variant": str(OUT_CSV),
        "dipole": {
            "v_sun_km_s": V_SUN_KM_S,
            "apex_galactic_l_deg": APEX_L_DEG,
            "apex_galactic_b_deg": APEX_B_DEG,
            "reference": "Planck 2018 I, arXiv:1807.06205",
        },
        "pv_error_fill": PV_ERROR_FILL,
        "expected_rows": EXPECTED_ROWS,
    }

    print("PASS 1: transform validation on flag2==0 rows ...", flush=True)
    t1 = time.time()
    stats.update(pass1_validate())
    print(f"PASS 1 done in {time.time() - t1:.1f}s", flush=True)
    for conv, r in stats["transform_validation_residuals"].items():
        marker = " <-- chosen" if conv == stats["chosen_convention"] else ""
        print(
            f"  {conv:13s} median={r['median']:.3e}  p99={r['p99']:.3e}  "
            f"max={r['max']:.3e}{marker}",
            flush=True,
        )

    # Row-count parity gate: the {1,3}-filtered raw stream MUST match the live CSV.
    live_rows = count_lines(LIVE_REDUCED)
    stats["live_reduced_rows"] = live_rows
    stats["row_parity_ok"] = stats["n_rows_total"] == live_rows == EXPECTED_ROWS
    print(
        f"Row parity: filtered raw = {stats['n_rows_total']}, live reduced = {live_rows}, "
        f"expected = {EXPECTED_ROWS} -> {'OK' if stats['row_parity_ok'] else 'MISMATCH'}",
        flush=True,
    )

    if stats["chosen_median_residual"] >= MEDIAN_ACCEPT:
        stats["status"] = "FAILED_TRANSFORM_VALIDATION"
        OUT_STATS.write_text(json.dumps(stats, indent=2) + "\n")
        raise SystemExit(
            f"ABORT: best convention {stats['chosen_convention']} has median residual "
            f"{stats['chosen_median_residual']:.3e} >= {MEDIAN_ACCEPT:.0e}; "
            "no variant written."
        )
    if not stats["row_parity_ok"]:
        stats["status"] = "FAILED_ROW_PARITY"
        OUT_STATS.write_text(json.dumps(stats, indent=2) + "\n")
        raise SystemExit(
            "ABORT: row-count parity check failed; investigate before building the variant."
        )

    print(
        f"PASS 2: building variant with convention '{stats['chosen_convention']}' ...",
        flush=True,
    )
    t2 = time.time()
    n_written = pass2_build(stats["chosen_convention"])
    print(f"PASS 2 done in {time.time() - t2:.1f}s ({n_written} rows)", flush=True)

    stats["variant_rows_written"] = n_written
    stats["variant_row_parity_ok"] = n_written == EXPECTED_ROWS
    stats["status"] = "OK" if stats["variant_row_parity_ok"] else "FAILED_VARIANT_ROWCOUNT"
    stats["runtime_s"] = round(time.time() - t0, 1)
    OUT_STATS.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"DONE in {stats['runtime_s']}s — stats written to {OUT_STATS}", flush=True)


if __name__ == "__main__":
    main()
