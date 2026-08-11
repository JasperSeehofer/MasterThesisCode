"""Seeded observed-catalogue realization writer (campaign #53).

[PHYSICS] Realistic host-observation model, RATIFIED 2026-07-29
(docs/derivations/realistic_host_observation_model.md, gates
[RATIFY-R1]..[RATIFY-R9]). Convention (A): the reduced catalogue's stored
values are declared TRUE; observations are realized FORWARD::

    z_obs   = z_g + sigma_scale * N(0, z_error_g)          (§1.2 / §2.1)
    ln M_obs = ln M_g + sigma_scale * N(0, M_error_g / M_g)  (§1.3 / §2.3)

Counted-once [RATIFY-R2]: the redshift draw uses exactly the stored
``z_error`` column (the #40b total: measurement (+) BORG-PV (+) per-class PV,
folded once at parse time) — NEVER a re-derived component split (the PV class
is irrecoverable from the reduced CSV, §2.1/F2). The mass draw uses exactly
the lognormal width ``M_error/M`` the with-BH-mass kernel consumes — i.e. the
BH-mass width produced by the SAME load-time Reines & Volonteri mapping
(:func:`~darksiren_emri.galaxy_catalogue.handler._empiric_stellar_mass_to_BH_mass_relation`).
Because the on-disk column is the STELLAR mass and the RV mean relation is
affine in log space (ln M_BH = alpha + beta ln(M*/10)), the BH-mass draw
delta_lnM is written to disk as M*_obs = M* exp(delta_lnM / beta), so the
load-time mapping reproduces ln M_BH_obs = ln M_BH + delta_lnM exactly.

The z >= GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT (1e-5) floor is a CLIP (point
mass at the floor, no redraw) — the author-accepted approximation of §2.4;
the affected fraction is logged in the sidecar.

sigma_scale = 0 is the mandatory regression gate [RATIFY-R6]: the observed
catalogue is a BYTE-IDENTICAL copy of the parent (string copy, never a float
round-trip), verifiable by sha256 equality of parent and child.

Provenance (§6.1): every realization writes a hashed sidecar
``<observed>.meta.json`` recording the realization seed, sigma_scale, parent
and own sha256, git commit, timestamp, row count, floor-clip and mass-window
crossing counts, and per-flag realized-vs-stored width check statistics.
Guard [RATIFY-R9] item 5: an existing observed catalogue is never overwritten
with a different parent hash.
"""

import datetime
import hashlib
import json
import logging
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

from darksiren_emri.constants import (
    GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT,
    M_SOURCE_FRAME_MAX,
    M_SOURCE_FRAME_MIN,
)
from darksiren_emri.galaxy_catalogue.handler import (
    CatalogueColumns,
    _empiric_stellar_mass_to_BH_mass_relation,
    _reduced_catalog_column_names,
    beta,
    d_alpha,
    d_beta,
    sigma_int,
)

_LOGGER = logging.getLogger()

_DERIVATION = "docs/derivations/realistic_host_observation_model.md"

# sha256 streaming chunk size (the production catalogue is ~1 GB).
_HASH_CHUNK_BYTES = 1 << 20


def observed_catalogue_filename(realization_seed: int) -> str:
    """Canonical observed-catalogue file name for a realization seed (§6.1)."""
    return f"observed_catalogue_seed{realization_seed}.csv"


def sidecar_path_for(observed_csv_path: str) -> str:
    """Sidecar metadata path for an observed catalogue CSV.

    ``observed_catalogue_seed{S}.csv`` -> ``observed_catalogue_seed{S}.meta.json``
    (§6.1); a non-``.csv`` path gets ``.meta.json`` appended.
    """
    root, ext = os.path.splitext(observed_csv_path)
    if ext == ".csv":
        return root + ".meta.json"
    return observed_csv_path + ".meta.json"


def _sha256_of_file(path: str) -> str:
    """Streaming sha256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _get_git_commit() -> str:
    """Current git commit hash, or 'unknown' outside a repo (provenance only)."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_realization_sidecar(
    observed_csv_path: str, *, verify_hash: bool = True
) -> dict[str, object] | None:
    """Load and verify the sidecar for an observed catalogue CSV.

    Args:
        observed_csv_path: Path to the (observed) catalogue CSV.
        verify_hash: When True (default), re-hash the CSV and require it to
            match the sidecar's recorded ``observed_csv_sha256`` — one
            realization, all legs, verifiable (§6.1).

    Returns:
        The sidecar dict, or None when no sidecar exists (legacy/unscattered
        baseline catalogue, guard [RATIFY-R9] item 4 — callers must log this
        prominently and treat the catalogue as unscattered).

    Raises:
        ValueError: The sidecar exists but its recorded hash does not match
            the CSV on disk (a realization/catalogue mismatch would silently
            invalidate every evaluation leg).
    """
    sidecar_path = sidecar_path_for(observed_csv_path)
    if not os.path.isfile(sidecar_path):
        return None
    with open(sidecar_path) as handle:
        sidecar: dict[str, object] = json.load(handle)
    if verify_hash:
        actual = _sha256_of_file(observed_csv_path)
        recorded = sidecar.get("observed_csv_sha256")
        if actual != recorded:
            raise ValueError(
                f"observed-catalogue hash mismatch for {observed_csv_path}: sidecar "
                f"records sha256 {recorded!r} but the file on disk hashes to "
                f"{actual!r}. The realization and the catalogue have diverged — "
                f"regenerate the realization ({_DERIVATION} §6.1)."
            )
    return sidecar


def _count_data_rows(path: str) -> int:
    """Count non-empty lines of a headerless CSV without parsing it."""
    n = 0
    with open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def realize_observed_catalogue(
    parent_csv_path: str,
    output_csv_path: str,
    realization_seed: int,
    sigma_scale: float = 1.0,
) -> dict[str, object]:
    """Realize an OBSERVED catalogue from the reduced (TRUE) catalogue (§2, §6.1).

    Draws ONE total Gaussian per row from the stored width columns
    ([RATIFY-R2] counted-once):

    - ``z_obs = z_g + sigma_scale * z_error_g * N(0,1)``, clipped at the
      1e-5 floor (point mass, no redraw — §2.4);
    - ``ln M_BH,obs = ln M_BH,g + sigma_scale * (M_error_g / M_g) * N(0,1)``
      with (M_g, M_error_g) the load-time Reines & Volonteri BH mass/error of
      the stored stellar columns, written to disk through the exact affine
      inverse ``M*_obs = M* * exp(delta_lnM_BH / beta)`` (§1.3).

    ``z_error``, flags, sky positions and B magnitudes are copied as their
    original strings: the z width law is scale-free in z, so the stored column
    IS the width the kernel consumes and ``sigma_kernel == sigma_realized``
    identically. The stellar-ERROR column, by contrast, IS rewritten — the
    BH-mass width law is not scale-free in ``M*``, so its propagated term is
    re-solved per row to keep the width the inference recomputes equal to the
    width the mass scatter was drawn with ([RATIFY-R2]; leaving it untouched
    measured a 7 % pull deficit). With ``sigma_scale = 0`` the output is a
    byte-identical copy of the parent (the [RATIFY-R6] gate).

    Args:
        parent_csv_path: The reduced catalogue CSV (declared TRUTH, §1.2 (A)).
        output_csv_path: Destination observed-catalogue CSV; the sidecar is
            written next to it (:func:`sidecar_path_for`).
        realization_seed: Seed for ``np.random.default_rng`` — the realization
            is a pure function of (parent bytes, seed, sigma_scale).
        sigma_scale: Global width multiplier; 0 disables scatter (byte copy),
            1 is the production realization. Must be >= 0.

    Returns:
        The sidecar dict (also written to disk as JSON).

    Raises:
        ValueError: Negative sigma_scale, output equal to parent, or an
            existing output whose sidecar records a DIFFERENT parent hash
            (guard [RATIFY-R9] item 5) / an existing output without any
            sidecar (unknown provenance).
        FileNotFoundError: Missing parent CSV.
    """
    if sigma_scale < 0:
        raise ValueError(f"sigma_scale must be >= 0, got {sigma_scale}")
    if not os.path.isfile(parent_csv_path):
        raise FileNotFoundError(f"parent catalogue not found: {parent_csv_path}")
    if os.path.abspath(parent_csv_path) == os.path.abspath(output_csv_path):
        raise ValueError(
            "output_csv_path equals parent_csv_path — the realization must never "
            "overwrite the TRUE catalogue (convention (A), §1.2)."
        )

    parent_sha256 = _sha256_of_file(parent_csv_path)

    # Guard [RATIFY-R9] item 5: never overwrite an existing realization that
    # was produced from a DIFFERENT parent (or has unknown provenance).
    if os.path.isfile(output_csv_path):
        existing = load_realization_sidecar(output_csv_path, verify_hash=False)
        if existing is None:
            raise ValueError(
                f"refusing to overwrite {output_csv_path}: it exists without a "
                f"sidecar (unknown provenance). Remove it manually to proceed "
                f"({_DERIVATION} §9 guard 5)."
            )
        if existing.get("parent_csv_sha256") != parent_sha256:
            raise ValueError(
                f"refusing to overwrite {output_csv_path}: its sidecar records "
                f"parent sha256 {existing.get('parent_csv_sha256')!r} but the "
                f"current parent hashes to {parent_sha256!r} "
                f"({_DERIVATION} §9 guard 5)."
            )

    n_z_floor_clipped = 0
    n_mass_window_crossings = 0
    width_check: dict[str, object] | None = None

    if sigma_scale == 0.0:
        # [RATIFY-R6] sigma -> 0 regression gate: BYTE-IDENTICAL copy of the
        # parent — copy the file, never round-trip floats.
        shutil.copyfile(parent_csv_path, output_csv_path)
        n_rows = _count_data_rows(output_csv_path)
    else:
        n_rows, n_z_floor_clipped, n_mass_window_crossings, width_check = _realize_and_write(
            parent_csv_path, output_csv_path, realization_seed, sigma_scale
        )

    observed_sha256 = _sha256_of_file(output_csv_path)
    sidecar: dict[str, object] = {
        "realization_seed": int(realization_seed),
        "sigma_scale": float(sigma_scale),
        "parent_csv": os.path.abspath(parent_csv_path),
        "parent_csv_sha256": parent_sha256,
        "observed_csv_sha256": observed_sha256,
        "git_commit": _get_git_commit(),
        "timestamp": datetime.datetime.now().isoformat(),
        "n_rows": int(n_rows),
        "n_z_floor_clipped": int(n_z_floor_clipped),
        "n_mass_window_crossings": int(n_mass_window_crossings),
        "z_floor": GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT,
        "width_check": width_check,
        "derivation": _DERIVATION,
    }
    with open(sidecar_path_for(output_csv_path), "w") as handle:
        json.dump(sidecar, handle, indent=2)
    _LOGGER.info(
        "Observed catalogue realized: %s (seed=%d, sigma_scale=%g, rows=%d, "
        "z-floor clipped=%d, mass-window crossings=%d)",
        output_csv_path,
        realization_seed,
        sigma_scale,
        n_rows,
        n_z_floor_clipped,
        n_mass_window_crossings,
    )
    return sidecar


def _realize_and_write(
    parent_csv_path: str,
    output_csv_path: str,
    realization_seed: int,
    sigma_scale: float,
) -> tuple[int, int, int, dict[str, object]]:
    """Draw the per-row realization and write the observed CSV (sigma_scale > 0).

    The parent is read with ``dtype=str`` so every column the realization does
    NOT touch (sky, B-mag, z_error, M*_error, flag) round-trips as its
    original string. Only the REDSHIFT and STELLAR_MASS fields are replaced.

    Returns:
        (n_rows, n_z_floor_clipped, n_mass_window_crossings, width_check).
    """
    names = _reduced_catalog_column_names()
    z_col = CatalogueColumns.REDSHIFT.name
    z_err_col = CatalogueColumns.REDSHIFT_MEASUREMENT_ERROR.name
    mstar_col = CatalogueColumns.STELLAR_MASS.name
    mstar_err_col = CatalogueColumns.STELLAR_MASS_ABSOULTE_ERROR.name
    flag_col = CatalogueColumns.REDSHIFT_FLAG.name

    catalog = pd.read_csv(parent_csv_path, names=names, dtype=str)
    n_rows = len(catalog)

    z_true = pd.to_numeric(catalog[z_col], errors="coerce").to_numpy(dtype=np.float64)
    z_error = pd.to_numeric(catalog[z_err_col], errors="coerce").to_numpy(dtype=np.float64)
    mstar = pd.to_numeric(catalog[mstar_col], errors="coerce").to_numpy(dtype=np.float64)
    mstar_error = pd.to_numeric(catalog[mstar_err_col], errors="coerce").to_numpy(dtype=np.float64)
    flags = catalog[flag_col].to_numpy(dtype=object)

    rng = np.random.default_rng(realization_seed)
    # Fixed draw order (z for all rows, then mass for all rows) so the
    # realization is a pure function of (parent, seed, sigma_scale)
    # independent of any validity masks.
    z_std_normal = rng.standard_normal(n_rows)
    mass_std_normal = rng.standard_normal(n_rows)

    # --- redshift: ONE total Gaussian from the stored z_error column
    # ([RATIFY-R2]; z_error is the #40b counted-once total). Clip at the
    # 1e-5 floor — point mass, no redraw (§2.4, author-accepted).
    z_valid = np.isfinite(z_true) & np.isfinite(z_error)
    z_obs_unclipped = z_true + sigma_scale * z_error * z_std_normal
    z_obs = np.maximum(z_obs_unclipped, GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT)
    clipped_mask = z_valid & (z_obs_unclipped < GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT)
    n_z_floor_clipped = int(np.count_nonzero(clipped_mask))

    # --- mass: lognormal in the BH-mass width the kernel consumes (§1.3,
    # §2.3 counted-once-in-M). sigma_lnM = M_error/M from the SAME load-time
    # Reines & Volonteri mapping the handler applies; the affine relation
    # ln M_BH = alpha + beta ln(M*/10) makes the stellar-column update
    # M*_obs = M* exp(delta_lnM_BH / beta) EXACT (delta_lnM* = delta_lnM_BH/beta).
    mass_valid = (
        np.isfinite(mstar) & (mstar > 0.0) & np.isfinite(mstar_error) & (mstar_error >= 0.0)
    )
    mstar_safe = np.where(mass_valid, mstar, 1.0)
    mstar_error_safe = np.where(mass_valid, mstar_error, 0.0)
    # The relation is written scalar-typed but is pure elementwise numpy —
    # the handler itself calls it with whole pandas Series (handler.py:885-891).
    _bh_mass_raw, _bh_mass_error_raw = _empiric_stellar_mass_to_BH_mass_relation(
        mstar_safe,  # type: ignore[arg-type]
        mstar_error_safe,  # type: ignore[arg-type]
    )
    bh_mass = np.asarray(_bh_mass_raw, dtype=np.float64)
    bh_mass_error = np.asarray(_bh_mass_error_raw, dtype=np.float64)
    sigma_ln_bh = bh_mass_error / bh_mass
    delta_ln_bh = sigma_scale * sigma_ln_bh * mass_std_normal
    mstar_obs = mstar_safe * np.exp(delta_ln_bh / beta)
    # [RATIFY-R2 counted-once — exact width preservation]
    # sigma_lnM is NOT scale-free in M*: the load-time law
    #   sigma_lnM(M*, s)^2 = sigma_int^2 + d_alpha^2 + (d_beta ln(M*/10))^2
    #                        + (beta s / M*)^2                (handler.py:1196-1203)
    # moves when M* moves while the stored stellar-error column stays put, so a
    # realization that only rewrites M* leaves the INFERENCE consuming a width
    # different from the one the scatter was drawn with — MEASURED pull vs the
    # recomputed width 0.929, per-row drift up to +-18%. (A fixed-point sweep on
    # the width is NOT a remedy: sigma grows as M* shrinks, so the map is not a
    # contraction and diverges for the negative tail.)
    # Exact one-shot remedy: rewrite the stellar-error column too, solving the
    # propagated term so the loaded width equals the width actually used:
    #   (beta s_obs / M*_obs)^2 = sigma_used^2 - sigma_int^2 - d_alpha^2
    #                             - (d_beta ln(M*_obs/10))^2 .
    # Rows whose mass-independent + log-lever terms already exceed sigma_used
    # (far from the 1e11 pivot) have no solution: clamp s_obs = 0 and count them
    # (`n_mass_width_floor` in the sidecar) — their loaded width is then slightly
    # WIDER than drawn, a conservative, reported residual.
    sigma_used = sigma_scale * sigma_ln_bh
    _fixed_terms = sigma_int**2 + d_alpha**2 + (np.log(mstar_obs / 10.0) * d_beta) ** 2
    _prop_term_sq = np.where(mass_valid, sigma_used**2 - _fixed_terms, 0.0)
    n_mass_width_floor = int(np.count_nonzero(mass_valid & (_prop_term_sq < 0.0)))
    mstar_error_obs = np.where(
        mass_valid,
        np.sqrt(np.clip(_prop_term_sq, 0.0, None)) * mstar_obs / beta,
        mstar_error_safe,
    )
    # The width the inference will recompute from the written row (== sigma_used
    # wherever the solve succeeded).
    _bh_obs_chk, _bh_err_obs_chk = _empiric_stellar_mass_to_BH_mass_relation(
        mstar_obs,  # type: ignore[arg-type]
        mstar_error_obs,  # type: ignore[arg-type]
    )
    sigma_ln_obs = np.asarray(_bh_err_obs_chk, dtype=np.float64) / np.asarray(
        _bh_obs_chk, dtype=np.float64
    )

    # §5.3 mass-edge logging: rows whose error-inflated pruning-window
    # membership (handler._get_pruned_galaxy_catalog mass legs, constants
    # M_SOURCE_FRAME_MIN/MAX) changes between TRUE and OBSERVED BH mass.
    bh_mass_obs = bh_mass * np.exp(delta_ln_bh)
    _bh_err_obs = bh_mass_obs * sigma_ln_obs  # observed-row width (fixed point above)
    in_window_true = (bh_mass + bh_mass_error >= M_SOURCE_FRAME_MIN) & (
        bh_mass - bh_mass_error <= M_SOURCE_FRAME_MAX
    )
    in_window_obs = (bh_mass_obs + _bh_err_obs >= M_SOURCE_FRAME_MIN) & (
        bh_mass_obs - _bh_err_obs <= M_SOURCE_FRAME_MAX
    )
    n_mass_window_crossings = int(np.count_nonzero(mass_valid & (in_window_true != in_window_obs)))

    # --- realized-vs-stored width identity check, per redshift flag
    # ([RATIFY-R2] verification statistic F3: the normalized residual
    # (z_obs_unclipped - z_g)/z_error must have std == sigma_scale per flag).
    z_width_check: dict[str, object] = {}
    normalized_z_residual = np.where(
        z_valid & (z_error > 0),
        (z_obs_unclipped - z_true) / np.where(z_error > 0, z_error, 1.0),
        np.nan,
    )
    for flag_value in sorted({str(f) for f in flags}):
        flag_mask = (flags == flag_value) & np.isfinite(normalized_z_residual)
        n_flag = int(np.count_nonzero(flag_mask))
        z_width_check[flag_value] = {
            "n": n_flag,
            "n_z_floor_clipped": int(np.count_nonzero(clipped_mask & (flags == flag_value))),
            "normalized_residual_std": (
                float(np.std(normalized_z_residual[flag_mask])) if n_flag > 1 else None
            ),
            "expected_std": float(sigma_scale),
        }
    # Normalize by the OBSERVED-row width (the one the inference recomputes and
    # the fixed point above targets), not the parent width — the two differ by
    # up to ~18% per row and normalizing by the parent hid a 7% pull deficit.
    mass_residual = np.where(
        sigma_ln_obs > 0, delta_ln_bh / np.where(sigma_ln_obs > 0, sigma_ln_obs, 1.0), np.nan
    )
    mass_ok = mass_valid & np.isfinite(mass_residual)
    n_mass = int(np.count_nonzero(mass_ok))
    width_check: dict[str, object] = {
        "z_per_flag": z_width_check,
        "mass": {
            "n": n_mass,
            "normalized_residual_std": (
                float(np.std(mass_residual[mass_ok])) if n_mass > 1 else None
            ),
            "expected_std": float(sigma_scale),
        },
    }

    # --- write: replace ONLY the realized fields; every other column keeps its
    # original string. z_error and the flag are untouched (the z width law is
    # scale-free in z: the kernel reads the stored column verbatim, so
    # sigma_kernel == sigma_realized identically). The STELLAR-error column IS
    # rewritten, because the BH-mass width is NOT scale-free in M* — the solve
    # above chooses s_obs so the load-time law reproduces exactly the width the
    # mass scatter was drawn with ([RATIFY-R2]); leaving it untouched measured a
    # 7 % pull deficit.
    z_out = catalog[z_col].to_numpy(dtype=object)
    z_out[z_valid] = z_obs[z_valid]
    catalog[z_col] = z_out
    mstar_out = catalog[mstar_col].to_numpy(dtype=object)
    mstar_out[mass_valid] = mstar_obs[mass_valid]
    catalog[mstar_col] = mstar_out
    mstar_err_out = catalog[mstar_err_col].to_numpy(dtype=object)
    mstar_err_out[mass_valid] = mstar_error_obs[mass_valid]
    catalog[mstar_err_col] = mstar_err_out

    catalog.to_csv(output_csv_path, header=False, index=False)
    width_check["n_mass_width_floor"] = n_mass_width_floor
    return n_rows, n_z_floor_clipped, n_mass_window_crossings, width_check
